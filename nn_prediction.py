import os
import numpy as np
import pandas as pd
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
import sys
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Predictor(nn.Module):
    """Simple feed-forward predictor returning logits."""
    def __init__(self, input_dim, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, classes),
        )

    def forward(self, x):
        # expect x: torch.Tensor of shape (N, input_dim)
        return self.net(x)  # raw logits


def train(model, x, y, val_x=None, val_y=None, epochs=20000, batch_size=64, lr=1e-3, device=None, benchmark_batch_sizes=True, batch_size_candidates=None, optimizer_type='adam', weight_decay=0.0):
    """Train the model.

    Args:
        model: torch.nn.Module
        x: array-like (N, D) training features
        y: array-like (N,) training labels (strings or ints)
        val_x, val_y: optional validation sets
        epochs, batch_size, lr: training hyperparams

    Returns:
        model (trained), history (dict), label_encoder
    """
    device = device or ("cuda" if pt.cuda.is_available() else "cpu")
    model.to(device)

    X = pt.tensor(np.asarray(x), dtype=pt.float32)
    y_arr = np.asarray(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_arr)
    y_tensor = pt.tensor(y_enc, dtype=pt.long)

    # prepare logging
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'prediction_train.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Starting training')

    dataset = TensorDataset(X, y_tensor)

    if benchmark_batch_sizes:
        if batch_size_candidates is None:
            batch_size_candidates = [32, 64, 128, 256]

        def _benchmark(bs_candidates):
            """ Tries to estimate the maximum batch size on a device """
            results = {}
            for bs in bs_candidates:
                try:
                    dl = DataLoader(dataset, batch_size=bs, shuffle=True)
                    model.train()
                    start = time.time()
                    processed = 0
                    # try a small number of batches to estimate throughput
                    max_batches = min(5, len(dl))
                    it = iter(dl)
                    # create a temporary optimizer for the quick benchmark (match chosen optimizer)
                    opt_cls = pt.optim.AdamW if optimizer_type.lower() == 'adamw' else pt.optim.Adam
                    tmp_opt = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
                    for i in range(max_batches):
                        xb, yb = next(it)
                        xb = xb.to(device)
                        yb = yb.to(device)
                        tmp_opt.zero_grad()
                        logits = model(xb)
                        loss = nn.CrossEntropyLoss()(logits, yb)
                        loss.backward()
                        tmp_opt.step()
                        processed += xb.size(0)
                    elapsed = time.time() - start
                    throughput = processed / max(1e-9, elapsed)
                    results[bs] = (throughput, None)
                    logging.info(f'Batch-size benchmark: bs={bs} throughput={throughput:.1f} samples/sec')
                except RuntimeError as e:
                    # catch CUDA OOM
                    if 'out of memory' in str(e).lower():
                        logging.warning(f'Batch-size {bs} caused OOM on device {device}')
                        results[bs] = (0.0, 'OOM')
                        if pt.cuda.is_available():
                            pt.cuda.empty_cache()
                    else:
                        logging.exception(f'Error benchmarking batch-size {bs}: {e}')
                        results[bs] = (0.0, 'ERROR')
            # pick best bs with highest throughput and not OOM
            successful = {bs: v for bs, v in results.items() if v[1] is None}
            if successful:
                best_bs = max(successful.items(), key=lambda kv: kv[1][0])[0]
            else:
                # fallback to provided batch_size
                best_bs = batch_size
            return best_bs, results

        # run benchmark
        try:
            best_bs, bench_results = _benchmark(batch_size_candidates)
            logging.info(f'Chosen best batch size: {best_bs}')
            batch_size = best_bs
        except Exception:
            logging.exception('Batch-size benchmarking failed, using default')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Choose optimizer (Adam or AdamW)
    opt_cls = pt.optim.AdamW if optimizer_type.lower() == 'adamw' else pt.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    val_X = None
    val_y_tensor = None
    if val_x is not None and val_y is not None:
        val_X = pt.tensor(np.asarray(val_x), dtype=pt.float32).to(device)
        val_y_tensor = pt.tensor(le.transform(np.asarray(val_y)), dtype=pt.long).to(device)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Use stdout for tqdm so it updates in most terminals; use explicit nested bars with positions
    for epoch in tqdm(range(1, epochs + 1), desc='Epochs', file=sys.stdout):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_trues = []

        # progress bar for batches (position 0 to allow nested bars to render)
        batch_iter = tqdm(loader, desc=f'Epoch {epoch}', leave=False, file=sys.stdout, position=0)
        for xb, yb in batch_iter:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(yb.cpu().numpy().tolist())

        avg_loss = total_loss / len(dataset)
        train_acc = accuracy_score(all_trues, all_preds)

        val_loss = None
        val_acc = None
        if val_X is not None:
            model.eval()
            with pt.no_grad():
                logits = model(val_X)
                val_loss = criterion(logits, val_y_tensor).item()
                val_preds = logits.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(val_y_tensor.cpu().numpy(), val_preds)

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # log epoch metrics and GPU usage
        gpu_info = None
        if pt.cuda.is_available():
            try:
                dev = pt.device(device)
                gpu_info = {
                    'device_name': pt.cuda.get_device_name(dev.index if dev.type=='cuda' else 0),
                    'memory_allocated': int(pt.cuda.memory_allocated(dev)) if hasattr(pt.cuda, 'memory_allocated') else None,
                    'max_memory_allocated': int(pt.cuda.max_memory_allocated(dev)) if hasattr(pt.cuda, 'max_memory_allocated') else None
                }
            except Exception:
                gpu_info = {'device': device}

        logging.info(f'Epoch {epoch} train_loss={avg_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss} val_acc={val_acc} batch_size={batch_size} gpu_info={gpu_info}')

        # Use tqdm.write so the message doesn't break the progress bar rendering
        try:
            tqdm.write(f"Epoch {epoch}/{epochs} train_loss={avg_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss if val_loss is not None else float('nan'):.4f} val_acc={val_acc if val_acc is not None else float('nan'):.4f}")
        except Exception:
            # fallback
            print(f"Epoch {epoch}/{epochs} train_loss={avg_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss if val_loss is not None else float('nan'):.4f} val_acc={val_acc if val_acc is not None else float('nan'):.4f}")

    return model, history, le


def main():
    os.makedirs('observations', exist_ok=True)

    df = pd.read_csv("/n/fs/vision-mix/jl0796/qcb/qcb455_project/Breast_GSE45827_simulated_10x.csv")

    # Determine id column (to split on unique original sample ids when available)
    id_col = 'original_sample_id' #, 'samples', 'sample_id', 'original']
    # id_col = next((c for c in id_col_candidates if c in df.columns), None)

    label_col_candidates = ['cluster']
    label_col = next((c for c in label_col_candidates if c in df.columns), df.columns[1])

    # Choose input columns: prefer floats (the simulated features are floating-point)
    # numeric_cols = df.select_dtypes(include=[np.floating]).columns.tolist()
    # if len(numeric_cols) == 0:
        # fallback to columns after common metadata columns
    
    numeric_cols = df.columns[4:].tolist()
    # numeric_cols = df.columns[1004:1005].tolist()

    # Create train/validation split by unique original sample id
    if id_col is not None:
        unique_ids = np.asarray(df[id_col].unique())
        # Do an 80/20 split by id.
        np.random.seed(42)
        n_train_ids = max(1, int(len(unique_ids) * 0.8))
        train_ids = np.random.choice(unique_ids, size=n_train_ids, replace=False)
        train_mask = df[id_col].isin(train_ids)
        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[~train_mask].reset_index(drop=True)

    X_train = train_df[numeric_cols].values
    y_train = train_df[label_col].values


    shuffled_indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    X_val = val_df[numeric_cols].values
    y_val = val_df[label_col].values

    input_dim = X_train.shape[1]
    num_classes = len(set(y_train))

    model = Predictor(input_dim, num_classes)
    model, history, le = train(model, X_train, y_train, val_x=X_val, val_y=y_val, epochs=1000)

    # Print final performance
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    print(f"Final train acc: {final_train_acc:.4f} | Final val acc: {final_val_acc:.4f}")

    # Plot training curves
    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='train_loss')
    if any(v is not None for v in history['val_loss']):
        plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='train_acc')
    if any(v is not None for v in history['val_acc']):
        plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.tight_layout()
    out_path = 'observations/prediction_training.png'
    plt.savefig(out_path, dpi=150)
    print(f"Saved training plot to {out_path}")


if __name__ == '__main__':
    main()

