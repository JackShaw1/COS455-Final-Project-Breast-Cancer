import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import argparse


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance from the XGBoost model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Select top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Most Important Features (XGBoost)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")

    return top_indices, top_importances


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None, title='Confusion Matrix (Validation Set - No Data Leakage)'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    return cm


def plot_averaged_confusion_matrix(cm_avg, labels, n_folds=5, save_path=None):
    """Plot averaged confusion matrix from K-fold cross-validation."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_avg, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Averaged Confusion Matrix ({n_folds}-Fold ID-Based CV - No Data Leakage)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Averaged confusion matrix saved to {save_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train XGBoost classifier with ID-based split (no data leakage)")
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Path to augmented dataset CSV file (e.g., augmented_data/Breast_GSE45827_simulated_50x_normalized.csv)')
    parser.add_argument('-o', '--original-data', type=str,
                        default='/n/fs/vision-mix/jl0796/qcb/qcb455_project/renamed_subtyping_by_clustering_new.csv',
                        help='Path to original data with labels')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for shuffling IDs before K-fold split (default: 42)')
    parser.add_argument('--n-splits', type=int, default=5,
                        help='Number of folds for K-fold cross-validation on IDs (default: 5)')
    args = parser.parse_args()

    # Extract dataset name from path for output naming
    dataset_basename = os.path.basename(args.dataset).replace('.csv', '')
    output_dir = f'final_xgb_id_split_{dataset_basename}'
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'xgboost_train_id_split.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                       format='%(asctime)s %(levelname)s %(message)s',
                       force=True)
    logger = logging.getLogger(__name__)
    logger.info(f'='*60)
    logger.info(f'Starting {args.n_splits}-fold ID-based cross-validation')
    logger.info(f'Random seed: {args.random_seed}')
    logger.info(f'='*60)

    # Load data
    print(f"Loading augmented data from: {args.dataset}")
    augment_df = pd.read_csv(args.dataset)
    print(f"Loading original data from: {args.original_data}")
    original_df = pd.read_csv(args.original_data)

    id_col = 'original_sample_id'

    # Verify ID column exists
    if id_col not in augment_df.columns:
        print(f"Error: '{id_col}' not found in augmented data. Available columns: {augment_df.columns.tolist()}")
        return
    if id_col not in original_df.columns:
        print(f"Error: '{id_col}' not found in original data. Available columns: {original_df.columns.tolist()}")
        return

    # Find label column (cluster)
    label_col_candidates = ['cluster']
    label_col = next((c for c in label_col_candidates if c in original_df.columns), original_df.columns[2])
    print(f"Using label column: {label_col}")

    # Features are all columns after the first 4
    numeric_cols = original_df.columns[4:].tolist()
    print(f"Number of features: {len(numeric_cols)}")

    # Train XGBoost with K-fold ID-based cross-validation
    print("\n" + "="*60)
    print(f"Training XGBoost with {args.n_splits}-Fold ID-Based Cross-Validation")
    print("="*60)

    # Create a global label encoder using all data
    label_encoder = LabelEncoder()
    label_encoder.fit(original_df[label_col])
    num_classes = len(label_encoder.classes_)
    print(f"\nFound {num_classes} classes: {label_encoder.classes_}")

    # Split IDs into K folds
    unique_ids = original_df[id_col].unique()
    np.random.seed(args.random_seed)
    shuffled_ids = np.random.permutation(unique_ids)
    id_folds = np.array_split(shuffled_ids, args.n_splits)

    print(f"Total unique IDs: {len(unique_ids)}")
    print(f"IDs per fold: {[len(fold) for fold in id_folds]}")

    all_models = []
    all_metrics = []
    all_confusion_matrices = []
    all_val_accs = []

    for fold_idx in range(args.n_splits):
        print(f"\n{'#'*60}")
        print(f"FOLD {fold_idx + 1}/{args.n_splits}")
        print(f"{'#'*60}")

        # Validation IDs = current fold, Training IDs = all other folds
        val_ids = id_folds[fold_idx]
        train_ids = np.concatenate([id_folds[i] for i in range(args.n_splits) if i != fold_idx])

        print(f"Training IDs: {len(train_ids)}, Validation IDs: {len(val_ids)}")

        # Create train/val splits
        train_mask = augment_df[id_col].isin(train_ids)
        val_mask = original_df[id_col].isin(val_ids)

        train_df = augment_df[train_mask].reset_index(drop=True)
        val_df = original_df[val_mask].reset_index(drop=True)

        print(f"Augmented training samples: {len(train_df)}")
        print(f"Original validation samples: {len(val_df)}")

        # Prepare features and labels
        X_train = train_df[numeric_cols].values
        y_train = train_df[label_col].values
        X_val = val_df[numeric_cols].values
        y_val = val_df[label_col].values

        # Encode labels using global label encoder
        y_train_enc = label_encoder.transform(y_train)
        y_val_enc = label_encoder.transform(y_val)

        # Create and train XGBoost model
        if num_classes > 2:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softmax',
                num_class=num_classes,
                n_jobs=-1,
                random_state=args.random_seed,
                eval_metric='mlogloss'
            )
        else:
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                n_jobs=-1,
                random_state=args.random_seed,
                eval_metric='logloss'
            )

        xgb_model.fit(X_train, y_train_enc, eval_set=[(X_val, y_val_enc)], verbose=False)

        # Evaluate
        train_preds = xgb_model.predict(X_train)
        val_preds = xgb_model.predict(X_val)
        train_acc = accuracy_score(y_train_enc, train_preds)
        val_acc = accuracy_score(y_val_enc, val_preds)

        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")

        logger.info(f'Fold {fold_idx + 1}: Train acc = {train_acc:.4f}, Val acc = {val_acc:.4f}')

        # Store results
        metrics = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_preds': train_preds,
            'val_preds': val_preds,
            'y_train_enc': y_train_enc,
            'y_val_enc': y_val_enc,
            'n_train_ids': len(train_ids),
            'n_val_ids': len(val_ids),
            'train_size': len(train_df),
            'val_size': len(val_df)
        }

        all_models.append(xgb_model)
        all_metrics.append(metrics)
        all_val_accs.append(val_acc)

        # Compute confusion matrix for this fold (ensure consistent shape)
        cm = confusion_matrix(y_val_enc, val_preds, labels=range(num_classes))
        all_confusion_matrices.append(cm)

    # Calculate average confusion matrix
    cm_avg = np.mean(all_confusion_matrices, axis=0)

    # Print final performance summary
    print("\n" + "="*60)
    print(f"{args.n_splits}-FOLD CROSS-VALIDATION RESULTS (No Data Leakage)")
    print("="*60)
    print(f"Mean validation accuracy: {np.mean(all_val_accs):.4f} (+/- {np.std(all_val_accs):.4f})")
    print(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in all_val_accs]}")
    print(f"\nNote: Each ID appeared in validation exactly once across all folds")

    # Select best model based on validation accuracy
    best_fold_idx = np.argmax(all_val_accs)
    best_model = all_models[best_fold_idx]
    best_metrics = all_metrics[best_fold_idx]

    print(f"\nBest model from Fold {best_fold_idx + 1} with validation accuracy: {all_val_accs[best_fold_idx]:.4f}")

    # Log final results
    logger.info(f'='*60)
    logger.info(f'{args.n_splits}-FOLD CV RESULTS')
    logger.info(f'='*60)
    logger.info(f'Mean validation accuracy: {np.mean(all_val_accs):.4f} (+/- {np.std(all_val_accs):.4f})')
    logger.info(f'Individual fold accuracies: {[f"{acc:.4f}" for acc in all_val_accs]}')
    logger.info(f'Best fold: {best_fold_idx + 1} with validation accuracy: {all_val_accs[best_fold_idx]:.4f}')

    # Plot feature importance (from best model)
    print("\nPlotting feature importance (from best model)...")
    plot_feature_importance(
        best_model,
        numeric_cols,
        top_n=20,
        save_path=os.path.join(output_dir, 'xgb_feature_importance.png')
    )

    # Plot confusion matrix for best fold
    print("Plotting confusion matrix (best fold)...")
    plot_confusion_matrix(
        best_metrics['y_val_enc'],
        best_metrics['val_preds'],
        labels=label_encoder.classes_.astype(str),
        save_path=os.path.join(output_dir, 'xgb_confusion_matrix_best_fold.png'),
        title=f'Confusion Matrix - Best Fold {best_fold_idx + 1} (Val Acc: {all_val_accs[best_fold_idx]:.4f})'
    )

    # Plot AVERAGED confusion matrix
    print("Plotting averaged confusion matrix...")
    plot_averaged_confusion_matrix(
        cm_avg,
        labels=label_encoder.classes_.astype(str),
        n_folds=args.n_splits,
        save_path=os.path.join(output_dir, 'xgb_confusion_matrix_averaged.png')
    )

    # Save the best model
    model_save_path = os.path.join(output_dir, 'xgboost_best_model.joblib')
    joblib.dump(best_model, model_save_path)
    print(f"\nBest model saved to {model_save_path}")

    # Save label encoder
    le_save_path = os.path.join(output_dir, 'label_encoder.joblib')
    joblib.dump(label_encoder, le_save_path)
    print(f"Label encoder saved to {le_save_path}")

    # Predict on the full original dataset using best model
    print("\n" + "="*60)
    print("Generating predictions for the full original dataset (using best model)")
    print("="*60)
    X_full = original_df[numeric_cols].values
    preds_encoded = best_model.predict(X_full)
    preds_decoded = label_encoder.inverse_transform(preds_encoded)

    original_df_copy = original_df.copy()
    original_df_copy['xgb_prediction'] = preds_decoded

    # Also get prediction probabilities
    pred_probs = best_model.predict_proba(X_full)
    for i, class_label in enumerate(label_encoder.classes_):
        original_df_copy[f'xgb_prob_{class_label}'] = pred_probs[:, i]

    # Save predictions
    csv_out_path = os.path.join(output_dir, 'xgb_predictions_full_dataset.csv')
    original_df_copy.to_csv(csv_out_path, index=False)
    print(f"Saved full dataset with predictions to {csv_out_path}")

    # Calculate accuracy on full original dataset
    if label_col in original_df.columns:
        y_full_enc = label_encoder.transform(original_df[label_col])
        full_original_acc = accuracy_score(y_full_enc, preds_encoded)
        print(f"\nAccuracy on full original dataset (best model): {full_original_acc:.4f}")

        # Save metrics summary
        metrics_summary = {
            'n_splits': args.n_splits,
            'mean_validation_accuracy': np.mean(all_val_accs),
            'std_validation_accuracy': np.std(all_val_accs),
            'best_fold_index': best_fold_idx + 1,
            'best_validation_accuracy': all_val_accs[best_fold_idx],
            'best_train_accuracy': best_metrics['train_acc'],
            'full_dataset_accuracy': full_original_acc,
            'avg_n_train_ids': best_metrics['n_train_ids'],
            'avg_n_val_ids': best_metrics['n_val_ids'],
            'avg_train_samples': best_metrics['train_size'],
            'avg_val_samples': best_metrics['val_size'],
            'total_original_samples': len(original_df),
            'n_features': len(numeric_cols),
            'n_classes': len(label_encoder.classes_)
        }

        metrics_path = os.path.join(output_dir, 'metrics_summary.txt')
        with open(metrics_path, 'w') as f:
            f.write("XGBoost K-Fold ID-Based Cross-Validation - Metrics Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"K-Fold CV Strategy: {metrics_summary['n_splits']}-fold on original_sample_id\n")
            f.write(f"Each ID appears in validation exactly once\n\n")
            f.write(f"Mean validation accuracy: {metrics_summary['mean_validation_accuracy']:.4f} (+/- {metrics_summary['std_validation_accuracy']:.4f})\n")
            f.write(f"Individual fold accuracies: {[f'{acc:.4f}' for acc in all_val_accs]}\n\n")
            f.write(f"Best fold: {metrics_summary['best_fold_index']}\n")
            f.write(f"Best validation accuracy: {metrics_summary['best_validation_accuracy']:.4f}\n")
            f.write(f"Best train accuracy: {metrics_summary['best_train_accuracy']:.4f}\n\n")
            for key, value in metrics_summary.items():
                if key not in ['n_splits', 'mean_validation_accuracy', 'std_validation_accuracy',
                              'best_fold_index', 'best_validation_accuracy', 'best_train_accuracy']:
                    f.write(f"{key}: {value}\n")
        print(f"Metrics summary saved to {metrics_path}")

        # Save averaged confusion matrix as CSV
        cm_csv_path = os.path.join(output_dir, 'averaged_confusion_matrix.csv')
        cm_df = pd.DataFrame(cm_avg,
                           index=label_encoder.classes_.astype(str),
                           columns=label_encoder.classes_.astype(str))
        cm_df.to_csv(cm_csv_path)
        print(f"Averaged confusion matrix saved to {cm_csv_path}")


if __name__ == '__main__':
    main()
