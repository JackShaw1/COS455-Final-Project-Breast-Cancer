import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
import argparse


def train_xgboost_cv(X, y, n_splits=5, n_estimators=100, max_depth=6,
                     learning_rate=0.3, subsample=1.0, colsample_bytree=1.0,
                     n_jobs=-1, random_state=42):
    """Train an XGBoost classifier with 5-fold cross-validation.

    Args:
        X: array-like (N, D) features
        y: array-like (N,) labels
        n_splits: number of cross-validation folds
        n_estimators: number of boosting rounds
        max_depth: maximum depth of each tree
        learning_rate: boosting learning rate
        subsample: subsample ratio of the training instances
        colsample_bytree: subsample ratio of columns when constructing each tree
        n_jobs: number of parallel threads
        random_state: random state for reproducibility

    Returns:
        model (trained on full data), label_encoder, cv_scores
    """
    # Prepare logging
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'xgboost_train.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                       format='%(asctime)s %(levelname)s %(message)s')
    logging.info('Starting XGBoost training with cross-validation')

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    print(f"\n{'='*60}")
    print(f"5-Fold Cross-Validation")
    print(f"{'='*60}")

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = []
    cv_models = []  # Store all models

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_enc), 1):
        print(f"\nFold {fold_idx}/{n_splits}")
        print("-" * 40)

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_enc[train_idx], y_enc[val_idx]

        # Create XGBoost classifier
        if num_classes > 2:
            xgb_model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective='multi:softmax',
                num_class=num_classes,
                n_jobs=n_jobs,
                random_state=random_state,
                eval_metric='mlogloss'
            )
        else:
            xgb_model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                objective='binary:logistic',
                n_jobs=n_jobs,
                random_state=random_state,
                eval_metric='logloss'
            )

        # Train on fold
        xgb_model.fit(X_train_fold, y_train_fold, verbose=False)

        # Evaluate on validation fold
        val_preds = xgb_model.predict(X_val_fold)
        fold_acc = accuracy_score(y_val_fold, val_preds)
        cv_scores.append(fold_acc)
        cv_models.append(xgb_model)  # Save the model

        print(f"  Validation accuracy: {fold_acc:.4f}")
        logging.info(f"Fold {fold_idx} validation accuracy: {fold_acc:.4f}")

    # Print CV summary and select best model
    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"Individual fold accuracies: {[f'{s:.4f}' for s in cv_scores]}")
    logging.info(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Use the best model from CV (the one with highest validation accuracy)
    best_fold_idx = np.argmax(cv_scores)
    final_model = cv_models[best_fold_idx]
    print(f"\nUsing best model from Fold {best_fold_idx + 1} (validation accuracy: {cv_scores[best_fold_idx]:.4f})")
    logging.info(f"Best model from Fold {best_fold_idx + 1} with accuracy: {cv_scores[best_fold_idx]:.4f}")

    # Evaluate best model on full dataset
    print(f"\n{'='*60}")
    print(f"Evaluating best model on full dataset...")
    print(f"{'='*60}")

    full_preds = final_model.predict(X)
    full_acc = accuracy_score(y_enc, full_preds)
    print(f"Accuracy on full dataset: {full_acc:.4f}")
    logging.info(f"Full dataset accuracy: {full_acc:.4f}")

    # Detailed classification report on full dataset
    print("\nClassification Report (Full Dataset):")
    report = classification_report(y_enc, full_preds,
                                  target_names=le.classes_.astype(str))
    print(report)
    logging.info(f"Classification Report:\n{report}")

    metrics = {
        'cv_scores': cv_scores,
        'mean_cv_acc': np.mean(cv_scores),
        'std_cv_acc': np.std(cv_scores),
        'full_acc': full_acc,
        'full_preds': full_preds,
        'y_enc': y_enc
    }

    return final_model, le, metrics


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


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (XGBoost)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train XGBoost classifier on augmented data")
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Path to augmented dataset CSV file (e.g., augmented_data/Breast_GSE45827_simulated_50x_normalized.csv)')
    parser.add_argument('-o', '--original-data', type=str,
                        default='/n/fs/vision-mix/jl0796/qcb/qcb455_project/renamed_subtyping_by_clustering_new.csv',
                        help='Path to original data with labels')
    args = parser.parse_args()

    # Extract dataset name from path for output naming
    dataset_basename = os.path.basename(args.dataset).replace('.csv', '')
    output_dir = f'xgb_observations_{dataset_basename}'
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading augmented data from: {args.dataset}")
    augment_df = pd.read_csv(args.dataset)
    print(f"Loading original data from: {args.original_data}")
    df = pd.read_csv(args.original_data)

    id_col = 'original_sample_id'

    # Find label column (cluster)
    label_col_candidates = ['cluster']
    label_col = next((c for c in label_col_candidates if c in df.columns), df.columns[2])
    print(f"Using label column: {label_col}")

    # Features are all columns after the first 4
    numeric_cols = df.columns[4:].tolist()
    print(f"Number of features: {len(numeric_cols)}")

    # Prepare training data (use augmented data)
    if id_col in augment_df.columns:
        X_train = augment_df[numeric_cols].values
        y_train = augment_df[label_col].values

        # Shuffle training data
        shuffled_indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        print(f"\nAugmented training set size: {len(X_train)} rows")
    else:
        print("Warning: 'original_sample_id' not found in augmented data. Exiting.")
        return

    # Get number of classes
    temp_le = LabelEncoder()
    temp_le.fit(y_train)
    num_classes = len(temp_le.classes_)
    print(f"Found {num_classes} classes: {temp_le.classes_}")

    # Train XGBoost with 5-fold CV
    print("\n" + "="*60)
    print("Training XGBoost Classifier with 5-Fold Cross-Validation")
    print("="*60)

    xgb_model, le, metrics = train_xgboost_cv(
        X_train, y_train,
        n_splits=5,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )

    # Print final performance
    print("\n" + "="*60)
    print("Final Performance Summary")
    print("="*60)
    print(f"Mean CV accuracy: {metrics['mean_cv_acc']:.4f} (+/- {metrics['std_cv_acc']:.4f})")
    print(f"Full dataset accuracy: {metrics['full_acc']:.4f}")

    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(
        xgb_model,
        numeric_cols,
        top_n=20,
        save_path=os.path.join(output_dir, 'xgb_feature_importance.png')
    )

    # Plot confusion matrix for full dataset
    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        metrics['y_enc'],
        metrics['full_preds'],
        labels=le.classes_.astype(str),
        save_path=os.path.join(output_dir, 'xgb_confusion_matrix.png')
    )

    # Save the trained model
    model_save_path = os.path.join(output_dir, 'xgboost_model.joblib')
    joblib.dump(xgb_model, model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # Save label encoder
    le_save_path = os.path.join(output_dir, 'label_encoder.joblib')
    joblib.dump(le, le_save_path)
    print(f"Label encoder saved to {le_save_path}")

    # Predict on the full original dataset
    print("\n" + "="*60)
    print("Generating predictions for the full original dataset")
    print("="*60)
    X_full = df[numeric_cols].values
    preds_encoded = xgb_model.predict(X_full)
    preds_decoded = le.inverse_transform(preds_encoded)

    df['xgb_prediction'] = preds_decoded

    # Also get prediction probabilities
    pred_probs = xgb_model.predict_proba(X_full)
    for i, class_label in enumerate(le.classes_):
        df[f'xgb_prob_{class_label}'] = pred_probs[:, i]

    # Save predictions
    csv_out_path = os.path.join(output_dir, 'xgb_predictions_full_dataset.csv')
    df.to_csv(csv_out_path, index=False)
    print(f"Saved full dataset with predictions to {csv_out_path}")

    # Compare predictions to actual labels (if they exist in the original data)
    if label_col in df.columns:
        y_full_enc = le.transform(df[label_col])
        full_original_acc = accuracy_score(y_full_enc, preds_encoded)
        print(f"Accuracy on full original dataset: {full_original_acc:.4f}")


if __name__ == '__main__':
    main()
