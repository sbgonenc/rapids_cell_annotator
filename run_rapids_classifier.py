#!/usr/bin/env python3
"""
CLI entrypoints for training and prediction.
"""

import os
import sys
import argparse
import json
import pandas as pd
import anndata as ad
from rich.console import Console
from rich.table import Table


console = Console()


def train_command(args):
    console.print(f"[bold blue]Training Cell Type Classifier[/bold blue]")
    if args.cv_folds < 0:
        console.print("[red]Error: --cv-folds must be >= 0[/red]")
        sys.exit(2)
    if args.test_split is None or args.test_split < 0:
        console.print("[red]Error: --test-split must be >= 0[/red]")
        sys.exit(2)
    if args.validation_split is None or args.validation_split < 0:
        console.print("[red]Error: --validation-split must be >= 0[/red]")
        sys.exit(2)
    if not os.path.exists(args.input):
        console.print(f"[red]Error: Input file {args.input} not found[/red]")
        sys.exit(1)
    console.print(f"[blue]Loading training data: {args.input}[/blue]")
    try:
        adata_train = ad.read_h5ad(args.input)
    except Exception as e:
        console.print(f"[red]Error loading {args.input}: {e}[/red]")
        sys.exit(1)
    if args.label_key not in adata_train.obs.columns:
        console.print(f"[red]Error: Label key '{args.label_key}' not found in .obs[/red]")
        console.print(f"Available columns: {', '.join(adata_train.obs.columns)}")
        sys.exit(1)
    if args.feature_key not in adata_train.obsm.keys():
        console.print(f"[red]Error: Feature key '{args.feature_key}' not found in .obsm[/red]")
        console.print(f"Available keys: {', '.join(adata_train.obsm.keys())}")
        sys.exit(1)



    from utils import prepare_adata
    feature_data, label_data = prepare_adata(
        adata=adata_train,
        label_key=args.label_key,
        feature_key=args.feature_key
    )

    from config import classifier_default_values
    from rapids import RAPIDS_Classifier

    classifier = RAPIDS_Classifier()
    os.makedirs(args.outdir, exist_ok=True)
    classifier.init_classifier(model_type=args.model_type, **classifier_default_values[args.model_type])

    cv_enabled = args.cv_folds >= 2
    test_enabled = args.test_split > 0
    if test_enabled and (args.validation_split + args.test_split >= 1):
        console.print("[red]Error: --validation-split + --test-split must be < 1[/red]")
        sys.exit(2)
    if cv_enabled:
        console.print("[yellow]Warning: --validation-split is ignored when cross-validation is enabled (--cv-folds >= 2).[/yellow]")
        cv_runner = getattr(classifier, "cross_validate", None)
        if cv_runner is None:
            console.print(
                "[red]Error: cross-validation is not implemented yet. "
                "TODO(Task 2): implement RAPIDS_Classifier.cross_validate(...) and wire results into the CLI.[/red]"
            )
            sys.exit(2)

        # Optional held-out test set for final evaluation.
        # When enabled, cross-validation runs on the remaining data (train+val)
        # and the model is evaluated on test exactly once after final training.
        features_trainval = feature_data
        labels_trainval = label_data
        features_test = None
        labels_test = None
        if test_enabled:
            import cupy as cp
            from rapids import cu_train_test_split

            if cu_train_test_split is None:
                console.print("[red]Error: RAPIDS dependencies not available (cuml/cupy).[/red]")
                sys.exit(2)

            idx = cp.arange(len(label_data))
            trainval_idx, test_idx = cu_train_test_split(
                idx, test_size=args.test_split, random_state=classifier.random_state
            )
            trainval_idx = cp.asnumpy(trainval_idx)
            test_idx = cp.asnumpy(test_idx)

            features_trainval = feature_data[trainval_idx]
            labels_trainval = label_data[trainval_idx]
            features_test = feature_data[test_idx]
            labels_test = label_data[test_idx]

        try:
            cv_results = cv_runner(
                features_data=features_trainval,
                label_data=labels_trainval,
                k=args.cv_folds,
                seed=args.cv_seed,
                model_type=args.model_type,
                model_params=classifier_default_values[args.model_type],
            )
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(2)

        # Write CV summary JSON (schema per spec)
        cv_out = {
            "k": int(args.cv_folds),
            "seed": int(args.cv_seed),
            "model_type": str(args.model_type),
            "feature_key": str(args.feature_key),
            "label_key": str(args.label_key),
            "n_samples": int(cv_results.get("n_samples", len(labels_trainval) if labels_trainval is not None else 0)),
            "n_features": int(cv_results.get("n_features", feature_data.shape[1] if hasattr(feature_data, "shape") else 0)),
            "n_classes": int(cv_results.get("n_classes", len(set(labels_trainval)) if labels_trainval is not None else 0)),
            "class_counts": cv_results.get("class_counts", {}),
            "fold_acc": cv_results.get("fold_acc", []),
            "mean_acc": float(cv_results.get("mean_acc")),
            "std_acc": float(cv_results.get("std_acc")),
        }
        with open(os.path.join(args.outdir, "cv_results.json"), "w") as fh:
            json.dump(cv_out, fh)

        # Train ONE final model on all samples and save artifacts.
        trainer = getattr(classifier, "train_full", None)
        if trainer is None:
            console.print("[red]Error: RAPIDS_Classifier.train_full(...) is missing[/red]")
            sys.exit(2)
        classifier.train_full(
            features_data=features_trainval,
            label_data=labels_trainval,
            model_type=args.model_type,
            model_params=classifier_default_values[args.model_type],
        )

        # Evaluate once on held-out test set (if enabled), without including test features in training.
        if test_enabled:
            test_acc = classifier.evaluate_decoded_accuracy(features_test, labels_test)
            classifier.train_stats["test_acc"] = test_acc

        # Keep training_acc.json backward compatible + add CV summary fields.
        classifier.train_stats.update(
            {
                "cv_mean_acc": cv_out["mean_acc"],
                "cv_std_acc": cv_out["std_acc"],
                "cv_folds": int(args.cv_folds),
            }
        )
    else:
        if test_enabled:
            classifier.process_training_with_test(
                features_data=feature_data,
                label_data=label_data,
                validation_split=args.validation_split,
                test_split=args.test_split,
            )
        else:
            classifier.process_training(
                features_data=feature_data,
                label_data=label_data,
                test_split=args.validation_split,
            )

    classifier.save_status(args.outdir)


    table = Table(title="Training Summary")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Model Type", args.model_type)
    table.add_row("Training Samples", f"{len(adata_train):,}")
    table.add_row("Features", str(adata_train.obsm[args.feature_key].shape[1]))
    table.add_row("Cell Types", str(len(set(label_data))))
    table.add_row("Feature Key", args.feature_key)
    table.add_row("Label Key", args.label_key)
    #table.add_row("Training Accuracy", classifier.train_stats["training_acc"])
    #table.add_row("Validation Accuracy", classifier.train_stats["test_acc"])
    console.print(table)

    console.print(classifier.train_stats)


def predict_command(args):
    console.print(f"[bold blue]Cell Type Prediction[/bold blue]")
    if not os.path.exists(args.model):
        console.print(f"[red]Error: Model file {args.model} not found[/red]")
        sys.exit(1)
    if not os.path.exists(args.input):
        console.print(f"[red]Error: Input file {args.input} not found[/red]")
        sys.exit(1)
    console.print(f"[blue]Loading model: {args.model}[/blue]")

    from rapids import RAPIDS_Classifier

    classifier = RAPIDS_Classifier()
    classifier.load_classifier(args.model)
    classifier.load_encoder(args.encoder)
    
    #info = classifier.get_model_info()
    #console.print(f"[green]Model: {info['model_type']} ({info['backend']} backend)[/green]")
    #console.print(f"[green]Classes: {info['n_classes']} cell types[/green]")
    #console.print(f"[blue]Loading test data: {args.input}[/blue]")
    try:
        adata_test = ad.read_h5ad(args.input)
    except Exception as e:
        console.print(f"[red]Error loading test data: {e}[/red]")
        sys.exit(1)
    
    

    feature_key = args.feature_key #or classifier.feature_names
    if feature_key not in adata_test.obsm.keys():
        console.print(f"[red]Error: Feature key '{feature_key}' not found in test data[/red]")
        console.print(f"Available keys: {', '.join(adata_test.obsm.keys())}")
        console.print(f"Model was trained with: {classifier.feature_names}")
        sys.exit(1)
    
    from utils import prepare_adata
    features, _ = prepare_adata(adata_test, feature_key=args.feature_key, label_key=None)

    try:
        predictions = classifier.predict(features)
    except Exception as e:
        console.print(f"[red]Error during prediction: {e}[/red]")
        sys.exit(1)

    decoded_preds = classifier.get_decoded_labels(predictions).to_numpy()

    pred_df = pd.DataFrame(
        { 'label_pred': decoded_preds },
        index=adata_test.obs.index
    )

    output = ad.AnnData(
        obs=pred_df,
        uns={
            'dataset_id': adata_test.uns.get("dataset_id", "unknown"),
            'normalization_id': adata_test.uns.get("normalization_id", "unknown"),
            'model_path': args.model,
            #'model_type': info['model_type'],
            'prediction_timestamp': pd.Timestamp.now().isoformat(),
        },
    )
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    console.print(f"[blue]Writing results: {args.output}[/blue]")
    output.write_h5ad(args.output, compression="gzip")
    table = Table(title="Prediction Summary")
    table.add_column("Cell Type", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")
    vc = pred_df['label_pred'].value_counts().sort_values(ascending=False)
    total = len(pred_df)
    for cell_type, count in vc.items():
        percentage = f"{(count/total)*100:.1f}%"
        table.add_row(str(cell_type), f"{count:,}", percentage)
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="RAPIDS-Accelerated Cell Type Annotation CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    train_parser = subparsers.add_parser('train', help='Train a cell type classifier')
    train_parser.add_argument('-i', '--input', required=True, help='Input training h5ad file')
    train_parser.add_argument('-o', '--outdir', required=True, help='Output folder to save classifier, data, encoder')
    train_parser.add_argument('--label-key', default='label', help='Label column name in .obs (default: label)')
    train_parser.add_argument('--feature-key', default='X_pca', help='Feature key in .obsm (default: X_pca)')
    train_parser.add_argument('-m', '--model-type', default='logistic_regression', choices=['logistic_regression', 'random_forest', 'svm'], help='Model type (default: logistic_regression)')
    train_parser.add_argument('--validation-split', type=float, default=0.1, help='Validation split fraction (default: 0.1, 0 = no validation)')
    train_parser.add_argument('--test-split', type=float, default=0.0, help='Held-out test split fraction (default: 0; 0 disables test set)')
    train_parser.add_argument('--cv-folds', type=int, default=0, help='Cross-validation folds (default: 0; 0/1 disables CV; >=2 enables CV)')
    train_parser.add_argument('--cv-seed', type=int, default=42, help='Random seed for cross-validation (default: 42)')
    train_parser.add_argument('--max-iter', type=int, help='Maximum iterations for logistic regression')
    train_parser.add_argument('--C', type=float, help='Regularization parameter for logistic regression/SVM')
    train_parser.add_argument('--n-estimators', type=int, help='Number of trees for random forest')


    predict_parser = subparsers.add_parser('predict', help='Predict cell types')
    predict_parser.add_argument('-i', '--input', required=True, help='Input test h5ad file')
    predict_parser.add_argument('-m', '--model', required=True, help='Trained classifier file (.pkl)')
    predict_parser.add_argument('-e', '--encoder', required=True, help='Trained model file encoder (.pkl)')
    predict_parser.add_argument('-o', '--output', required=True, help='Output h5ad file with predictions')
    predict_parser.add_argument('--feature-key', help='Feature key in .obsm (uses training key if not specified)')
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)


if __name__ == "__main__":
    main()


