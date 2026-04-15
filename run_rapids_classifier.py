#!/usr/bin/env python3
"""
CLI entrypoints for training and prediction.
"""

import os
import sys
import argparse
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

    ##E Check anndata inputs
    import anndata as ad
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


    from lib.utils import prepare_adata
    from config import classifier_default_values
    from lib.rapids import RAPIDS_Classifier

    feature_data, label_data = prepare_adata(
        adata=adata_train,
        label_key=args.label_key,
        feature_key=args.feature_key
    )

    

    classifier = RAPIDS_Classifier()
    os.makedirs(args.outdir, exist_ok=True)

    optimisation_results_list = []
    for model_type, default_params in classifier_default_values.items():
        classifier.init_classifier(model_type=model_type, **default_params)
        optimization_results = classifier.holdout_test_set_cv(
            features_data=feature_data,
            label_data=label_data,
            test_size=args.test_split,
            n_splits=args.cv_folds,
            classifier_params=default_params,
            param_grid=None,
            use_optuna=True,
            n_trials=args.optimization_trials,
            optuna_direction="maximize",
        )    
        print(f"Model type: {model_type}")
        optimization_results["model_type"] = model_type
        print(f"Optimization results: {optimization_results}")
        optimisation_results_list.append(optimization_results)

    ### Get the best model type based on the optimization results
    best_model_dict = max(optimisation_results_list, key=lambda x: x["test_score"])
    best_model_type = best_model_dict["model_type"]
    best_model_params = best_model_dict["best_params"]
    best_model_test_score = best_model_dict["test_score"]
    best_model_cv_scores = best_model_dict["cv_scores"]
    best_model_avg_cv_score = best_model_dict["avg_cv_score"]
    print(f"Best model type: {best_model_type}")
    print(f"Best model params: {best_model_params}")
    print(f"Best model test score: {best_model_test_score}")
    print(f"Best model cv scores: {best_model_cv_scores}")
    print(f"Best model avg cv score: {best_model_avg_cv_score}")

    ### Train the best model on the full dataset
    classifier.init_classifier(model_type=best_model_type, **best_model_params)
    classifier.train_full(
        features_data=feature_data,
        label_data=label_data,
        model_type=best_model_type,
        model_params=best_model_params,
    )
    classifier.save_status(args.outdir)

    import json
    with open(os.path.join(args.outdir, "optimization_results.json"), "w") as fh:
        json.dump(best_model_dict, fh)

    sys.exit(0)


def predict_command(args):
    console.print(f"[bold blue]Cell Type Prediction[/bold blue]")
    if not os.path.exists(args.model):
        console.print(f"[red]Error: Model file {args.model} not found[/red]")
        sys.exit(1)
    if not os.path.exists(args.input):
        console.print(f"[red]Error: Input file {args.input} not found[/red]")
        sys.exit(1)
    console.print(f"[blue]Loading model: {args.model}[/blue]")

    from lib.rapids import RAPIDS_Classifier
    from lib.utils import prepare_adata
    import anndata as ad
    import pandas as pd


    classifier = RAPIDS_Classifier()
    classifier.load_classifier(args.model)
    classifier.load_encoder(args.encoder)
    
    try:
        adata_test = ad.read_h5ad(args.input)
    except Exception as e:
        console.print(f"[red]Error loading test data: {e}[/red]")
        sys.exit(1)
    
    
    feature_key = args.feature_key #or classifier.feature_names
    if feature_key not in adata_test.obsm.keys():
        console.print(f"[red]Error: Feature key '{feature_key}' not found in test data[/red]")
        console.print(f"Available keys: {', '.join(adata_test.obsm.keys())}")
        #console.print(f"Model was trained with: {classifier.feature_names}")
        sys.exit(1)
    
    features, _ = prepare_adata(adata_test, feature_key=args.feature_key, label_key=None)

    ### transform the features to model's 

    try:
        predictions, pred_probs = classifier.get_predictions(features)
    except Exception as e:
        console.print(f"[red]Error during prediction: {e}[/red]")
        sys.exit(1)

    decoded_preds = classifier.get_decoded_labels(predictions).to_numpy()

    pred_df = pd.DataFrame(
        { 
            'label_pred': decoded_preds,
            'label_pred_prob': pred_probs
         },
        index=adata_test.obs.index
    )

    output = ad.AnnData(
        obs=pred_df,
        uns={
            'dataset_id': adata_test.uns.get("dataset_id", "unknown"),
            'normalization_id': adata_test.uns.get("normalization_id", "unknown"),
            'model_path': args.model,
            'model_type': type(classifier.classifier).__name__,
            'prediction_timestamp': pd.Timestamp.now().isoformat(),
        },
    )
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    console.print(f"[blue]Writing results: {args.output}[/blue]")
    
    table = Table(title="Prediction Summary")
    table.add_column("Cell Type", style="cyan")
    table.add_column("Count", style="magenta")
    table.add_column("Percentage", style="green")
    vc = pred_df['label_pred'].value_counts().sort_values(ascending=False)
    total = len(pred_df)
    cell_percentages = {}
    for cell_type, count in vc.items():
        percentage = f"{(count/total)*100:.1f}%"
        cell_percentages[cell_type] = percentage
        table.add_row(str(cell_type), f"{count:,}", percentage)
    console.print(table)

    output.uns["cell_percentages"] = cell_percentages
    output.write_h5ad(args.output, compression="gzip")
    output.obs.to_csv(args.output.replace(".h5ad", ".csv"))
    import json
    with open(args.output.replace(".h5ad", ".json"), "w") as fh:
        json.dump(obj=output.uns, fp=fh, indent=4)



def main():
    parser = argparse.ArgumentParser(description="RAPIDS-Accelerated Cell Type Annotation CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    train_parser = subparsers.add_parser('train', help='Train a cell type classifier')
    train_parser.add_argument('-i', '--input', required=True, help='Input training h5ad file')
    train_parser.add_argument('-o', '--outdir', required=True, help='Output folder to save classifier, data, encoder')
    train_parser.add_argument('--label-key', default='label', help='Label column name in .obs (default: label)')
    train_parser.add_argument('--feature-key', default='X_pca', help='Feature key in .obsm (default: X_pca)')
    train_parser.add_argument('--optimization-trials', type=int, default=10, help='Number of Optuna trials (default: 10)')
    train_parser.add_argument('--validation-split', type=float, default=0.1, help='Validation split fraction (default: 0.1, 0 = no validation)')
    train_parser.add_argument('--test-split', type=float, default=0.0, help='Held-out test split fraction (default: 0; 0 disables test set)')
    train_parser.add_argument('--cv-folds', type=int, default=0, help='Cross-validation folds (default: 0; 0/1 disables CV; >=2 enables CV)')


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


