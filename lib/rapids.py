# GPU / RAPIDS dependencies (optional at import time so CPU-only helpers can be used)
import cupy as cp

# cuml classifiers
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
from cuml.svm import LinearSVC as cuSVC

from cuml.preprocessing import LabelEncoder as cuLabelEncoder
from cuml.model_selection import train_test_split as cu_train_test_split
from cuml.metrics import accuracy_score as cu_accuracy_score

from cuml.model_selection import StratifiedKFold

import optuna
import itertools

from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    classification_report,
    cohen_kappa_score,
)

#from cuml.naive_bayes import CategoricalNB as cuCategoricalNB

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial noise


# ── Search space registry ──────────────────────────────────────────────────────
# Each entry is a callable: (trial) → params dict
SEARCH_SPACES = {
    cuLogisticRegression: lambda trial: {
        "C":              trial.suggest_float("C", 1e-3, 1e3, log=True),
        "max_iter":       trial.suggest_int("max_iter", 1000, 5000),
        "linesearch_max_iter": trial.suggest_int("linesearch_max_iter", 50, 100),
        "class_weight":   trial.suggest_categorical("class_weight", [None, "balanced"]),
        #"penalty":        trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),

        #"solver":         trial.suggest_categorical("solver", ["qn", "lbfgs"]),
    },
    cuLinearRegression: lambda trial: {
        # LinearRegression has very few tunable knobs in cuml
        "fit_intercept":  trial.suggest_categorical("fit_intercept", [True, False]),
        "normalize":      trial.suggest_categorical("normalize",     [True, False]),
    },
    cuRandomForestClassifier: lambda trial: {
        "n_estimators":   trial.suggest_int("n_estimators", 50, 500),
        "max_depth":      trial.suggest_int("max_depth", 20, 500),
        "max_features":   trial.suggest_float("max_features", 0.3, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "n_bins":         trial.suggest_int("n_bins", 64, 256),
    },
    cuSVC: lambda trial: {
        "C":              trial.suggest_float("C", 1e-2, 1e3, log=True),
        "probability":    trial.suggest_categorical("probability", [True, False]),
        "fit_intercept":  trial.suggest_categorical("fit_intercept", [True, False]),
        "tol":            trial.suggest_float("tol", 1e-4, 1e-1, log=True),
        "max_iter":       trial.suggest_int("max_iter", 1000, 5000),
        "loss":           trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        "class_weight":   trial.suggest_categorical("class_weight", [None, "balanced"]),
    },
   # cuCategoricalNB: lambda trial: {
   #     "alpha":          trial.suggest_float("alpha", 1e-3, 1e3, log=True),
   # },
}


class RAPIDS_Classifier:

    def __init__(self, random_state=42) -> None:
        
        self.random_state = random_state

        self.classifier = None
        self.label_encoder = None
        self.train_data = {}
        self.train_stats = {}
    
    def init_classifier(self,model_type, **model_params):
        if cuLabelEncoder is None:
            raise ImportError("RAPIDS dependencies not available (cuml/cupy).")
        if model_type == "linear_regression":
            self.classifier = cuLinearRegression(**model_params)
        elif model_type == "logistic_regression":
            self.classifier = cuLogisticRegression(**model_params)
        elif model_type == "random_forest":
            self.classifier = cuRandomForestClassifier(**model_params)
        elif model_type == "svm":
            self.classifier = cuSVC(**model_params)
        else:
            raise ValueError("Unsupported classifier type")
        self.label_encoder = cuLabelEncoder()
        return self.classifier
    
    def process_training_data(self, X, y, validation_split:float= 0.125):
        """
        X: features
        y: label (target)
        """
               
        if cp is None:
            raise ImportError("RAPIDS dependencies not available (cuml/cupy).")
        # Ensure float32 for GPU efficiency
        X = cp.asarray(X, dtype=cp.float32)
        y_encoded = self.label_encoder.fit_transform(y)
        
            
        X_train, X_val, y_train, y_val = cu_train_test_split(X, y_encoded, test_size=validation_split, random_state=self.random_state)

        self.train_data = {
            "X_train" : X_train,
            "X_val" : X_val,
            "y_train" : y_train,
            "y_val" : y_val
        }


    def save_status(self, out_dir):
        import os
        self.save_classifier(os.path.join(out_dir, "classifier.pkl"))
        self.save_label_encoder(os.path.join(out_dir, "encoder.pkl"))
        self.save_train_stats(os.path.join(out_dir, "training_acc.json"))

    def save_label_encoder(self, out_path):
        import pickle
        with open(out_path, "wb") as fh:
            pickle.dump(self.label_encoder, fh)

    def save_classifier(self, out_path):
        import pickle
        with open(out_path, "wb") as fh:
            pickle.dump(self.classifier, fh)
    
    def save_train_stats(self, out_path):
        import json
        with open(out_path, "w") as fh:
            json.dump(self.train_stats, fh)

    def load_classifier(self, model_path):
        import pickle
        with open(model_path, "rb") as fh:
            self.classifier = pickle.load(fh)
        return self.classifier
    
    def load_encoder(self, encoder_path):
        import pickle
        with open(encoder_path, "rb") as fh:
            self.label_encoder = pickle.load(fh)
        return self.label_encoder
    
    def get_decoded_labels(self, y_pred_encoded):
        return self.label_encoder.inverse_transform(y_pred_encoded) 
    
    def train(self, X, y):
        if self.classifier is None:
            raise ValueError("classifier has not been initiated!")
        self.classifier.fit(X, y)
        return self.classifier
    
    def get_predictions(self, X):
        pred_proba = cp.max(self.classifier.predict_proba(X), axis=1)
        # Note: If you are seeing "argmax() got an unexpected keyword argument 'dtype'", it means you may be using a version
        # of cupy where cp.argmax no longer takes 'dtype' (as newer cupy removed this argument).
        # If code somewhere uses: cp.argmax(..., dtype=...), remove the dtype argument.
        # This line itself should work as is unless self.classifier.predict_proba(X) returns an object not compatible with cupy.
        # If still problematic, try using numpy instead of cupy:
        #import numpy as np
        #pred_proba = np.argmax(self.classifier.predict_proba(X), axis=1)
        return self.predict(X), pred_proba

    
    def predict(self, X):
        return self.classifier.predict(X)

    @staticmethod
    def get_accuracy(y_hat, y_val):
        return float(cu_accuracy_score(y_val, y_hat))
    
    @staticmethod
    def get_cohen_kappa(y_hat, y_val):
        return cohen_kappa_score(y_val.get(), y_hat.get())
    
    @staticmethod
    def get_mcc(y_true, y_pred):
    # cupy arrays → numpy for sklearn
        y_true_np = y_true.get() if hasattr(y_true, "get") else y_true
        y_pred_np = y_pred.get() if hasattr(y_pred, "get") else y_pred
        return matthews_corrcoef(y_true_np, y_pred_np)
    
    
    def holdout_test_set_cv(
            self,
            features_data,
            label_data,
            test_size: float = 0.2,
            n_splits: int = 5,
            classifier_params: dict = None,
            param_grid: dict = None,
            use_optuna: bool = False,
            n_trials: int = 30,
            optuna_direction: str = "maximize",
            random_state: int = None,
            scorer=None,
        ):
        """
        Hold-out test set + K-Fold CV with three hyperparameter modes:

            1. Fixed params      – pass classifier_params, leave param_grid/use_optuna unset
            2. Grid search       – pass param_grid (exhaustive, uses itertools)
            3. Optuna search     – pass use_optuna=True (smart sampling, recommended)

        Args:
            features_data:      Features (array-like)
            label_data:         Labels (array-like)
            test_size:          Hold-out fraction (default 0.2)
            n_splits:           CV folds (default 5)
            classifier_params:  Fixed params dict (mode 1)
            param_grid:         Grid dict  (mode 2), e.g. {"C": [0.1, 1, 10]}
            use_optuna:         Enable Optuna search (mode 3)
            n_trials:           Number of Optuna trials (default 30)
            optuna_direction:   "maximize" or "minimize" (default "maximize")
            random_state:       RNG seed
            scorer:             Callable(y_true, y_pred) → float. Defaults to cu_accuracy_score.

        Returns:
            dict with:
                cv_scores       – fold scores of the winning configuration
                avg_cv_score    – mean of cv_scores
                test_score      – final held-out test score
                best_params     – winning params  (modes 2 & 3)
                optuna_study    – the live Study object (mode 3 only)
                param_search    – full grid log      (mode 2 only)
        """
        if cp is None or cu_train_test_split is None or cu_accuracy_score is None:
            raise ImportError("RAPIDS dependencies not available (cuml/cupy).")

        if scorer is None:
            scorer = self.get_mcc
            #scorer = self.get_accuracy
        
        if random_state is None:
            random_state = self.random_state

        # ── Data prep ──────────────────────────────────────────────────────────────
        X         = cp.asarray(features_data, dtype=cp.float32)
        #y_encoded = self.label_encoder.fit_transform(label_data)
        y_encoded = cp.asarray(self.label_encoder.fit_transform(label_data))


        X_dev, X_test, y_dev, y_test = cu_train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )

        skf           = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        clf_class     = type(self.classifier)

        # ── Shared CV evaluator ───────────────────────────────────────────────────
        def _cross_val_score(params: dict) -> list[float]:
            fold_scores = []
            for train_idx, val_idx in skf.split(X_dev, y_dev):
                clf = clf_class(**params) if params else clf_class()
                clf.fit(X_dev[train_idx], y_dev[train_idx])
                fold_scores.append(scorer(y_dev[val_idx], clf.predict(X_dev[val_idx])))
            return fold_scores

        # ══════════════════════════════════════════════════════════════════════════
        # MODE 3 — Optuna
        # ══════════════════════════════════════════════════════════════════════════
        study      = None
        search_log = None
        best_params = classifier_params or {}

        if use_optuna:
            if clf_class not in SEARCH_SPACES:
                raise ValueError(
                    f"No Optuna search space defined for {clf_class.__name__}. "
                    f"Supported: {[c.__name__ for c in SEARCH_SPACES]}"
                )

            space_fn = SEARCH_SPACES[clf_class]

            def _objective(trial):
                params = space_fn(trial)

                # poly-specific: gamma/degree irrelevant for non-poly kernels
                if clf_class is cuSVC and params.get("kernel") != "poly":
                    params.pop("degree", None)

                scores = _cross_val_score(params)
                # Store fold scores as user attribute for later inspection
                trial.set_user_attr("cv_scores", scores)
                return float(cp.mean(cp.array(scores)))

            sampler = optuna.samplers.TPESampler(seed=random_state)
            study   = optuna.create_study(direction=optuna_direction, sampler=sampler)
            study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

            best_params = study.best_params
            cv_scores   = study.best_trial.user_attrs["cv_scores"]

        # ══════════════════════════════════════════════════════════════════════════
        # MODE 2 — Grid search
        # ══════════════════════════════════════════════════════════════════════════
        elif param_grid is not None:
            keys    = list(param_grid.keys())
            combos  = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]

            search_log  = []
            best_mean   = -float("inf")
            cv_scores   = None

            for params in combos:
                scores     = _cross_val_score(params)
                mean_score = float(cp.mean(cp.array(scores)))
                search_log.append({"params": params, "cv_scores": scores, "mean_cv_score": mean_score})

                if mean_score > best_mean:
                    best_mean   = mean_score
                    best_params = params
                    cv_scores   = scores

        # ══════════════════════════════════════════════════════════════════════════
        # MODE 1 — Fixed / default params
        # ══════════════════════════════════════════════════════════════════════════
        else:
            cv_scores = _cross_val_score(best_params)

        avg_cv_score = float(cp.mean(cp.array(cv_scores)))

        # ── Final refit on full dev set → evaluate on test set ────────────────────
        final_clf = clf_class(**best_params) if best_params else clf_class()
        final_clf.fit(X_dev, y_dev)


        y_hat = final_clf.predict(X_test)
        test_score = float(scorer(y_test, y_hat))

        # Convert once for all sklearn metrics
        y_test_np = y_test.get()
        y_hat_np  = y_hat.get()

        mcc        = matthews_corrcoef(y_test_np, y_hat_np)
        kappa      = cohen_kappa_score(y_test_np, y_hat_np)
        macro_f1   = f1_score(y_test_np, y_hat_np, average="macro")
        weighted_f1= f1_score(y_test_np, y_hat_np, average="weighted")
        # per-class breakdown — decode labels back to cell type names
        class_names = self.label_encoder.classes_.to_numpy()
        clf_report  = classification_report(
            y_test_np, y_hat_np,
            target_names=class_names,
            output_dict=True,  # returns a dict you can log or convert to DataFrame
        )
        # ── Package results ───────────────────────────────────────────────────────
        results = {
            "cv_scores":         cv_scores,
            "avg_cv_score":      avg_cv_score,
            "test_score":        test_score,          # now MCC by default
            "mcc":               mcc,
            "cohen_kappa_score": kappa,
            "accuracy_score": self.get_accuracy(y_test, y_hat),
            "macro_f1":          macro_f1,
            "weighted_f1":       weighted_f1,
            "classification_report": clf_report,      # per-class precision/recall/F1
            
        }

        if use_optuna or param_grid is not None:
            results["best_params"] = best_params
        if use_optuna:
            results["optuna_study"] = study        # full Study: trials, plots, etc.
        if param_grid is not None:
            results["param_search"] = search_log

        rv_results = {k: v for k, v in results.items() if k != "optuna_study"}
        self.classifier = final_clf
        self.train_stats.update(rv_results)
        return rv_results
    
    def process_training(self, features_data, label_data, test_split:float=0.125):
        if self.classifier is None:
            raise ValueError("classifier has not been initiated!")

        self.process_training_data(
            features_data,
            label_data,
            test_split
        )

        self.train(self.train_data["X_train"], self.train_data["y_train"])
        train_predict = self.predict(self.train_data["X_train"])
        train_acc = self.get_accuracy(train_predict, self.train_data["y_train"])
        test_predict = self.predict(self.train_data["X_val"])
        test_acc =  self.get_accuracy(test_predict, self.train_data["y_val"])
        self.train_stats.update({
            "training_acc" : train_acc,
            "test_acc" : test_acc
        })

    def process_training_with_test(
        self,
        features_data,
        label_data,
        validation_split: float = 0.1,
        test_split: float = 0.1,
    ):
        """
        Train on a training set, compute accuracy on a validation set, and
        compute held-out accuracy on a separate test set.

        `validation_split` and `test_split` are fractions of the full dataset.
        """
        if test_split is None or test_split <= 0:
            # Backwards compatible path: the existing `process_training` uses
            # `test_split` as the validation fraction (naming kept for legacy).
            return self.process_training(features_data, label_data, test_split=validation_split)

        if cp is None or cuLabelEncoder is None or cu_accuracy_score is None or cu_train_test_split is None:
            raise ImportError("RAPIDS dependencies not available (cuml/cupy).")

        if validation_split < 0:
            raise ValueError("validation_split must be >= 0")
        if test_split < 0:
            raise ValueError("test_split must be >= 0")
        if validation_split + test_split >= 1:
            raise ValueError("validation_split + test_split must be < 1")

        X = cp.asarray(features_data, dtype=cp.float32)
        y_encoded = self.label_encoder.fit_transform(label_data)

        # First split: hold out explicit test set
        X_trainval, X_test, y_trainval, y_test = cu_train_test_split(
            X, y_encoded, test_size=test_split, random_state=self.random_state
        )

        # Second split: split the remainder into train/validation
        val_split_adj = validation_split / (1 - test_split)
        X_train, X_val, y_train, y_val = cu_train_test_split(
            X_trainval, y_trainval, test_size=val_split_adj, random_state=self.random_state
        )

        self.train(X_train, y_train)

        train_predict = self.predict(X_train)
        train_acc = self.get_accuracy(train_predict, y_train)

        val_predict = self.predict(X_val)
        val_acc = self.get_accuracy(val_predict, y_val)

        test_predict = self.predict(X_test)
        test_acc = self.get_accuracy(test_predict, y_test)

        self.train_stats.update(
            {
                "training_acc": train_acc,
                "validation_acc": val_acc,
                "test_acc": test_acc,
            }
        )
        return self.train_stats

    def train_full(
        self,
        features_data,
        label_data,
        model_type: str,
        model_params: dict,
    ):
        """
        Train a single final model on ALL provided samples.

        Also computes a sanity-check accuracy by evaluating on the same data.
        The resulting stats are stored in self.train_stats with keys:
          - training_acc
          - test_acc (same-data evaluation for backward compatibility)
        """
        if cp is None or cuLabelEncoder is None or cu_accuracy_score is None:
            raise ImportError("RAPIDS dependencies not available (cuml/cupy).")

        # Reinitialize model + encoder deterministically from provided params
        self.init_classifier(model_type=model_type, **model_params)

        X = cp.asarray(features_data, dtype=cp.float32)
        y_encoded = self.label_encoder.fit_transform(label_data)
        y_encoded = cp.asarray(y_encoded)

        self.classifier.fit(X, y_encoded)
        y_hat = self.classifier.predict(X)
        acc = float(cu_accuracy_score(y_encoded, y_hat))

        # Backward compatible schema: always include training_acc and test_acc
        self.train_stats = {
            "training_acc": acc,
            "test_acc": acc,
        }

    

