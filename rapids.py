try:
    # GPU / RAPIDS dependencies (optional at import time so CPU-only helpers can be used)
    import cupy as cp
    from cuml.linear_model import LinearRegression as cuLinearRegression
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.ensemble import RandomForestClassifier as cuRandomForestClassifier
    from cuml.preprocessing.LabelEncoder import LabelEncoder as cuLabelEncoder
    from cuml.model_selection import train_test_split as cu_train_test_split
    from cuml.metrics import accuracy_score as cu_accuracy_score
except ModuleNotFoundError:  # pragma: no cover
    cp = None
    cuLinearRegression = None
    cuLogisticRegression = None
    cuRandomForestClassifier = None
    cuLabelEncoder = None
    cu_train_test_split = None
    cu_accuracy_score = None


def preflight_stratified_kfold(y_labels, k: int):
    """
    Pure-python validation for stratified K-fold CV.

    Returns:
        (class_counts, min_class_count)
    """
    from collections import Counter

    if not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if k < 0:
        raise ValueError("k must be >= 0")

    labels_list = list(y_labels) if y_labels is not None else []
    n = len(labels_list)
    class_counts = dict(Counter(labels_list))
    min_class_count = min(class_counts.values()) if class_counts else 0

    # Treat k in {0,1} as "CV disabled" (no error); still return counts.
    if k <= 1:
        return class_counts, min_class_count

    if n == 0:
        raise ValueError("Cannot run cross-validation with zero samples")
    if k > n:
        raise ValueError(f"k ({k}) must be <= n_samples ({n})")
    if k > min_class_count:
        raise ValueError(
            f"k ({k}) must be <= min class count ({min_class_count}) for stratified K-fold"
        )

    return class_counts, min_class_count


def _preflight_stratified_kfold_cpu_only_smoke_check():
    """
    Minimal CPU-only check for the preflight helper (not executed automatically).
    """
    labels = ["a", "a", "b", "b", "b"]
    cc, m = preflight_stratified_kfold(labels, k=2)
    assert cc == {"a": 2, "b": 3}
    assert m == 2


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
    
    def predict(self, X):
        return self.classifier.predict(X)

    @staticmethod
    def get_accuracy(y_hat, y_val):
        return float(cu_accuracy_score(y_val, y_hat))
    
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

    def evaluate_decoded_accuracy(self, features_data, label_data) -> float:
        """
        Evaluate accuracy by decoding predicted class ids back to original labels
        and comparing with provided ground-truth labels.
        """
        if cp is None or cuLabelEncoder is None:
            raise ImportError("RAPIDS dependencies not available (cuml/cupy).")
        if self.classifier is None or self.label_encoder is None:
            raise ValueError("classifier has not been initiated!")

        import numpy as np

        X = cp.asarray(features_data, dtype=cp.float32)
        y_pred_encoded = self.predict(X)
        y_pred_decoded = self.get_decoded_labels(y_pred_encoded).to_numpy()
        y_true = np.asarray(label_data).astype(str)

        return float((y_pred_decoded == y_true).mean())

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

    def cross_validate(
        self,
        features_data,
        label_data,
        k: int,
        seed: int,
        model_type: str,
        model_params: dict,
    ) -> dict:
        """
        Stratified K-fold cross-validation (accuracy).

        Returns a dict with:
          - fold_acc (list[float]), mean_acc (float), std_acc (float)
          - class_counts (dict[str,int]) computed from raw string labels
          - n_samples, n_features, n_classes
        """
        if cp is None or cuLabelEncoder is None or cu_accuracy_score is None:
            raise ImportError("RAPIDS dependencies not available (cuml/cupy).")

        # Preflight and label counts (pure python, raw string labels)
        class_counts, _min_class_count = preflight_stratified_kfold(label_data, k)

        if k <= 1:
            return {
                "fold_acc": [],
                "mean_acc": float("nan"),
                "std_acc": float("nan"),
                "class_counts": class_counts,
                "n_samples": int(len(label_data) if label_data is not None else 0),
                "n_features": int(getattr(features_data, "shape", [0, 0])[1]),
                "n_classes": int(len(class_counts)),
                "k": int(k),
                "seed": int(seed),
                "model_type": str(model_type),
            }

        import numpy as np

        # Convert features once for GPU use
        X = cp.asarray(features_data, dtype=cp.float32)

        # Encode labels once for stable mapping across folds
        encoder = cuLabelEncoder()
        y_encoded = encoder.fit_transform(label_data)
        y_encoded = cp.asarray(y_encoded)

        n_samples = int(X.shape[0])
        n_features = int(X.shape[1]) if len(X.shape) > 1 else 1
        n_classes = int(len(class_counts))

        # Build splitter (prefer cuML if available)
        splitter = None
        try:
            from cuml.model_selection import StratifiedKFold as cuStratifiedKFold  # type: ignore

            splitter = cuStratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        except Exception:
            try:
                from sklearn.model_selection import StratifiedKFold as skStratifiedKFold  # type: ignore

                splitter = skStratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            except Exception as e:
                raise ImportError(
                    "No StratifiedKFold available: install scikit-learn or use a RAPIDS version "
                    "that provides cuml.model_selection.StratifiedKFold."
                ) from e

        # Ensure we don't mutate the currently configured model on self
        orig_classifier = self.classifier
        orig_encoder = self.label_encoder

        def _make_fresh_model():
            if model_type == "linear_regression":
                return cuLinearRegression(**model_params)
            if model_type == "logistic_regression":
                return cuLogisticRegression(**model_params)
            if model_type == "random_forest":
                return cuRandomForestClassifier(**model_params)
            raise ValueError(f"Unsupported classifier type for CV: {model_type}")

        fold_acc = []
        first_fold = True

        try:
            # Prefer numpy-based splitting (works for sklearn; often also acceptable for cuML)
            try:
                split_iter = splitter.split(np.zeros(n_samples), cp.asnumpy(y_encoded))
            except Exception:
                split_iter = splitter.split(X, y_encoded)

            for train_idx, val_idx in split_iter:
                train_idx_gpu = cp.asarray(train_idx)
                val_idx_gpu = cp.asarray(val_idx)

                X_train = X[train_idx_gpu]
                X_val = X[val_idx_gpu]
                y_train = y_encoded[train_idx_gpu]
                y_val = y_encoded[val_idx_gpu]

                if first_fold:
                    if int(X_train.shape[0]) != int(len(train_idx)) or int(X_val.shape[0]) != int(len(val_idx)):
                        raise RuntimeError(
                            "Indexing sanity check failed: fold slice sizes do not match index lengths"
                        )
                    first_fold = False

                model = _make_fresh_model()
                model.fit(X_train, y_train)
                y_hat = model.predict(X_val)
                acc = float(cu_accuracy_score(y_val, y_hat))
                fold_acc.append(acc)
        finally:
            # Restore state even if CV errors
            self.classifier = orig_classifier
            self.label_encoder = orig_encoder

        mean_acc = float(np.mean(fold_acc)) if fold_acc else float("nan")
        std_acc = float(np.std(fold_acc, ddof=0)) if fold_acc else float("nan")

        return {
            "k": int(k),
            "seed": int(seed),
            "model_type": str(model_type),
            "fold_acc": fold_acc,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "class_counts": class_counts,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": n_classes,
        }


    

