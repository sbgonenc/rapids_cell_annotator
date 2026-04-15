import anndata as ad
from sklearn.preprocessing import StandardScaler
from scipy import sparse


def prepare_adata(adata: ad.AnnData, label_key: str, feature_key: str) -> tuple:
        """Prepare feature/labels data for classifier processing"""
        # Get features
        if feature_key is None or feature_key not in adata.obsm:
            print(f"[yellow]Warning: {feature_key} not found, using .X[/yellow]")
            X = adata.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = adata.obsm[feature_key]
        
        ##Normalize the features
        if isinstance(X, sparse.spmatrix):
            X = X.toarray()
        
        X = StandardScaler().fit_transform(X)
        
        
        # Get labels
        y = None
        if label_key and label_key in adata.obs.columns:
            y = adata.obs[label_key].astype(str).values
        
        return X, y


