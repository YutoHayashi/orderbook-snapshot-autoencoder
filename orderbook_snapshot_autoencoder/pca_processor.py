import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Union, List

class PCAProcessor:    
    def __init__(self, n_components: int, standardize: bool = True):
        self.n_components = n_components
        self.standardize = standardize
        self.pca = None
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, features: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]) -> 'PCAProcessor':
        X = self._prepare_data(features)
        
        if self.standardize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        
        self.is_fitted = True
        return self
    
    def transform(self, features: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("PCAProcessor must be fitted before transform")
        
        X = self._prepare_data(features)
        
        if self.standardize and self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.pca.transform(X)
    
    def fit_transform(self, features: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        return self.fit(features).transform(features)
    
    def _prepare_data(self, features: Union[np.ndarray, torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(features, list):
            if all(isinstance(f, torch.Tensor) for f in features):
                features = torch.stack(features)
            else:
                features = np.array(features)
        
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        if isinstance(features, np.ndarray):
            if features.ndim == 1:
                features = features.reshape(1, -1)
            return features
        else:
            raise ValueError(f"Unsupported input type: {type(features)}")
