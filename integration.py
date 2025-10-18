import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()
model_path = os.getenv('MODEL_PATH', 'models')

import torch
from typing import Optional

from trainer import SnapType, load_ae_model, prepare_snap, load_pca_model

ae = load_ae_model(model_path)
hparams = ae.hparams
effective_depth_level = hparams['K']

def encode_snapshot(snap: SnapType, pca_components: Optional[int] = None):
    with torch.no_grad():
        x = prepare_snap(snap, effective_depth_level)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        device = next(ae.parameters()).device
        x = x.to(device)
        latent_features = ae.encoder(x).flatten()
        latent_features = latent_features.detach().cpu()
    
    if pca_components is None:
        return latent_features
    
    pca_processor = load_pca_model(model_path)
    
    if pca_processor.n_components != pca_components:
        raise ValueError(f"PCA model has {pca_processor.n_components} components, but {pca_components} requested.")
    
    compressed_features = pca_processor.transform([latent_features])
    return torch.tensor(compressed_features[0], dtype=torch.float32)

if __name__ == "__main__":
    import json
    import pandas as pd
    
    df = pd.read_csv('csv/board_snapshots.csv')
    df = df[lambda x: x.index == df.index.max()]
    df['snap_data'] = df['json_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.filter(items=['snap_data'])
    
    row = df.iloc[0]
    snap = row['snap_data']
    
    latent_features = encode_snapshot(snap, pca_components=40)
    print("Latent representation:", latent_features)