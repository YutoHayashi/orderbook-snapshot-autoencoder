import os
import sys

from dotenv import load_dotenv
load_dotenv()
model_path = os.getenv('MODEL_PATH', 'models')

import torch

from orderbook_snapshot_autoencoder.src.trainer import SnapType, load_ae_model, prepare_snap

ae = load_ae_model(model_path)
hparams = ae.hparams
effective_depth_level = hparams['K']

def encode_snapshot(snap: SnapType):
    with torch.no_grad():
        x = prepare_snap(snap, effective_depth_level)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        device = next(ae.parameters()).device
        x = x.to(device)
        latent_features = ae.encoder(x).flatten()
    return latent_features.detach().cpu()

if __name__ == "__main__":
    import json
    import pandas as pd
    
    df = pd.read_csv('csv/board_snapshots.csv')
    df = df[lambda x: x.index == df.index.max()]
    df['snap_data'] = df['json_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.filter(items=['snap_data'])
    
    row = df.iloc[0]
    snap = row['snap_data']
    
    latent_features = encode_snapshot(snap)
    print("Latent representation:", latent_features)