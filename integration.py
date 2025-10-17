import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()
model_path = os.getenv('MODEL_PATH', 'models')

import torch

from trainer import SnapType, load_ae_model, prepare_snap

def encode_snapshot(snap: SnapType):
    ae = load_ae_model(model_path)
    hparams = ae.hparams
    effective_depth_level = hparams['K']
    
    with torch.no_grad():
        x = prepare_snap(snap, effective_depth_level)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        latent_features = ae.encoder(x).flatten()
    return latent_features.detach()

if __name__ == "__main__":
    import json
    import pandas as pd
    
    ae = load_ae_model(model_path)
    hparams = ae.hparams
    effective_depth_level = hparams['K']
    
    df = pd.read_csv('csv/board_snapshots.csv')
    df = df[lambda x: x.index == df.index.max()]
    df['snap_data'] = df['json_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df = df.filter(items=['snap_data'])
    
    row = df.iloc[0]
    snap = row['snap_data']
    
    latent_features = encode_snapshot(snap)
    print("Latent representation:", latent_features)