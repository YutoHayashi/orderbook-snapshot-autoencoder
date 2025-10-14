import os

import numpy as np

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

effective_depth_level = os.getenv('EFFECTIVE_DEPTH_LEVEL', 10)

def prepare_snap(snap: dict) -> torch.Tensor:
    if not isinstance(snap, dict) or 'mid_price' not in snap or 'bids' not in snap or 'asks' not in snap:
        raise ValueError("Invalid snapshot format")
    
    mid_price = snap.get('mid_price', np.float32(0))
    asks = sorted(snap.get('asks', []), key=lambda x: x.get('price', np.float32(0)))[:effective_depth_level]
    bids = sorted(snap.get('bids', []), key=lambda x: x.get('price', np.float32(0)), reverse=True)[:effective_depth_level]
    
    def list_to_array(lst: list) -> np.ndarray:
        arr = np.zeros((effective_depth_level, 2), dtype=np.float32)
        for i, item in enumerate(lst):
            arr[i, 0] = float(item.get('price', 0)) - float(mid_price)
            arr[i, 1] = np.log1p(float(item.get('size')))
        return arr
    
    bids_arr = list_to_array(bids) # Shape: (effective_depth_level, 2)
    asks_arr = list_to_array(asks) # Shape: (effective_depth_level, 2)
    
    # Stack channels: (channels, effective_depth_level) -> here chennels=4 (ask_price_rel, ask_size, bid_price_rel, bid_size)
    stacked = np.concatenate([asks_arr.T, bids_arr.T], axis=0)
    
    return torch.tensor(stacked).unsqueeze(0) # Shape: (1, 4, effective_depth_level)