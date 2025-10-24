from typing import TypedDict
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .pca_processor import PCAProcessor

class OrderType(TypedDict):
    price: float
    size: float

class SnapType(TypedDict):
    mid_price: float
    bids: list[OrderType]
    asks: list[OrderType]

def prepare_snap(snap: SnapType, effective_depth_level: int) -> torch.Tensor:
    mid_price = snap.get('mid_price', np.float32(0))
    asks = sorted(snap.get('asks', []), key=lambda x: x.get('price', np.float32(0)))[:effective_depth_level]
    bids = sorted(snap.get('bids', []), key=lambda x: x.get('price', np.float32(0)), reverse=True)[:effective_depth_level]
    
    def pad(side):
        pad_len = effective_depth_level - len(side)
        if pad_len > 0:
            side = side + [{"price": mid_price, "size": 0.0}] * pad_len
        return side

    asks = pad(asks)
    bids = pad(bids)

    ask_prices = [(a["price"] - mid_price) / mid_price for a in asks]
    ask_sizes  = [a["size"] for a in asks]
    bid_prices = [(b["price"] - mid_price) / mid_price for b in bids]
    bid_sizes  = [b["size"] for b in bids]

    features = np.array(
        [v for pair in zip(ask_prices, ask_sizes) for v in pair] +
        [v for pair in zip(bid_prices, bid_sizes) for v in pair],
        dtype=np.float32
    )

    return features

class OrderBookDataset(Dataset):
    def __init__(self, data: pd.DataFrame, effective_depth_level: int):
        if 'json_data' not in data.columns:
            raise ValueError("DataFrame must contain 'json_data' column")
        
        data['snap_data'] = data['json_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        data = data.filter(items=['snap_data'])
        
        self.data = data
        self.effective_depth_level = effective_depth_level

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        snap = row['snap_data']
        x = prepare_snap(snap, effective_depth_level=self.effective_depth_level)  # Shape: (4 * K,)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0) # Shape: (1, 4 * K)

class FoecastingConv1dAE(LightningModule):
    def __init__(self,
                 channels: int,
                 latent_dim: int,
                 K: int = 10,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss_fn = nn.MSELoss()

        input_len = channels * K  # ex: 4×10=40

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),  # (1, L) -> (32, L/2)
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # (32, L/2) -> (64, L/4)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),  # (batch, 64*4)
        )

        self.fc_mu = nn.Linear(64 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64 * 4)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 4)),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 8)
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # -> (16, 16)
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),  # -> (1, 16)
        )

        self.final_resize = nn.AdaptiveAvgPool1d(input_len)

    def forward(self, x):
        z = self.encoder(x)
        latent = self.fc_mu(z)
        dec = self.fc_dec(latent)
        dec = self.decoder(dec)
        out = self.final_resize(dec)
        return out

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def load_ae_model(model_path: str) -> FoecastingConv1dAE:
    print("Loading the best model from checkpoint...")
    checkpoint_files = [f for f in os.listdir(model_path) if f.startswith('conv1dae-') and f.endswith('.ckpt')]
    if not checkpoint_files:
        raise FileNotFoundError("No conv1dae checkpoint files found in the specified output path.")
    
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(model_path, f)))
    checkpoint_path = os.path.join(model_path, latest_checkpoint)
    
    model = FoecastingConv1dAE.load_from_checkpoint(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loaded model from {checkpoint_path}")
    
    return model

def load_pca_model(model_path: str) -> PCAProcessor|None:
    print("Loading PCA model...")
    pca_path = os.path.join(model_path, 'pca_processor.pkl')
    
    if not os.path.exists(pca_path):
        return None
    
    with open(pca_path, 'rb') as f:
        pca_processor = pickle.load(f)
    
    print(f"Loaded PCA model from {pca_path}")
    return pca_processor

class Conv1dAETrainer:
    def __init__(self,
                 data_path: str,
                 epochs: int = 15,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 effective_depth_level: int = 10,
                 pca_components: int = 0,
                 model_path: str = 'models',
                 **kwargs):
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.effective_depth_level = effective_depth_level
        self.pca_components = pca_components
        self.model_path = model_path
        
        self.df = pd.read_csv(data_path)
        
        self.training_dataset, self.validation_dataset, self.testing_dataset = self.create_dataset(self.df)
    
    def create_dataset(self, df: pd.DataFrame) -> tuple[OrderBookDataset, OrderBookDataset, OrderBookDataset]:
        train_cutoff = df.index.max() * 0.6
        validation_cutoff = df.index.max() * 0.8
        
        training = OrderBookDataset(df[lambda x: x.index <= train_cutoff], effective_depth_level=self.effective_depth_level)
        validation = OrderBookDataset(df[lambda x: (x.index > train_cutoff) & (x.index <= validation_cutoff)], effective_depth_level=self.effective_depth_level)
        testing = OrderBookDataset(df[lambda x: x.index > validation_cutoff], effective_depth_level=self.effective_depth_level)
        
        print(f"Number of training samples: {len(training)}")
        print(f"Number of validation samples: {len(validation)}")
        print(f"Number of testing samples: {len(testing)}")
        
        return training, validation, testing
    
    def train(self):
        print(f"Starting training...")
        
        train_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        ae = FoecastingConv1dAE(
            channels=4,
            latent_dim=64,
            K=self.effective_depth_level,
            lr=self.learning_rate
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-6,
            patience=3,
            verbose=False,
            mode='min'
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_path,
            filename="conv1dae-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )
        
        trainer = Trainer(
            max_epochs=self.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stopping_callback, checkpoint_callback],
            logger=False
        )
        
        trainer.fit(ae, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        if (self.pca_components > 0):
            self.train_pca(ae)
        
        return ae
    
    def train_pca(self, model: FoecastingConv1dAE) -> None:
        print("Starting PCA training...")
        model.eval()
        
        latent_features_list = []
        train_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        with torch.no_grad():
            for batch in train_dataloader:
                batch = batch.to(next(model.parameters()).device)
                encoded = model.encoder(batch)
                encoded_flat = model.flatten(encoded)
                
                for feature in encoded_flat:
                    latent_features_list.append(feature.detach().cpu())
        
        pca_processor = PCAProcessor(n_components=self.pca_components)
        pca_processor.fit(latent_features_list)
        
        pca_path = os.path.join(self.model_path, 'pca_processor.pkl')
        os.makedirs(os.path.dirname(pca_path), exist_ok=True)
        
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_processor, f)
        
        print(f"PCA training completed. Model saved to {pca_path}")
        print(f"PCA components: {self.pca_components}")
        print(f"Training samples: {len(latent_features_list)}")
    
    def evaluate(self, model: FoecastingConv1dAE) -> None:
        print("Starting evaluation...")
        model.eval()
        
        trainer = Trainer(
            logger=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1 if torch.cuda.is_available() else "auto",
        )
        
        test_dataloader = DataLoader(self.testing_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        results = trainer.test(model, dataloaders=test_dataloader, verbose=False)
        
        print(f"===== Evaluation Results =====")
        for key, value in results[0].items():
            print(f"{key}: {value}")
        print(f"==============================")