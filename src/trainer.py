import os
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from lightning.pytorch import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from data_processor import prepare_snap

class OrderBookDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        if 'json_data' not in data.columns:
            raise ValueError("DataFrame must contain 'json_data' column")
        
        data['snap_data'] = data['json_data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        data = data.filter(items=['snap_data'])
        
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        snap = row['snap_data']
        x = prepare_snap(snap)  # Shape: (1, 4, effective_depth_level)
        y = x.clone()  # For autoencoder, target is the same as input
        return x.squeeze(0), y.squeeze(0)  # Remove batch dimension for DataLoader compatibility

class Conv1dAE(nn.Module):
    def __init__(self, channels: int = 4, latent_dim: int = 64, K: int = 20):
        super(Conv1dAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        flat_size = 128 * (K // 4)
        self.fc_enc = nn.Linear(flat_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_size)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, K // 4)),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.fc_enc(z)
        x_reconstruct = self.fc_dec(z)
        x_reconstruct = self.decoder(x_reconstruct)
        return x_reconstruct

class FoecastingConv1dAE(LightningModule):
    def __init__(self,
                 channels: int = 4,
                 latent_dim: int = 64,
                 K: int = 20,
                 rl: float = 1e-3):
        super(FoecastingConv1dAE, self).__init__()
        self.save_hyperparameters()
        self.model = Conv1dAE(channels, latent_dim, K)
        self.loss_fn = nn.MSELoss()
        self.lr = rl
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

def create_dataset(df: pd.DataFrame) -> tuple[OrderBookDataset, OrderBookDataset, OrderBookDataset]:
    train_cutoff = df.index.max() * 0.6
    validation_cutoff = df.index.max() * 0.8
    
    training = OrderBookDataset(df[lambda x: x.index <= train_cutoff])
    validation = OrderBookDataset(df[lambda x: (x.index > train_cutoff) & (x.index <= validation_cutoff)])
    testing = OrderBookDataset(df[lambda x: x.index > validation_cutoff])
    
    print(f"Number of training samples: {len(training)}")
    print(f"Number of validation samples: {len(validation)}")
    print(f"Number of testing samples: {len(testing)}")
    
    return training, validation, testing

def load_ae_model(model_path: str) -> FoecastingConv1dAE:
    print("Loading the best model from checkpoint...")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), model_path)
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in the specified output path.")
    
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    model = FoecastingConv1dAE.load_from_checkpoint(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loaded model from {checkpoint_path}")
    
    return model

class Conv1dAETrainer:
    def __init__(self,
                 data_path: str,
                 epochs: int = 15,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 effective_depth_level: int = 20,
                 model_path: str = 'models',
                 **kwargs):
        self.data_path = data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.effective_depth_level = effective_depth_level
        self.model_path = model_path
        
        self.df = pd.read_csv(data_path)
        
        self.training_dataset, self.validation_dataset, self.testing_dataset = create_dataset(self.df)
    
    def train(self):
        print(f"Starting training...")
        
        train_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        ae = FoecastingConv1dAE(
            channels=4,
            latent_dim=64,
            K=self.effective_depth_level,
            rl=self.learning_rate
        )
        
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-6,
            patience=3,
            verbose=False,
            mode='min'
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(os.path.dirname(__file__), self.model_path),
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
        
        return ae
    
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