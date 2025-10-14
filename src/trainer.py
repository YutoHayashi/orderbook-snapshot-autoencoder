import torch.nn as nn

def load_ae_model():
    pass

class Conv1dAE(nn.Module):
    def __init__(self, channels = 4, latent_dim = 64):
        super(Conv1dAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class ConvAETrainer:
    def __init__(self,
                 **kwargs):
        pass
    
    def train(self):
        pass
    
    def evaluate(self):
        pass