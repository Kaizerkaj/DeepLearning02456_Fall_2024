import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm.autonotebook import tqdm
from alive_progress import alive_bar

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        # Linearly spaced betas
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        # print(f"{self.betas = }")
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]]) 

    def forward_process(self, x0, t):
        """
        Adds noise to the data at timestep `t` during the forward process.
        x0: Original data.
        t: Current timestep.
        """
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.alpha_cumprod[t - 1].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = (1 - self.alpha_cumprod[t - 1]).sqrt().view(-1, 1, 1, 1)

        # print(f"{sqrt_alpha_cumprod_t.shape = }", f"{x0.shape = }", sep = "\n")

        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_t * noise
        return xt, noise

class AttentionBlock(nn.Module):
    """
    Self-attention block for 2D feature maps.
    Applies self-attention on input feature maps of size [B, C, H, W].
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)  # Normalize features
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)  # Query, Key, Value projection
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)  # Output projection

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)  # Normalize input

        # Compute Q, K, V
        qkv = self.qkv(h).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into Query, Key, Value

        # Compute attention weights
        scale = q.shape[-1] ** -0.5  # Scale factor for softmax
        attention = (q @ k.transpose(-2, -1)) * scale  # Compute scaled dot-product
        attention = attention.softmax(dim=-1)  # Apply softmax

        # Apply attention to values
        out = (attention @ v).reshape(B, C, H, W)
        out = self.proj_out(out)  # Project back to original channel size

        return x + out  # Add residual connection

class SinusoidalPositionEmbedding(nn.Module):
    """
    Encodes the timestep `t` as a sinusoidal position embedding.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t):
        half_dim = self.embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64, time_emb_dim=128):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=False),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU(inplace=False),
        )

        # Encoder with attention
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=False),
            AttentionBlock(hidden_dim),  # Add self-attention block
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        # Get time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb[:, :, None, None]  # Expand to spatial dimensions

        # Encoder
        encoded = self.encoder(x)

        # Add time embedding
        encoded = encoded + t_emb

        # Decoder
        decoded = self.decoder(encoded)
        return decoded


def reverse_process(model, xt, t, betas, alpha_cumprod, epsilon=1e-5):
    """
    Computes x_{t-1} from x_t in the reverse process.
    xt: Current noisy data at timestep `t`.
    t: Current timestep.
    betas: Noise schedule.
    alpha_cumprod: Cumulative product of alphas.
    epsilon: Small value to prevent division by zero.
    """
    with torch.no_grad():
        # Convert `t` to a tensor with the correct shape
        t_tensor = torch.tensor([t], dtype=torch.float32, device=xt.device)

        predicted_noise = model(xt, t_tensor)  # Predict noise
        alpha_t = alpha_cumprod[t - 1]
        alpha_prev_t = alpha_cumprod[t - 2] if t > 1 else torch.tensor(1.0, device=xt.device)
        beta_t = betas[t - 1]
        
        # Compute the mean
        mean = (xt - beta_t * predicted_noise / (1 - alpha_t).sqrt()) / (1 - beta_t).sqrt() # alpha_t.sqrt()
        
        if t > 1:
            # Compute the variance and add noise
            # variance = beta_t * (1 - alpha_prev_t) / (1 - alpha_t) 
            variance = beta_t
            noise = torch.randn_like(xt)
            xt_prev = mean + noise * variance.sqrt()
        else:
            # No noise added for the final timestep
            xt_prev = mean
        
        # Optional clamping for numerical stability (can be removed if unnecessary)
        # xt_prev = torch.clamp(xt_prev, -1, 1)
        
        return xt_prev

from torch.utils.data import Subset

# Define a subset of indices
subset_indices = range(60000)  # First 1000 samples for example
batch_size = 32

# Transformations for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_subset = Subset(train_dataset, subset_indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_subset = Subset(test_dataset, subset_indices)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# Define diffusion parameters
timesteps = 1000
diffusion = Diffusion(timesteps=timesteps, beta_start = 0.0001, beta_end = 0.02)

# Initialize the U-Net model
device = torch.device('cpu')  # use cpu
model = UNet(in_channels=1, out_channels=1, hidden_dim=64)
model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, data_loader, diffusion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x0, _ in (pbar := tqdm(data_loader, desc = f"Epoch {epoch + 1} / {epochs}")):
            device = torch.device('cpu') # use cpu
            x0 = x0.to(device)  # Load batch to GPU
            # print(f"{x0.shape = }")
            t = torch.randint(1, diffusion.timesteps + 1, (x0.size(0),)).to(device)  # Random timesteps
            # print(f"{t[0] = } | {t[1] = }")
            xt, noise = diffusion.forward_process(x0, t)  # Forward process

            # print(f"{xt.shape = } | {t.shape = }")

            predicted_noise = model(xt, t)  # Model predicts noise
            
            # print(f"{predicted_noise.shape = }")

            # Compute loss: MSE between predicted and true noise
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"Total Loss: {epoch_loss / len(data_loader):.4f}")
            
            epoch_loss += loss.item()

def generate_images(model, diffusion, num_images=16):
    model.eval()
    generated_images = []
    with torch.no_grad():
        device = torch.device('cpu')  # Use CPU
        # Start with pure noise
        x = torch.randn((num_images, 1, 28, 28)).to(device)
        for t in (pbar := tqdm(range(diffusion.timesteps, 0, -1), desc="Generating Images")):
            x = reverse_process(model, x, t, diffusion.betas, diffusion.alpha_cumprod)
            pbar.set_postfix_str(f"t = {t}")
        # Rescale the output to [0, 1] after reverse diffusion
        # x = x.clip(-1, 1)
        # x = (x + 1) / 2
        generated_images.append(x.cpu())
    return torch.cat(generated_images, dim=0)

import pickle
# Train the model on MNIST


# torch.autograd.set_detect_anomaly(True)


# Load the data from a pickle file

autoRun = False

if autoRun:
    train_model(model, train_loader, diffusion, optimizer, epochs=5)
else:
    if input("Load model? (y/n)").lower() == "y":
        with open('C:/Users/kaize/DTU/Deep_Learning/ExamProjekt'  + '/data.pickle', 'rb') as f:
            model = pickle.load(f)
    else:
        train_model(model, train_loader, diffusion, optimizer, epochs=5)
        # Dump the data into a pickle file
        if input("Save model? (y/n)").lower() == "y":
            with open('C:/Users/kaize/DTU/Deep_Learning/ExamProjekt' + '/data.pickle', 'wb') as f:
                pickle.dump(model, f)

# Generate images and visualize
generated_images = generate_images(model, diffusion, num_images=4)

# Plot generated images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(generated_images[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.show()
