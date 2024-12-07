import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from math import isqrt
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.optim as optim
import os
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")
warnings.filterwarnings("ignore", "The parameter 'pretrained' is deprecated since 0.13*.")
warnings.filterwarnings("ignore", "Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13*.")

from scipy.linalg import sqrtm

path = os.getcwd()

epochs = 30 # 1980 epoch for DDPM paper 
subset_indices = range(60000)
batch_size = 128
num_samples = 4
channels = 3
imgSize = (32, 32)

# Transformations for MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define the transformations for the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1] range
])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    blockPrint()
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    # train_subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    enablePrint()

    # Define diffusion parameters
    timesteps = 1000
    diffusion = Diffusion(timesteps=timesteps, beta_start = 0.0001, beta_end = 0.02)


    model = loadModel(diffusion, train_loader)
    model.eval()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"\nNumber of trainable parameters for model: {params:,}\n".replace(",", "."))
    images = reverse_process(model, diffusion, device, num_steps=1000, num_samples=num_samples)

    if input("\n\tPlot generated images? (y/n): ").lower() == "y":
        if input("\n\t(Time plot: 0), (Final images plot: 1)\n\n\t Plot type: ").lower() == "0":
            fig, axes = plt.subplots(len(images), num_samples, figsize=(12, 12))
            for i, img_set in enumerate(images):
                for j, img in enumerate(img_set):
                    if channels == 1:
                        axes[i, j].imshow(img[0], cmap='gray')
                    else:
                        imgVal = np.moveaxis(img[:], 0, -1)
                        imgVal = (imgVal - np.min(imgVal)) / (np.max(imgVal) - np.min(imgVal))
                        axes[i, j].imshow(imgVal)
                    axes[i, j].axis('off')
        else:
            sqrtVal = isqrt(num_samples)
            fig, axes = plt.subplots(sqrtVal, sqrtVal, figsize=(12, 12))
            for i in range(sqrtVal):
                for j in range(sqrtVal):
                    if channels == 1:
                        axes[i, j].imshow(images[-1][i % sqrtVal + j * sqrtVal, 0, :, :], cmap='gray')
                    else:
                        imgVal = np.moveaxis(images[-1][i % sqrtVal + j * sqrtVal, :, :, :], 0, -1)
                        imgVal = (imgVal - np.min(imgVal)) / (np.max(imgVal) - np.min(imgVal))
                        axes[i, j].imshow(imgVal)
                    axes[i, j].axis('off')
        plt.tight_layout()
        plt.savefig(path + "/GeneratedImages.png", dpi = 1200)
        plt.show()

    generated_images = torch.tensor(np.array(images[-1])).to(device)

    # Calculate FID
    fid_score = compute_fid(test_loader, generated_images, device)
    print(f"\nFID Score: {fid_score}\n")

class Diffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device) # self.betas = self.betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.alpha_cumprod_sqrt = torch.sqrt(self.alpha_cumprod)
        self.alpha_cumprod_sqrt_minus = torch.sqrt(1 - self.alpha_cumprod)

        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_cumprod[:-1]]) 

    def forward_process(self, x0, t):
        """
        Adds noise to the data at timestep `t` during the forward process.
        x0: Original data.
        t: Current timestep.
        """

        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.alpha_cumprod[t - 1].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = (1 - self.alpha_cumprod[t - 1]).sqrt().view(-1, 1, 1, 1)
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_t * noise

        return xt, noise

class TimeEmbedding(nn.Module):
    """
    Transform the time embedding into the required output size.
    """
    def __init__(self, 
                 n_out: int, # Output Dimension
                 t_emb_dim: int = 128 # Time Embedding Dimension
                ):
        super().__init__()
        self.te_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, n_out)
        )

    def forward(self, x):
        return self.te_block(x)

def get_time_embedding(
    time_steps: torch.Tensor,
    t_emb_dim: int
) -> torch.Tensor:
    
    """
    Convert a scalar time-step into a vector embedding of a specified dimension.

    Args:
        time_steps (torch.Tensor): A 1D tensor of shape (Batch,) representing the time steps.
        t_emb_dim (int): The desired embedding dimension (e.g., 128).

    Returns:
        torch.Tensor: A tensor of shape (Batch, t_emb_dim) containing the vector embeddings.
    """
    
    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."

    factor = 10000 ** (2 * torch.arange(0, t_emb_dim // 2, 
                        dtype=torch.float32, 
                        device=time_steps.device) / t_emb_dim)

    t_emb = time_steps[:, None] / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)

    return t_emb

class SelfAttentionBlock(nn.Module):
    """
    Perform GroupNorm and Multi-head Self-Attention over the input.
    """
    def __init__(self, 
                 num_channels: int,
                 num_groups: int = 8, 
                 num_heads: int = 8,
                 norm: bool = True
                ):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups, num_channels) if norm else nn.Identity()
        self.attn = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)

    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h * w)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        # x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x

class Downsample(nn.Module):
    """
    Downsample the input by a factor of k across its Height and Width.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 k: int = 2,  # Downsampling factor
                 use_conv: bool = True,  # Use convolution for downsampling
                 use_mpool: bool = True  # Use max-pooling for downsampling
                ):
        super().__init__()
        self.use_conv = use_conv
        self.use_mpool = use_mpool
        
        # Downsampling using Convolution
        self.cv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Conv2d(
                in_channels, 
                out_channels // 2 if use_mpool else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            )
        ) if use_conv else nn.Identity()
        
        # Downsampling using MaxPool
        self.mpool = nn.Sequential(
            nn.MaxPool2d(k, k),
            nn.Conv2d(
                in_channels, 
                out_channels // 2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_mpool else nn.Identity()
        
    def forward(self, x):
        if not self.use_conv:
            return self.mpool(x)
        if not self.use_mpool:
            return self.cv(x)
        return torch.cat([self.cv(x), self.mpool(x)], dim=1)

class Upsample(nn.Module):
    """
    Upsample the input by a factor of k across its Height and Width.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 k: int = 2,  # Upsampling factor
                 use_conv: bool = True,  # Use convolution for upsampling
                 use_upsample: bool = True  # Use nn.Upsample for upsampling
                ):
        super().__init__()
        self.use_conv = use_conv
        self.use_upsample = use_upsample
        
        # Upsampling using Convolution
        self.cv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels // 2 if use_upsample else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            ),
            nn.Conv2d(
                out_channels // 2 if use_upsample else out_channels, 
                out_channels // 2 if use_upsample else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_conv else nn.Identity()
        
        # Upsampling using nn.Upsample
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=k, 
                mode='bilinear', 
                align_corners=False
            ),
            nn.Conv2d(
                in_channels,
                out_channels // 2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_upsample else nn.Identity()
        
    def forward(self, x):
        if not self.use_conv:
            return self.up(x)
        if not self.use_upsample:
            return self.cv(x)
        return torch.cat([self.cv(x), self.up(x)], dim=1)

class NormActConv(nn.Module):
    """
    Apply GroupNorm, Activation, and Convolution layers sequentially.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 num_groups: int = 8, 
                 kernel_size: int = 5, 
                 norm: bool = True,
                 act: bool = True
                ):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups, in_channels) if norm else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class DownC(nn.Module):
    """
    Apply down-convolution to the input using the following sequence:
    1. Convolution + TimeEmbedding
    2. Convolution
    3. Skip-connection from the input x.
    4. Self-Attention
    5. Skip-connection from step 3.
    6. Downsampling
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 t_emb_dim: int = 128,  # Time embedding dimension
                 num_layers: int = 2,
                 down_sample: bool = True,  # Use downsampling if True
                 selfAttention = False
                ):
        super().__init__()
        
        self.num_layers = num_layers
        self.selfAttention = selfAttention
        
        self.conv1 = nn.ModuleList([
            NormActConv(
                in_channels if i == 0 else out_channels, 
                out_channels
            ) for i in range(num_layers)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels) 
            for _ in range(num_layers)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) 
            for _ in range(num_layers)
        ])
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) 
            for _ in range(num_layers)
        ])
        
        self.down_block = Downsample(out_channels, out_channels) if down_sample else nn.Identity()
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers)
        ])
        
    def forward(self, x, t_emb):
        out = x
        
        for i in range(self.num_layers):
            resnet_input = out
            
            # Residual Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # Self-Attention
            if self.selfAttention:
                out_attn = self.attn_block[i](out)
                out = out + out_attn

        # Downsampling
        out = self.down_block(out)
        return out

class MidC(nn.Module):
    """
    Refine the features extracted from the DownC block.
    This refinement is achieved using the following sequence:

    1. A ResNet Block incorporating Time Embedding
    2. A series of Self-Attention followed by ResNet Blocks with Time Embedding
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 t_emb_dim: int = 128,
                 num_layers: int = 2
                ):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.conv1 = nn.ModuleList([
            NormActConv(
                in_channels if i == 0 else out_channels, 
                out_channels
            ) for i in range(num_layers + 1)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels) 
            for _ in range(num_layers + 1)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) 
            for _ in range(num_layers + 1)
        ])
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) 
            for _ in range(num_layers)
        ])
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers + 1)
        ])
        
    def forward(self, x, t_emb):
        out = x
        
        # Initial ResNet Block
        resnet_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        out = self.conv2[0](out)
        out = out + self.res_block[0](resnet_input)
        
        # Sequence of Self-Attention and ResNet Blocks
        for i in range(self.num_layers):
            
            # ResNet Block
            resnet_input = out
            out = self.conv1[i + 1](out)
            out = out + self.te_block[i + 1](t_emb)[:, :, None, None]
            out = self.conv2[i + 1](out)
            out = out + self.res_block[i + 1](resnet_input)
            
        return out

class UpC(nn.Module):
    """
    Perform Up-convolution on the input using the following operations:

    1. Upsampling
    2. Convolution followed by Time Embedding
    3. Additional Convolution
    4. Adding a Skip Connection from the Upsampling step
    5. Self-Attention Layer
    6. Adding another Skip Connection after Step 3
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 t_emb_dim: int = 128,  # Time Embedding Dimension
                 num_layers: int = 2,
                 up_sample: bool = True,  # If True, perform Upsampling
                 selfAttention = False
                ):
        super().__init__()
        
        self.selfAttention = selfAttention
        self.num_layers = num_layers
        
        self.conv1 = nn.ModuleList([
            NormActConv(
                in_channels if i == 0 else out_channels, 
                out_channels
            ) for i in range(num_layers)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, out_channels) 
            for _ in range(num_layers)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) 
            for _ in range(num_layers)
        ])
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels) 
            for _ in range(num_layers)
        ])
        
        self.up_block = Upsample(in_channels, in_channels // 2) if up_sample else nn.Identity()
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i == 0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers)
        ])
        
    def forward(self, x, down_out, t_emb):
        
        # Upsampling and Concatenation with Downsample Output
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            
            # ResNet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)

            # Self-Attention Block
            if self.selfAttention:
                out_attn = self.attn_block[i](out)
                out = out + out_attn
        
        return out       

class UNet(nn.Module):
    """
    U-Net architecture for predicting noise, as described in 
    the "Denoising Diffusion Probabilistic Models" paper.
    
    Structure:
    - Initial convolution for input projection.
    - Series of DownC blocks for feature extraction.
    - MidC blocks for feature refinement.
    - Series of UpC blocks for upsampling and reconstruction.
    - Final convolution to map back to image space.
    """
    
    def __init__(self,
                 im_channels: int, # Input image channels (e.g., 1 for grayscale, 3 for RGB)
                 down_ch: list = [x for x in [64, 128, 256, 512]],  # Channels for DownC blocks
                 mid_ch: list = [x for x in [512, 512, 256]],  # Channels for MidC blocks
                 up_ch: list[int] = [x for x in [512, 256, 128, 64]],  # Channels for UpC blocks
                 down_sample: list[bool] = [True, True, False],  # Downsample flag per DownC block
                 t_emb_dim: int = 512,  # Time embedding dimension
                 num_downc_layers: int = 4,  # Layers in each DownC block
                 num_midc_layers: int = 3,  # Layers in each MidC block
                 num_upc_layers: int = 4,  # Layers in each UpC block
                 attentionBlocks: list[bool] = [False, True, False, False]
                ):
        super(UNet, self).__init__()
        
        # Save hyperparameters
        self.im_channels = im_channels
        self.down_ch = down_ch
        self.mid_ch = mid_ch
        self.up_ch = up_ch
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.num_downc_layers = num_downc_layers
        self.num_midc_layers = num_midc_layers
        self.num_upc_layers = num_upc_layers
        self.attentionBlocks = attentionBlocks

        # Reverse downsampling pattern for upsampling
        self.up_sample = list(reversed(self.down_sample))  # [False, True, True]

        # Initial convolution
        self.cv1 = nn.Sequential(
            nn.Conv2d(self.im_channels, self.down_ch[0], kernel_size=5, padding=2),
        )

        # Time embedding projection
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim), 
            nn.SiLU(), 
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # DownC blocks
        self.downs = nn.ModuleList([
            DownC(
                self.down_ch[i], 
                self.down_ch[i + 1], 
                self.t_emb_dim, 
                self.num_downc_layers, 
                self.down_sample[i],
                attentionBlocks[i]
            ) for i in range(len(self.down_ch) - 1)
        ])

        # MidC blocks
        self.mids = nn.ModuleList([
            MidC(
                self.mid_ch[i], 
                self.mid_ch[i + 1], 
                self.t_emb_dim, 
                self.num_midc_layers
            ) for i in range(len(self.mid_ch) - 1)
        ])

        # UpC blocks
        self.ups = nn.ModuleList([
            UpC(
                self.up_ch[i], 
                self.up_ch[i + 1], 
                self.t_emb_dim, 
                self.num_upc_layers, 
                self.up_sample[i],
                list(reversed(attentionBlocks))[i]
            ) for i in range(len(self.up_ch) - 1)
        ])

        # Final convolution
        self.cv2 = nn.Sequential(
            nn.GroupNorm(8, self.up_ch[-1]), 
            nn.Conv2d(self.up_ch[-1], self.im_channels, kernel_size=5, padding=2)
        ) 
        
    def forward(self, x, t):
        """
        Forward pass through the U-Net.

        Parameters:
        - x: Input image tensor of shape (batch_size, im_channels, H, W).
        - t: Time step tensor for diffusion process, shape (batch_size,).

        Returns:
        - Output tensor of shape (batch_size, im_channels, H, W).
        """
        # Initial convolution
        out = self.cv1(x)
        
        # Time embedding
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        # DownC outputs for skip connections
        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)
        
        # MidC processing
        for mid in self.mids:
            out = mid(out, t_emb)
        
        # UpC blocks with skip connections
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            
        # Final convolution
        out = self.cv2(out)
        
        return out

def train_model(model, data_loader, diffusion, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        i = 1
        for x0, _ in (pbar := tqdm(data_loader, desc = f"Epoch {epoch + 1} / {epochs}")):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use cpu
            x0 = x0.to(device)  # Load batch to GPU
            t = torch.randint(1, diffusion.timesteps + 1, (x0.size(0),)).to(device)  # Random timesteps
            xt, noise = diffusion.forward_process(x0, t)  # Forward process

            # aVal = diffusion.alpha_cumprod_sqrt[t-1].view(-1, 1, 1, 1)
            # bVal = diffusion.alpha_cumprod_sqrt_minus[t-1].view(-1, 1, 1, 1)
            # predicted_noise = model(aVal * x0 + bVal * noise, t)  # Model noiseprediction
            predicted_noise = model(xt, t)
            
            loss = F.mse_loss(predicted_noise, noise) # Compute loss: MSE between predicted and true noise
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"Avg Loss: {(epoch_loss / i):.3f}")
            i += 1
            
            epoch_loss += loss.item()

# Reverse process for generating images
def reverse_process(model, diffusion, device, num_steps=1000, num_samples=10):
    with torch.no_grad():
        x_t = torch.randn((num_samples, channels, imgSize[0], imgSize[1]), device=device)  # Sample noise from standard normal
        images = []
 
        timesteps = torch.linspace(num_steps, 1, num_samples, dtype=torch.long) # Define equally spaced timesteps

        for t in (pbar := tqdm(range(num_steps, 0, -1), desc="Generating Images")):
            pbar.set_postfix_str(f"t = {t}")
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            epsilon_theta = model(x_t, t_tensor) # Predict noise using the trained model

            # Calculate constants
            alpha_t = diffusion.alphas[t - 1]
            alpha_bar_t = diffusion.alpha_cumprod[t - 1]
            alpha_bar_t_prev = diffusion.alpha_cumprod[t - 2] if t > 1 else torch.tensor(1.0, device=x_t.device)
            beta_t = diffusion.betas[t - 1]

            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            sqrt_alpha_t_inv = 1 / torch.sqrt(alpha_t)

            # Mean calculation
            mu_theta = sqrt_alpha_t_inv * (x_t - beta_t / sqrt_one_minus_alpha_bar_t * epsilon_theta)

            if t > 1:
                # Variance for stochasticity
                sigma_t = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
            else:
                sigma_t = 0
                z = 0

            # Sample x_{t-1}
            x_t = mu_theta + sigma_t * z

            # Save intermediate images at specific timesteps
            if t in timesteps:
                images.append(x_t.detach().cpu().numpy())
    return images

# Function to calculate mean and covariance of features
def calculate_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

# Function to calculate FID
def calculate_fid(mu1, sigma1, mu2, sigma2):
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False) # Compute the square root of the product of covariance matrices
    
    # Numerical stability check
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Compute FID score
    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def extract_features(data_loader, model, device):
    import torchvision.transforms as T

    model.eval()
    features = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc = "Extracting features"):
            images = batch[0]
            images = images.to(device)

            if images.shape[1] == 1:  # For MNIST (single channel), convert to 3-channel
                images = images.repeat(1, 3, 1, 1)
            
            images = T.Resize((299, 299))(images) # Resize images to 299x299 (InceptionV3 input size)

            # Extract features
            feature = model(images).detach().cpu().numpy()
            features.append(feature)
    return np.vstack(features)

def compute_fid(test_loader, generated_images, device):
    from torchvision.models import inception_v3

    # Load InceptionV3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = nn.Identity()  # Remove the classification head
    
    # Extract features for real images
    print("\nTraining data")
    real_features = extract_features(test_loader, inception_model, device)
    mu_real, sigma_real = calculate_statistics(real_features)

    # Create a DataLoader for generated images
    print("\nGenerated data")
    generated_dataset = torch.utils.data.TensorDataset(generated_images)
    generated_loader = DataLoader(generated_dataset, batch_size=batch_size, shuffle=False)

    # Extract features for generated images
    fake_features = extract_features(generated_loader, inception_model, device)
    mu_fake, sigma_fake = calculate_statistics(fake_features)

    # Calculate FID
    fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__

def loadModel(diffusion, train_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use cpu
    model = UNet(im_channels = channels).to(device)
    files = []
    for f in os.listdir(path):
        fPath = os.path.join(path, f)
        if os.path.isfile(fPath) and f[-4:] == ".pth":
            files.append(f)
        elif os.path.isdir(fPath):
            for z in os.listdir(fPath):
                zPath = os.path.join(fPath, z)
                if os.path.isfile(zPath) and z[-4:] == ".pth":
                    files.append(os.path.join(f, z))
    #files = [f for f  in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and f[-4:] == ".pth")]

    if len(files) == 0:
        modelStr =  ".train"
    else:
        keyVals = {i: files[i] for i in range(len(files))}
        namesStr = ", ".join([f"({f[:-4]}: {i})" for i, f in keyVals.items()])
        optionsStr = "\n\t" + namesStr  + ", (Train: *)\n\n\tModel: "
        optionsInt = -1
        result = input(optionsStr)
        print("")
        try:
            optionsInt = int(result)
        except:
            optionsInt = -1
         
        modelStr =  keyVals.get(optionsInt, ".train")
    
    if modelStr == ".train":
        optimizer = optim.Adam(model.parameters(), lr=2e-4)
        train_model(model, train_loader, diffusion, optimizer, epochs=epochs)
        if input("\n\tSave model? (y/n): ").lower() == "y" or True:
            inputStr = f"\n\tCurrent Names: {namesStr} |\n\n\tNew Name: " if len(files) > 0 else f"\n\tNew Name: "
            newName = input(inputStr)
            print("")
            torch.save(model, path + f'/{newName}.pth')
    else:
        model = torch.load(path + f"/{modelStr}").to(device)
    return model
        
main()