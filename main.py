import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import argparse
from typing import Dict, List, Tuple, Optional

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ResidualBlock(nn.Module):
    """Basic residual block with group normalization and SiLU activation"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + shortcut
        x = self.activation(x)
        return x

class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        k = self.k(h).view(B, C, -1)  # (B, C, H*W)
        v = self.v(h).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        
        attention = torch.bmm(q, k) * (C ** -0.5)  # (B, H*W, H*W)
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.bmm(attention, v)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        out = self.proj_out(out)
        return x + out

class UNet(nn.Module):
    """UNet architecture for diffusion model - adapted for MNIST"""
    def __init__(self, in_channels=1, out_channels=1, base_channels=32, channel_mults=(1, 2, 4, 8)):
        super().__init__()
        self.in_channels = in_channels
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels = base_channels
        for mult in channel_mults:
            out_channels_block = base_channels * mult
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(channels, out_channels_block),
                AttentionBlock(out_channels_block),
                ResidualBlock(out_channels_block, out_channels_block)
            ]))
            self.down_samples.append(nn.Conv2d(out_channels_block, out_channels_block, kernel_size=3, stride=2, padding=1))
            channels = out_channels_block
        
        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock(channels, channels),
            AttentionBlock(channels),
            ResidualBlock(channels, channels)
        ])
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            out_channels_block = base_channels * mult
            self.up_samples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(channels, out_channels_block, kernel_size=3, padding=1)
            ))
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(channels + out_channels_block, out_channels_block),
                AttentionBlock(out_channels_block),
                ResidualBlock(out_channels_block, out_channels_block)
            ]))
            channels = out_channels_block
        
        # Final layers
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_activation = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, time_emb=None):
        # Initial convolution
        x = self.init_conv(x)
        
        # Downsample path
        skip_connections = []
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            for layer in down_block:
                x = layer(x)
            skip_connections.append(x)
            x = down_sample(x)
        
        # Middle path
        for layer in self.mid_blocks:
            x = layer(x)
        
        # Upsample path
        for up_sample, up_block in zip(self.up_samples, self.up_blocks):
            x = up_sample(x)
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            for layer in up_block:
                x = layer(x)
        
        # Final layers
        x = self.final_norm(x)
        x = self.final_activation(x)
        x = self.final_conv(x)
        return x

class DDPM:
    """Denoising Diffusion Probabilistic Model for compressed sensing"""
    def __init__(self, model, device, beta_start=1e-4, beta_end=0.02, num_timesteps=1000):
        self.model = model.to(device)
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Create beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
    def sample_timesteps(self, batch_size):
        """Sample random timesteps for training"""
        return torch.randint(1, self.num_timesteps, (batch_size,), device=self.device).long()
    
    def noise_image(self, x_0, t):
        """Add noise to image at timestep t"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None]
        
        noise = torch.randn_like(x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
    
    def p_sample(self, x, t, measurement, A, measurement_weight=1.0):
        """Sample from p(x_{t-1} | x_t) with measurement guidance"""
        with torch.no_grad():
            # Predict noise
            pred_noise = self.model(x, t)
            
            # Calculate x_0 estimate
            x_0_est = (x - torch.sqrt(1 - self.alpha_bars[t])[:, None, None, None] * pred_noise) / torch.sqrt(self.alpha_bars[t])[:, None, None, None]
            
            # Apply measurement consistency
            if measurement is not None and A is not None:
                # Flatten image
                x_flat = x_0_est.view(x_0_est.size(0), -1)
                
                # Calculate measurement error
                measurement_error = A @ x_flat.t() - measurement.unsqueeze(0)
                
                # Calculate gradient of measurement error w.r.t. x_0
                grad = 2 * A.t() @ measurement_error.t() / A.shape[0]
                
                # Reshape gradient to image dimensions
                grad = grad.view(x_0_est.shape)
                
                # Apply gradient guidance
                x_0_est = x_0_est - measurement_weight * grad
            
            # Clamp to valid range
            x_0_est = torch.clamp(x_0_est, -1, 1)
            
            # Calculate mean and variance for p(x_{t-1} | x_t, x_0)
            mean = (1 / torch.sqrt(self.alphas[t]))[:, None, None, None] * (
                x - (self.betas[t] / torch.sqrt(1 - self.alpha_bars[t]))[:, None, None, None] * pred_noise
            )
            
            if t[0] > 0:
                variance = (1 - self.alpha_bars[t-1]) / (1 - self.alpha_bars[t]) * self.betas[t]
                variance = variance[:, None, None, None]
                noise = torch.randn_like(x)
                x_prev = mean + torch.sqrt(variance) * noise
            else:
                x_prev = mean
                
            return x_prev, x_0_est
    
    def sample(self, measurement, A, measurement_weight=1.0, batch_size=1, img_shape=(1, 28, 28)):
        """Sample from the diffusion model with measurement guidance"""
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(batch_size, *img_shape, device=self.device)
        
        x_0_estimates = []
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t] * batch_size, device=self.device)
            x, x_0_est = self.p_sample(x, t_batch, measurement, A, measurement_weight)
            
            if t % 100 == 0 or t < 10:
                x_0_estimates.append(x_0_est.detach().cpu())
        
        return x, x_0_estimates

def generate_random_matrix(m, n):
    """Generate random Gaussian measurement matrix"""
    return torch.randn(m, n) / np.sqrt(m)

def l1_recovery(y, A, lambda_val=0.1, iterations=1000):
    """L1 minimization recovery using ISTA algorithm"""
    m, n = A.shape
    x = torch.zeros(n, device=A.device)
    A_t = A.t()
    
    # Calculate step size
    L = torch.norm(A, p=2)**2
    t = 1.0 / L
    
    losses = []
    for i in range(iterations):
        # Gradient step
        x_prev = x.clone()
        gradient = A_t @ (A @ x - y)
        x = x - t * gradient
        
        # Soft thresholding (proximal operator for L1)
        x = torch.sign(x) * torch.relu(torch.abs(x) - lambda_val * t)
        
        # Calculate loss
        loss = 0.5 * torch.norm(A @ x - y)**2 + lambda_val * torch.norm(x, 1)
        losses.append(loss.item())
        
        # Check for convergence
        if torch.norm(x - x_prev) < 1e-6:
            break
    
    return x, losses

def evaluate_recovery(original, recovered):
    """Evaluate recovery quality using PSNR and SSIM"""
    # Convert to numpy and denormalize
    original_np = original.cpu().numpy().reshape(28, 28)
    recovered_np = recovered.cpu().numpy().reshape(28, 28)
    
    # Denormalize from [-1, 1] to [0, 1]
    original_np = (original_np + 1) / 2
    recovered_np = (recovered_np + 1) / 2
    
    # Calculate metrics
    psnr = peak_signal_noise_ratio(original_np, recovered_np, data_range=1.0)
    ssim = structural_similarity(original_np, recovered_np, data_range=1.0)
    
    return psnr, ssim

def plot_comparison(original, l1_recovered, ddpm_recovered, title, save_path):
    """Plot comparison of original and recovered images"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Denormalize images
    original = (original.reshape(28, 28) + 1) / 2
    l1_recovered = (l1_recovered.reshape(28, 28) + 1) / 2
    ddpm_recovered = (ddpm_recovered.reshape(28, 28) + 1) / 2
    
    axes[0].imshow(original.cpu().numpy(), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(l1_recovered.cpu().numpy(), cmap='gray')
    axes[1].set_title('L1 Recovery')
    axes[1].axis('off')
    
    axes[2].imshow(ddpm_recovered.cpu().numpy(), cmap='gray')
    axes[2].set_title('DDPM Recovery')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_progress(x_0_estimates, save_path):
    """Plot the progression of x_0 estimates through the reverse process"""
    n_steps = len(x_0_estimates)
    n_cols = min(5, n_steps)
    n_rows = (n_steps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, x_0_est in enumerate(x_0_estimates):
        row, col = i // n_cols, i % n_cols
        img = (x_0_est[0].squeeze().numpy() + 1) / 2
        axes[row, col].imshow(np.clip(img, 0, 1), cmap='gray')
        axes[row, col].set_title(f'Step {i * 100 if i < len(x_0_estimates) - 1 else "Final"}')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_steps, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_mnist_data(batch_size=32):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST is grayscale
    ])
    
    # Download and load training data
    train_data = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_data = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_ddpm(ddpm, train_loader, num_epochs=10, device='cpu'):
    """Train the DDPM model"""
    ddpm.model.train()
    optimizer = optim.Adam(ddpm.model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Sample timesteps
            t = ddpm.sample_timesteps(images.shape[0])
            
            # Add noise to images
            x_t, noise = ddpm.noise_image(images, t)
            
            # Predict noise
            pred_noise = ddpm.model(x_t, t)
            
            # Calculate loss
            loss = criterion(pred_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    return losses

def main():
    parser = argparse.ArgumentParser(description='DDPM Compressed Sensing Experiment on MNIST')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='./ddpm_mnist.pth', help='Path to DDPM model')
    parser.add_argument('--output_dir', type=str, default='./ddpm_mnist_results', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to test')
    parser.add_argument('--measurement_ratio', type=float, default=0.1, help='Ratio of measurements to dimension')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--measurement_weight', type=float, default=1.0, help='Weight for measurement guidance')
    parser.add_argument('--train', action='store_true', default=False, help='Train the DDPM model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST data
    train_loader, test_loader = load_mnist_data(batch_size=args.num_samples)
    
    # Create DDPM model
    unet = UNet(in_channels=1, out_channels=1, base_channels=32)
    ddpm = DDPM(unet, device, num_timesteps=1000)
    
    # Train or load the model
    if args.train:
        print("Training DDPM model...")
        losses = train_ddpm(ddpm, train_loader, num_epochs=args.epochs, device=device)
        torch.save(ddpm.model.state_dict(), args.model_path)
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('DDPM Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
        plt.close()
    else:
        try:
            ddpm.model.load_state_dict(torch.load(args.model_path, map_location=device))
            print("DDPM model loaded successfully.")
        except:
            print("Warning: Could not load DDPM model. Using random weights.")
    
    # Image dimension
    img_dim = 1 * 28 * 28  # 1 channel, 28x28 image
    m = int(args.measurement_ratio * img_dim)  # Number of measurements
    
    print(f"Image dimension: {img_dim}")
    print(f"Number of measurements: {m} ({(args.measurement_ratio * 100):.1f}%)")
    
    # Generate random measurement matrix
    A = generate_random_matrix(m, img_dim).to(device)
    
    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    # Flatten images
    x_true = images.view(images.size(0), -1)
    
    # Take measurements
    y = torch.matmul(x_true, A.t())
    
    # Add a small amount of noise
    noise_std = 0.01 * torch.std(y)
    y = y + noise_std * torch.randn_like(y)
    
    # Initialize results
    results = {
        'psnr_l1': [], 'ssim_l1': [], 
        'psnr_ddpm': [], 'ssim_ddpm': [],
        'l1_losses': [], 
        'images': []
    }
    
    # Process each image in the batch
    for i in range(min(args.num_samples, images.size(0))):
        print(f"Processing image {i+1}/{min(args.num_samples, images.size(0))}")
        
        # Get single image and measurement
        x_i = x_true[i]
        y_i = y[i]
        
        # L1 recovery
        start_time = time.time()
        x_l1, l1_losses = l1_recovery(y_i, A, lambda_val=0.1, iterations=1000)
        l1_time = time.time() - start_time
        
        # DDPM recovery
        start_time = time.time()
        x_ddpm, x_0_estimates = ddpm.sample(
            y_i, A, 
            measurement_weight=args.measurement_weight,
            batch_size=1, 
            img_shape=(1, 28, 28)
        )
        ddpm_time = time.time() - start_time
        
        # Flatten DDPM output
        x_ddpm_flat = x_ddpm.view(-1)
        
        # Evaluate recoveries
        psnr_l1, ssim_l1 = evaluate_recovery(x_i, x_l1)
        psnr_ddpm, ssim_ddpm = evaluate_recovery(x_i, x_ddpm_flat)
        
        # Store results
        results['psnr_l1'].append(psnr_l1)
        results['ssim_l1'].append(ssim_l1)
        results['psnr_ddpm'].append(psnr_ddpm)
        results['ssim_ddpm'].append(ssim_ddpm)
        results['l1_losses'].append(l1_losses)
        results['images'].append((x_i, x_l1, x_ddpm_flat))
        
        print(f"  L1: PSNR={psnr_l1:.2f}, SSIM={ssim_l1:.3f}, Time={l1_time:.2f}s")
        print(f"  DDPM: PSNR={psnr_ddpm:.2f}, SSIM={ssim_ddpm:.3f}, Time={ddpm_time:.2f}s")
        
        # Plot comparison for this image
        plot_comparison(
            x_i, x_l1, x_ddpm_flat, 
            f"Sample {i+1} (Digit: {labels[i].item()})\nL1: PSNR={psnr_l1:.2f}, SSIM={ssim_l1:.3f}\nDDPM: PSNR={psnr_ddpm:.2f}, SSIM={ssim_ddpm:.3f}",
            os.path.join(args.output_dir, f"sample_{i+1}_comparison.png")
        )
        
        # Plot progression of x_0 estimates
        plot_progress(
            x_0_estimates,
            os.path.join(args.output_dir, f"sample_{i+1}_progress.png")
        )
    
    # Plot overall results
    plot_overall_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    avg_psnr_l1 = np.mean(results['psnr_l1'])
    avg_ssim_l1 = np.mean(results['ssim_l1'])
    avg_psnr_ddpm = np.mean(results['psnr_ddpm'])
    avg_ssim_ddpm = np.mean(results['ssim_ddpm'])
    
    print(f"L1:     Avg PSNR = {avg_psnr_l1:.2f}, Avg SSIM = {avg_ssim_l1:.3f}")
    print(f"DDPM:   Avg PSNR = {avg_psnr_ddpm:.2f}, Avg SSIM = {avg_ssim_ddpm:.3f}")
    
    if avg_psnr_ddpm > avg_psnr_l1:
        improvement = ((avg_psnr_ddpm - avg_psnr_l1) / avg_psnr_l1) * 100
        print(f"DDPM improves PSNR by {improvement:.1f}%")
    else:
        degradation = ((avg_psnr_l1 - avg_psnr_ddpm) / avg_psnr_l1) * 100
        print(f"DDPM degrades PSNR by {degradation:.1f}%")
    
    print("\nExperiment completed. Results saved to:", args.output_dir)

def plot_overall_results(results, output_dir):
    """Plot overall results comparing performance"""
    # Create bar plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['L1', 'DDPM']
    avg_psnr = [np.mean(results['psnr_l1']), np.mean(results['psnr_ddpm'])]
    avg_ssim = [np.mean(results['ssim_l1']), np.mean(results['ssim_ddpm'])]
    
    x = np.arange(len(methods))
    width = 0.35
    
    ax1.bar(x, avg_psnr, width)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Average PSNR by Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    
    ax2.bar(x, avg_ssim, width)
    ax2.set_xlabel('Method')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Average SSIM by Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_results.png'))
    plt.close()
    
    # Plot loss curves for L1
    plt.figure(figsize=(8, 5))
    for i, losses in enumerate(results['l1_losses']):
        plt.plot(losses, label=f'Sample {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('L1 Recovery Loss Curves')
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'l1_loss_curves.png'))
    plt.close()

if __name__ == '__main__':
    main()



# install the requirements                                                                   ---> pip install torch torchvision matplotlib scikit-image
# Run the code with                                                                          ---> python ddpm_mnist.py --train --epochs 10
# After training, you can run the compressed sensing experiment                              ---> python ddpm_mnist.py --measurement_ratio 0.1 --num_samples 5
