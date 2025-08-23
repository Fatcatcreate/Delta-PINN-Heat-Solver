#!/usr/bin/env python3
"""
Delta-PINN implementation for 3D heat diffusion in irregular domains
Advanced architecture with residual connections and adaptive weights
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import csv
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from domain_shapes import Domain3D, DomainFactory, generate_interior_points, generate_boundary_points

def set_seed(seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)

class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for better high-frequency learning"""
    
    def __init__(self, input_dim: int, embed_dim: int, scale: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Fixed random frequencies
        self.register_buffer('B', torch.randn(input_dim, embed_dim // 2) * scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, input_dim]
        x_proj = 2 * np.pi * x @ self.B  # [N, embed_dim//2]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, hidden_dim: int, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))

class DeltaPINN3D(nn.Module):
    """Delta-PINN for 3D heat diffusion with advanced architecture"""
    
    def __init__(self, hidden_size: int = 128, num_layers: int = 6, 
                 use_fourier: bool = True, fourier_scale: float = 1.0,
                 use_residual: bool = True):
        super().__init__()
        
        self.use_fourier = use_fourier
        self.use_residual = use_residual
        
        # Input processing
        if use_fourier:
            self.fourier_embed = FourierFeatureEmbedding(4, hidden_size, fourier_scale)  # (x,y,z,t)
            input_dim = hidden_size
        else:
            input_dim = 4
            self.input_layer = nn.Linear(4, hidden_size)
        
        # Hidden layers
        layers = []
        for i in range(num_layers):
            if use_residual and i > 0:
                layers.append(ResidualBlock(hidden_size))
            else:
                layers.append(nn.Linear(hidden_size if i > 0 or use_fourier else input_dim, hidden_size))
                layers.append(nn.Tanh())
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)
        
        # Adaptive loss weights (trainable)
        self.log_sigma_pde = nn.Parameter(torch.zeros(1))
        self.log_sigma_bc = nn.Parameter(torch.zeros(1))
        self.log_sigma_ic = nn.Parameter(torch.zeros(1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization with special handling for residual blocks"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(init_weights)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Normalize inputs to [-1, 1] for better convergence
        x_norm = 2 * x - 1  
        y_norm = 2 * y - 1
        z_norm = 2 * z - 1
        t_norm = 2 * t - 1
        
        inputs = torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)
        
        # Feature embedding
        if self.use_fourier:
            h = self.fourier_embed(inputs)
        else:
            h = torch.tanh(self.input_layer(inputs))
        
        # Hidden layers
        for layer in self.hidden_layers:
            if isinstance(layer, ResidualBlock):
                h = layer(h)
            elif isinstance(layer, nn.Linear):
                h = layer(h)
            else:  # Activation
                h = layer(h)
        
        return self.output_layer(h)
    
    def get_adaptive_weights(self) -> Tuple[float, float, float]:
        """Get current adaptive loss weights"""
        return (
            torch.exp(-self.log_sigma_pde).item(),
            torch.exp(-self.log_sigma_bc).item(), 
            torch.exp(-self.log_sigma_ic).item()
        )

def create_multi_source_ic(heat_sources: List[Dict], X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Create multi-source initial condition"""
    ic = np.zeros_like(X)
    
    for source in heat_sources:
        x0, y0, z0 = source['position']
        amplitude = source['amplitude']
        sigma = source.get('sigma', 0.05)
        
        # Gaussian heat source
        ic += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2))
    
    return ic

def generate_training_data_3d(domain: Domain3D, n_pde: int = 8000, n_bc: int = 2000, 
                             n_ic: int = 2000, device: str = 'cpu') -> Tuple:
    """Generate 3D training data with domain constraints"""
    
    # PDE interior points
    x_pde, y_pde, z_pde = generate_interior_points(domain, n_pde, device)
    t_pde = torch.rand(len(x_pde), 1, device=device, requires_grad=True)
    
    # Boundary points
    x_bc, y_bc, z_bc = generate_boundary_points(domain, n_bc, device)
    t_bc = torch.rand(len(x_bc), 1, device=device, requires_grad=True)
    
    # Initial condition points (t=0)
    x_ic, y_ic, z_ic = generate_interior_points(domain, n_ic, device)
    t_ic = torch.zeros(len(x_ic), 1, device=device, requires_grad=True)
    
    return (x_pde, y_pde, z_pde, t_pde), (x_bc, y_bc, z_bc, t_bc), (x_ic, y_ic, z_ic, t_ic)

def compute_pde_loss_3d(model: DeltaPINN3D, x: torch.Tensor, y: torch.Tensor, 
                       z: torch.Tensor, t: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """3D heat equation residual: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)"""
    u = model(x, y, z, t)
    
    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                             create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                             create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                             create_graph=True, retain_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), 
                             create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                              create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), 
                              create_graph=True, retain_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), 
                              create_graph=True, retain_graph=True)[0]
    
    # PDE residual
    laplacian = u_xx + u_yy + u_zz
    pde_residual = u_t - alpha * laplacian
    
    return torch.mean(pde_residual**2)

def compute_bc_loss_3d(model: DeltaPINN3D, x: torch.Tensor, y: torch.Tensor, 
                      z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Boundary condition loss (Dirichlet u=0)"""
    u_bc = model(x, y, z, t)
    return torch.mean(u_bc**2)

def compute_ic_loss_3d(model: DeltaPINN3D, x: torch.Tensor, y: torch.Tensor, 
                      z: torch.Tensor, t: torch.Tensor, heat_sources: List[Dict]) -> torch.Tensor:
    """Initial condition loss with multi-source support"""
    u_ic = model(x, y, z, t)
    
    # Convert to numpy for IC computation
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    z_np = z.detach().cpu().numpy().flatten()
    
    # Create true IC values for each point
    u_true_list = []
    for i in range(len(x_np)):
        u_val = 0.0
        for source in heat_sources:
            x0, y0, z0 = source['position']
            amplitude = source['amplitude']
            sigma = source.get('sigma', 0.05)
            
            # Gaussian heat source
            u_val += amplitude * np.exp(-((x_np[i] - x0)**2 + (y_np[i] - y0)**2 + (z_np[i] - z0)**2) / (2 * sigma**2))
        
        u_true_list.append(u_val)
    
    u_true_tensor = torch.tensor(u_true_list, dtype=torch.float32, device=x.device).unsqueeze(1)
    
    return torch.mean((u_ic - u_true_tensor)**2)

class CurriculumScheduler:
    """Curriculum learning scheduler for progressive training"""
    
    def __init__(self, initial_complexity: float = 0.1, max_complexity: float = 1.0, 
                 warmup_epochs: int = 1000):
        self.initial_complexity = initial_complexity
        self.max_complexity = max_complexity
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step(self) -> float:
        """Get current complexity factor"""
        if self.current_epoch < self.warmup_epochs:
            complexity = self.initial_complexity + (self.max_complexity - self.initial_complexity) * \
                        (self.current_epoch / self.warmup_epochs)
        else:
            complexity = self.max_complexity
        
        self.current_epoch += 1
        return complexity

def train_delta_pinn_3d(args, domain: Domain3D, heat_sources: List[Dict]) -> DeltaPINN3D:
    """Main training loop for 3D Delta-PINN"""
    
    set_seed(args.seed)
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize model
    model = DeltaPINN3D(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        use_fourier=args.use_fourier,
        fourier_scale=args.fourier_scale,
        use_residual=args.use_residual
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-6)
    
    # Curriculum scheduler
    curriculum = CurriculumScheduler(warmup_epochs=args.warmup_epochs)
    
    # Training data
    pde_data, bc_data, ic_data = generate_training_data_3d(
        domain, args.n_pde, args.n_bc, args.n_ic, device
    )
    
    # Training history
    history = {
        'total_loss': [],
        'pde_loss': [],
        'bc_loss': [],
        'ic_loss': [],
        'adaptive_weights': []
    }
    
    best_loss = float('inf')
    best_model_state = None
    
    print(f"Starting training for {args.epochs} epochs...")
    
    with tqdm(range(args.epochs), desc="Training") as pbar:
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            
            # Get curriculum complexity
            complexity = curriculum.step()
            
            # Compute losses
            pde_loss = compute_pde_loss_3d(model, *pde_data, args.alpha)
            bc_loss = compute_bc_loss_3d(model, *bc_data)
            ic_loss = compute_ic_loss_3d(model, *ic_data, heat_sources)
            
            # Adaptive weights
            w_pde, w_bc, w_ic = model.get_adaptive_weights()
            
            # Total loss with curriculum learning
            total_loss = complexity * (w_pde * pde_loss + w_bc * bc_loss + w_ic * ic_loss)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(total_loss)
            
            # Record history
            history['total_loss'].append(total_loss.item())
            history['pde_loss'].append(pde_loss.item())
            history['bc_loss'].append(bc_loss.item())
            history['ic_loss'].append(ic_loss.item())
            history['adaptive_weights'].append([w_pde, w_bc, w_ic])
            
            # Save best model
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_model_state = model.state_dict().copy()
            
            # Update progress bar
            pbar.set_postfix({
                'Total': f'{total_loss.item():.2e}',
                'PDE': f'{pde_loss.item():.2e}',
                'BC': f'{bc_loss.item():.2e}',
                'IC': f'{ic_loss.item():.2e}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log progress
            if epoch % args.log_interval == 0:
                print(f"\nEpoch {epoch}/{args.epochs}")
                print(f"Total Loss: {total_loss.item():.6e}")
                print(f"PDE Loss: {pde_loss.item():.6e}")
                print(f"BC Loss: {bc_loss.item():.6e}")
                print(f"IC Loss: {ic_loss.item():.6e}")
                print(f"Adaptive Weights: PDE={w_pde:.3f}, BC={w_bc:.3f}, IC={w_ic:.3f}")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with loss: {best_loss:.6e}")
    
    # Save model and history
    os.makedirs(args.save_dir, exist_ok=True)
    
    model_path = os.path.join(args.save_dir, f"delta_pinn_3d_{domain.name.lower()}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'args': vars(args),
        'domain_name': domain.name,
        'heat_sources': heat_sources
    }, model_path)
    
    # Save training history to CSV
    history_path = os.path.join(args.save_dir, f"training_history_{domain.name.lower()}.csv")
    with open(history_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Total_Loss', 'PDE_Loss', 'BC_Loss', 'IC_Loss', 
                        'Weight_PDE', 'Weight_BC', 'Weight_IC'])
        
        for i, (total, pde, bc, ic, weights) in enumerate(zip(
            history['total_loss'], history['pde_loss'], 
            history['bc_loss'], history['ic_loss'], history['adaptive_weights']
        )):
            writer.writerow([i, total, pde, bc, ic, weights[0], weights[1], weights[2]])
    
    print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
    
    return model

def load_trained_model(model_path: str, device: str = 'cpu') -> Tuple[DeltaPINN3D, Dict]:
    """Load trained model and metadata"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model with saved args
    args_dict = checkpoint['args']
    model = DeltaPINN3D(
        hidden_size=args_dict['hidden_size'],
        num_layers=args_dict['num_layers'],
        use_fourier=args_dict['use_fourier'],
        fourier_scale=args_dict['fourier_scale'],
        use_residual=args_dict['use_residual']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def predict_solution_3d(model: DeltaPINN3D, domain: Domain3D, 
                       t_eval: float, nx: int = 32, ny: int = 32, nz: int = 32,
                       device: str = 'cpu') -> Dict[str, np.ndarray]:
    """Predict solution on regular grid at given time"""
    
    # Generate grid
    grid_data = domain.generate_grid(nx, ny, nz)
    X, Y, Z, mask = grid_data['X'], grid_data['Y'], grid_data['Z'], grid_data['mask']
    
    # Flatten for network input
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()
    t_flat = np.full_like(x_flat, t_eval)
    
    # Convert to tensors
    x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device).unsqueeze(1)
    y_tensor = torch.tensor(y_flat, dtype=torch.float32, device=device).unsqueeze(1)
    z_tensor = torch.tensor(z_flat, dtype=torch.float32, device=device).unsqueeze(1)
    t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Predict
    model.eval()
    with torch.no_grad():
        u_pred = model(x_tensor, y_tensor, z_tensor, t_tensor)
    
    # Reshape back to grid
    u_pred_grid = u_pred.cpu().numpy().reshape(X.shape)
    
    # Apply domain mask (set exterior to NaN)
    u_pred_grid[~mask] = np.nan
    
    return {
        'x': grid_data['x'],
        'y': grid_data['y'], 
        'z': grid_data['z'],
        'X': X,
        'Y': Y,
        'Z': Z,
        'u': u_pred_grid,
        'mask': mask,
        't': t_eval
    }

def main():
    parser = argparse.ArgumentParser(description="Train 3D Delta-PINN for heat diffusion")
    
    # Model parameters
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of hidden layers')
    parser.add_argument('--use_fourier', action='store_true', default=True, help='Use Fourier features')
    parser.add_argument('--fourier_scale', type=float, default=1.0, help='Fourier feature scale')
    parser.add_argument('--use_residual', action='store_true', default=True, help='Use residual connections')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=1000, help='Curriculum warmup epochs')
    parser.add_argument('--alpha', type=float, default=0.01, help='Thermal diffusivity')
    
    # Data parameters
    parser.add_argument('--n_pde', type=int, default=8000, help='Number of PDE collocation points')
    parser.add_argument('--n_bc', type=int, default=2000, help='Number of boundary points')
    parser.add_argument('--n_ic', type=int, default=2000, help='Number of initial condition points')
    
    # Domain and sources
    parser.add_argument('--domain', type=str, default='sphere', 
                       choices=['cube', 'sphere', 'lshape', 'torus', 'cylinder_holes'],
                       help='Domain shape')
    parser.add_argument('--num_sources', type=int, default=2, help='Number of heat sources')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto', help='Device: cpu, cuda, mps, or auto')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./models', help='Save directory')
    parser.add_argument('--log_interval', type=int, default=500, help='Logging interval')
    
    args = parser.parse_args()
    
    print("=== 3D Delta-PINN Heat Diffusion Training ===")
    print(f"Domain: {args.domain}")
    print(f"Heat sources: {args.num_sources}")
    print(f"Architecture: {args.num_layers} layers, {args.hidden_size} hidden units")
    print(f"Training: {args.epochs} epochs, lr={args.lr}")
    
    # Create domain
    domain = DomainFactory.create_domain(args.domain)
    print(f"Domain bounds: {domain.bounds()}")
    
    # Create heat sources
    np.random.seed(args.seed)
    bounds = domain.bounds()
    heat_sources = []
    
    for i in range(args.num_sources):
        x_range = bounds[0]
        y_range = bounds[1] 
        z_range = bounds[2]
        
        # Random position within domain bounds (with margin)
        x_pos = np.random.uniform(x_range[0] + 0.1, x_range[1] - 0.1)
        y_pos = np.random.uniform(y_range[0] + 0.1, y_range[1] - 0.1)
        z_pos = np.random.uniform(z_range[0] + 0.1, z_range[1] - 0.1)
        
        heat_sources.append({
            'position': (x_pos, y_pos, z_pos),
            'amplitude': np.random.uniform(0.5, 2.0),
            'sigma': np.random.uniform(0.03, 0.08)
        })
    
    print("Heat sources:")
    for i, source in enumerate(heat_sources):
        pos = source['position']
        print(f"  {i+1}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"amp={source['amplitude']:.3f}, σ={source['sigma']:.3f}")
    
    # Train model
    model = train_delta_pinn_3d(args, domain, heat_sources)
    
    print("\n=== Training Complete ===")
    print("Use visualization.py to visualize results")
    print("Use interactive_demo.py for interactive exploration")

if __name__ == '__main__':
    main()