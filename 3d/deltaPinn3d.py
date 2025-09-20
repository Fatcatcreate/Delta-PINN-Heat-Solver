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
from domainShapes import Domain3D, DomainFactory, generateInteriorPoints, generateBoundaryPoints

def setSeed(seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)

class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for better high-frequency learning"""
    
    def __init__(self, inputDim: int, embedDim: int, scale: float = 1.0):
        super().__init__()
        self.embedDim = embedDim
        self.register_buffer('B', torch.randn(inputDim, embedDim // 2) * scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xProj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(xProj), torch.cos(xProj)], dim=-1)

class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
    def __init__(self, hiddenDim: int, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim),
            activation,
            nn.Linear(hiddenDim, hiddenDim)
        )
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))

class DeltaPINN3D(nn.Module):
    """Delta-PINN for 3D heat diffusion with advanced architecture"""
    
    def __init__(self, hiddenSize: int = 128, numLayers: int = 6, 
                 useFourier: bool = True, fourierScale: float = 1.0,
                 useResidual: bool = True):
        super().__init__()
        
        self.useFourier = useFourier
        self.useResidual = useResidual
        
        if useFourier:
            self.fourierEmbed = FourierFeatureEmbedding(4, hiddenSize, fourierScale)
            inputDim = hiddenSize
        else:
            inputDim = 4
            self.inputLayer = nn.Linear(4, hiddenSize)
        
        layers = []
        for i in range(numLayers):
            if useResidual and i > 0:
                layers.append(ResidualBlock(hiddenSize))
            else:
                layers.append(nn.Linear(hiddenSize if i > 0 or useFourier else inputDim, hiddenSize))
                layers.append(nn.Tanh())
        
        self.hiddenLayers = nn.ModuleList(layers)
        
        self.outputLayer = nn.Linear(hiddenSize, 1)
        
        self.logSigmaPde = nn.Parameter(torch.zeros(1))
        self.logSigmaBc = nn.Parameter(torch.zeros(1))
        self.logSigmaIc = nn.Parameter(torch.zeros(1))
        
        self.initialiseWeights()
    
    def initialiseWeights(self):
        """Xavier initialisation with special handling for residual blocks"""
        def initWeights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(initWeights)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xNorm = 2 * x - 1  
        yNorm = 2 * y - 1
        zNorm = 2 * z - 1
        tNorm = 2 * t - 1
        
        inputs = torch.cat([xNorm, yNorm, zNorm, tNorm], dim=1)
        
        if self.useFourier:
            h = self.fourierEmbed(inputs)
        else:
            h = torch.tanh(self.inputLayer(inputs))
        
        for layer in self.hiddenLayers:
            if isinstance(layer, ResidualBlock):
                h = layer(h)
            elif isinstance(layer, nn.Linear):
                h = layer(h)
            else:
                h = layer(h)
        
        return self.outputLayer(h)
    
    def getAdaptiveWeights(self) -> Tuple[float, float, float]:
        """Get current adaptive loss weights"""
        return (
            torch.exp(-self.logSigmaPde).item(),
            torch.exp(-self.logSigmaBc).item(), 
            torch.exp(-self.logSigmaIc).item()
        )

def createMultiSourceIc(heatSources: List[Dict], X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Create multi-source initial condition"""
    ic = np.zeros_like(X)
    
    for source in heatSources:
        x0, y0, z0 = source['position']
        amplitude = source['amplitude']
        sigma = source.get('sigma', 0.05)
        
        ic += amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2 + (Z - z0)**2) / (2 * sigma**2))
    
    return ic

def generateTrainingData3d(domain: Domain3D, nPde: int = 8000, nBc: int = 2000, 
                             nIc: int = 2000, device: str = 'cpu') -> Tuple:
    """Generate 3D training data with domain constraints"""
    
    xPde, yPde, zPde = generateInteriorPoints(domain, nPde, device)
    tPde = torch.rand(len(xPde), 1, device=device, requires_grad=True)
    
    xBc, yBc, zBc = generateBoundaryPoints(domain, nBc, device)
    tBc = torch.rand(len(xBc), 1, device=device, requires_grad=True)
    
    xIc, yIc, zIc = generateInteriorPoints(domain, nIc, device)
    tIc = torch.zeros(len(xIc), 1, device=device, requires_grad=True)
    
    return (xPde, yPde, zPde, tPde), (xBc, yBc, zBc, tBc), (xIc, yIc, zIc, tIc)

def computePdeLoss3d(model: DeltaPINN3D, x: torch.Tensor, y: torch.Tensor, 
                       z: torch.Tensor, t: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """3D heat equation residual: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)"""
    u = model(x, y, z, t)
    
    uT = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                             create_graph=True, retain_graph=True)[0]
    uX = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                             create_graph=True, retain_graph=True)[0]
    uY = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                             create_graph=True, retain_graph=True)[0]
    uZ = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), 
                             create_graph=True, retain_graph=True)[0]
    
    uXX = torch.autograd.grad(uX, x, grad_outputs=torch.ones_like(uX), 
                              create_graph=True, retain_graph=True)[0]
    uYY = torch.autograd.grad(uY, y, grad_outputs=torch.ones_like(uY), 
                              create_graph=True, retain_graph=True)[0]
    uZZ = torch.autograd.grad(uZ, z, grad_outputs=torch.ones_like(uZ), 
                              create_graph=True, retain_graph=True)[0]
    
    laplacian = uXX + uYY + uZZ
    pdeResidual = uT - alpha * laplacian
    
    return torch.mean(pdeResidual**2)

def computeBcLoss3d(model: DeltaPINN3D, x: torch.Tensor, y: torch.Tensor, 
                      z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Boundary condition loss (Dirichlet u=0)"""
    uBc = model(x, y, z, t)
    return torch.mean(uBc**2)

def computeIcLoss3d(model: DeltaPINN3D, x: torch.Tensor, y: torch.Tensor, 
                      z: torch.Tensor, t: torch.Tensor, heatSources: List[Dict]) -> torch.Tensor:
    """Initial condition loss with multi-source support"""
    uIc = model(x, y, z, t)
    
    xNp = x.detach().cpu().numpy().flatten()
    yNp = y.detach().cpu().numpy().flatten()
    zNp = z.detach().cpu().numpy().flatten()
    
    uTrueList = []
    for i in range(len(xNp)):
        uVal = 0.0
        for source in heatSources:
            x0, y0, z0 = source['position']
            amplitude = source['amplitude']
            sigma = source.get('sigma', 0.05)
            
            uVal += amplitude * np.exp(-((xNp[i] - x0)**2 + (yNp[i] - y0)**2 + (zNp[i] - z0)**2) / (2 * sigma**2))
        
        uTrueList.append(uVal)
    
    uTrueTensor = torch.tensor(uTrueList, dtype=torch.float32, device=x.device).unsqueeze(1)
    
    return torch.mean((uIc - uTrueTensor)**2)

class CurriculumScheduler:
    """Curriculum learning scheduler for progressive training"""
    
    def __init__(self, initialComplexity: float = 0.1, maxComplexity: float = 1.0, 
                 warmupEpochs: int = 1000):
        super().__init__()
        self.initialComplexity = initialComplexity
        self.maxComplexity = maxComplexity
        self.warmupEpochs = warmupEpochs
        self.currentEpoch = 0
    
    def step(self) -> float:
        """Get current complexity factor"""
        if self.currentEpoch < self.warmupEpochs:
            complexity = self.initialComplexity + (self.maxComplexity - self.initialComplexity) * \
                        (self.currentEpoch / self.warmupEpochs)
        else:
            complexity = self.maxComplexity
        
        self.currentEpoch += 1
        return complexity

def trainDeltaPinn3d(args, domain: Domain3D, heatSources: List[Dict], progressCallback=None) -> DeltaPINN3D:
    """Main training loop for 3D Delta-PINN"""
    
    setSeed(args.seed)
    
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    model = DeltaPINN3D(
        hiddenSize=args.hiddenSize,
        numLayers=args.numLayers,
        useFourier=args.useFourier,
        fourierScale=args.fourierScale,
        useResidual=args.useResidual
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-6)
    
    curriculum = CurriculumScheduler(warmupEpochs=args.warmupEpochs)
    
    pdeData, bcData, icData = generateTrainingData3d(
        domain, args.nPde, args.nBc, args.nIc, device
    )
    
    history = {
        'total_loss': [],
        'pde_loss': [],
        'bc_loss': [],
        'ic_loss': [],
        'adaptive_weights': []
    }
    
    bestLoss = float('inf')
    bestModelState = None
    
    print(f"Starting training for {args.epochs} epochs...")
    
    with open('output.txt', 'w') as f, tqdm(range(args.epochs), desc="Training", file=f) as pbar:
        for epoch in pbar:
            model.train()
            optimizer.zero_grad()
            
            complexity = curriculum.step()
            
            pdeLoss = computePdeLoss3d(model, *pdeData, args.alpha)
            bcLoss = computeBcLoss3d(model, *bcData)
            icLoss = computeIcLoss3d(model, *icData, heatSources)
            
            wPde, wBc, wIc = model.getAdaptiveWeights()
            
            totalLoss = complexity * (wPde * pdeLoss + wBc * bcLoss + wIc * icLoss)
            
            totalLoss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(totalLoss)
            
            history['total_loss'].append(totalLoss.item())
            history['pde_loss'].append(pdeLoss.item())
            history['bc_loss'].append(bcLoss.item())
            history['ic_loss'].append(icLoss.item())
            history['adaptive_weights'].append([wPde, wBc, wIc])
            
            if totalLoss.item() < bestLoss:
                bestLoss = totalLoss.item()
                bestModelState = model.state_dict().copy()
            
            pbar.set_postfix({
                'Total': f'{totalLoss.item():.2e}',
                'PDE': f'{pdeLoss.item():.2e}',
                'BC': f'{bcLoss.item():.2e}',
                'IC': f'{icLoss.item():.2e}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            if progressCallback:
                progressCallback(epoch, args.epochs, totalLoss.item())
            
            if epoch % args.logInterval == 0:
                print(f"\nEpoch {epoch}/{args.epochs}")
                print(f"Total Loss: {totalLoss.item():.6e}")
                print(f"PDE Loss: {pdeLoss.item():.6e}")
                print(f"BC Loss: {bcLoss.item():.6e}")
                print(f"IC Loss: {icLoss.item():.6e}")
                print(f"Adaptive Weights: PDE={wPde:.3f}, BC={wBc:.3f}, IC={wIc:.3f}")
                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    if bestModelState is not None:
        model.load_state_dict(bestModelState)
        print(f"Loaded best model with loss: {bestLoss:.6e}")
    
    os.makedirs(args.saveDir, exist_ok=True)
    
    domain_filename = os.path.basename(domain.name)
    domainNameSanitised = os.path.splitext(domain_filename)[0]

    modelPath = os.path.join(args.saveDir, f"deltaPinn3d_{domainNameSanitised.lower()}.pt")
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'history': history,
            'args': vars(args),
            'domain_name': domain.name,
            'heat_sources': heatSources,
        },
        modelPath,
    )
    
    historyPath = os.path.join(args.saveDir, f"training_history_{domainNameSanitised.lower()}.csv")
    with open(historyPath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Total_Loss', 'PDE_Loss', 'BC_Loss', 'IC_Loss', 
                        'Weight_PDE', 'Weight_BC', 'Weight_IC'])
        
        for i, (total, pde, bc, ic, weights) in enumerate(zip(
            history['total_loss'], history['pde_loss'], 
            history['bc_loss'], history['ic_loss'], history['adaptive_weights']
        )):
            writer.writerow([i, total, pde, bc, ic, weights[0], weights[1], weights[2]])
    
    print(f"Model saved to: {modelPath}")
    print(f"History saved to: {historyPath}")
    
    return model

def loadTrainedModel(modelPath: str, device: str = 'cpu') -> Tuple[DeltaPINN3D, Dict]:
    """Load trained model and metadata"""
    checkpoint = torch.load(modelPath, map_location=device)
    
    argsDict = checkpoint['args']
    model = DeltaPINN3D(
        hiddenSize=argsDict['hidden_size'],
        numLayers=argsDict['num_layers'],
        useFourier=argsDict['use_fourier'],
        fourierScale=argsDict['fourier_scale'],
        useResidual=argsDict['useResidual']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def predictSolution3d(model: DeltaPINN3D, domain: Domain3D, 
                       tEval: float, nX: int = 32, nY: int = 32, nZ: int = 32,
                       device: str = 'cpu') -> Dict[str, np.ndarray]:
    """Predict solution on regular grid at given time"""
    
    gridData = domain.generateGrid(nX, nY, nZ)
    X, Y, Z, mask = gridData['X'], gridData['Y'], gridData['Z'], gridData['mask']
    
    xFlat = X.flatten()
    yFlat = Y.flatten()
    zFlat = Z.flatten()
    tFlat = np.full_like(xFlat, tEval)
    
    xTensor = torch.tensor(xFlat, dtype=torch.float32, device=device).unsqueeze(1)
    yTensor = torch.tensor(yFlat, dtype=torch.float32, device=device).unsqueeze(1)
    zTensor = torch.tensor(zFlat, dtype=torch.float32, device=device).unsqueeze(1)
    tTensor = torch.tensor(tFlat, dtype=torch.float32, device=device).unsqueeze(1)
    
    model.eval()
    with torch.no_grad():
        uPred = model(xTensor, yTensor, zTensor, tTensor)
    
    uPredGrid = uPred.cpu().numpy().reshape(X.shape)
    
    uPredGrid[~mask] = np.nan
    
    return {
        'x': gridData['x'],
        'y': gridData['y'], 
        'z': gridData['z'],
        'X': X,
        'Y': Y,
        'Z': Z,
        'u': uPredGrid,
        'mask': mask,
        't': tEval
    }

def main():
    parser = argparse.ArgumentParser(description="Train 3D Delta-PINN for heat diffusion")
    
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of hidden layers')
    parser.add_argument('--use_fourier', action='store_true', default=True, help='Use Fourier features')
    parser.add_argument('--fourier_scale', type=float, default=1.0, help='Fourier feature scale')
    parser.add_argument('--useResidual', action='store_true', default=True, help='Use residual connections')
    
    parser.add_argument('--epochs', type=int, default=5000, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=1000, help='Curriculum warmup epochs')
    parser.add_argument('--alpha', type=float, default=0.01, help='Thermal diffusivity')
    
    parser.add_argument('--n_pde', type=int, default=8000, help='Number of PDE collocation points')
    parser.add_argument('--n_bc', type=int, default=2000, help='Number of boundary points')
    parser.add_argument('--n_ic', type=int, default=2000, help='Number of initial condition points')
    
    parser.add_argument('--domain', type=str, default='sphere', 
                       choices=['cube', 'sphere', 'lshape', 'torus', 'cylinder_holes'],
                       help='Domain shape')
    parser.add_argument('--numSources', type=int, default=2, help='Number of heat sources')
    
    parser.add_argument('--device', type=str, default='auto', help='Device: cpu, cuda, mps, or auto')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--saveDir', type=str, default='./models', help='Save directory')
    parser.add_argument('--logInterval', type=int, default=500, help='Logging interval')
    
    args = parser.parse_args()
    
    print("=== 3D Delta-PINN Heat Diffusion Training ===")
    print(f"Domain: {args.domain}")
    print(f"Heat sources: {args.numSources}")
    print(f"Architecture: {args.num_layers} layers, {args.hidden_size} hidden units")
    print(f"Training: {args.epochs} epochs, lr={args.lr}")
    
    domain = DomainFactory.create_domain(args.domain)
    print(f"Domain bounds: {domain.bounds()}")
    
    np.random.seed(args.seed)
    bounds = domain.bounds()
    heatSources = []
    
    for i in range(args.numSources):
        xRange = bounds[0]
        yRange = bounds[1] 
        zRange = bounds[2]
        
        xPos = np.random.uniform(xRange[0] + 0.1, xRange[1] - 0.1)
        yPos = np.random.uniform(yRange[0] + 0.1, yRange[1] - 0.1)
        zPos = np.random.uniform(zRange[0] + 0.1, zRange[1] - 0.1)
        
        heatSources.append({
            'position': (xPos, yPos, zPos),
            'amplitude': np.random.uniform(0.5, 2.0),
            'sigma': np.random.uniform(0.03, 0.08)
        })
    
    print("Heat sources:")
    for i, source in enumerate(heatSources):
        pos = source['position']
        print(f"  {i+1}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"amp={source['amplitude']:.3f}, \u03c3={source['sigma']:.3f}")
    
    model = trainDeltaPinn3d(args, domain, heatSources)
    
    print("\n=== Training Complete ===")
    print("Use visualisation.py to visualise results")
    print("Use interactiveDemo.py for interactive exploration")

if __name__ == '__main__':
    main()

