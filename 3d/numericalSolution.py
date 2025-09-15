#!/usr/bin/env python3
"""
3D Numerical Reference Solver for Heat Diffusion in Irregular Domains
Uses finite difference method with domain masking for irregular geometries
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter
import os
import pickle
from typing import List, Dict, Tuple, Optional
from domainShapes import Domain3D, DomainFactory

class HeatSolver3D:
    """
    3D finite difference heat equation solver with domain masking
    Supports irregular domains through signed distance functions
    """
    
    def __init__(self, domain: Domain3D, nX: int = 32, nY: int = 32, nZ: int = 32,
                 alpha: float = 0.01):
        """
        Initialize 3D heat solver
        
        Args:
            domain: Domain3D object defining geometry
            nX, nY, nZ: Grid resolution in each dimension
            alpha: Thermal diffusivity coefficient
        """
        self.domain = domain
        self.nX, self.nY, self.nZ = nX, nY, nZ
        self.alpha = alpha
        
        # Generate computational grid
        self.setupGrid()
        self.setupOperators()
        
        print(f"Initialized 3D heat solver:")
        print(f"  Domain: {domain.name}")
        print(f"  Grid: {nX}×{nY}×{nZ}")
        print(f"  Interior points: {self.nInterior}")
        print(f"  Thermal diffusivity: {alpha}")
    
    def setupGrid(self):
        """Setup computational grid and domain masks"""
        # Generate domain grid
        gridData = self.domain.generateGrid(self.nX, self.nY, self.nZ)
        
        self.x = gridData['x']
        self.y = gridData['y']
        self.z = gridData['z']
        self.X = gridData['X']
        self.Y = gridData['Y']
        self.Z = gridData['Z']
        self.mask = gridData['mask']  # True for interior points
        
        # Grid spacing
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        
        # Create index mapping for interior points
        self.interiorIndices = np.where(self.mask)
        self.nInterior = len(self.interiorIndices[0])
        
        # Create mapping from 3D indices to 1D interior point index
        self.indexMap = -np.ones((self.nX, self.nY, self.nZ), dtype=int)
        for idx, (i, j, k) in enumerate(zip(*self.interiorIndices)):
            self.indexMap[i, j, k] = idx
    
    def setupOperators(self):
        """Setup finite difference operators"""
        # Coefficients for second derivatives
        self.cx = self.alpha / (self.dx**2)
        self.cy = self.alpha / (self.dy**2)
        self.cz = self.alpha / (self.dz**2)
        
        # Build sparse Laplacian matrix for interior points
        self.buildLaplacianMatrix()
    
    def buildLaplacianMatrix(self):
        """Build sparse matrix for 3D Laplacian with domain masking"""
        
        # Lists for sparse matrix construction
        rowIndices = []
        colIndices = []
        data = []
        
        # Stencil offsets for 6-point finite difference
        offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        coeffs = [self.cx, self.cx, self.cy, self.cy, self.cz, self.cz]
        
        # Build matrix row by row
        for idx, (i, j, k) in enumerate(zip(*self.interiorIndices)):
            diagonalSum = 0
            
            # Check each neighbor in 6-point stencil
            for (di, dj, dk), coeff in zip(offsets, coeffs):
                ni, nj, nk = i + di, j + dj, k + dk
                
                # Check bounds
                if (0 <= ni < self.nX and 0 <= nj < self.nY and 0 <= nk < self.nZ):
                    if self.mask[ni, nj, nk]:  # Neighbor is interior
                        neighborIdx = self.indexMap[ni, nj, nk]
                        rowIndices.append(idx)
                        colIndices.append(neighborIdx)
                        data.append(coeff)
                        diagonalSum -= coeff
                    else:  # Neighbor is boundary (Dirichlet BC: u=0)
                        diagonalSum -= coeff
                else:  # Outside domain bounds
                    diagonalSum -= coeff
            
            # Add diagonal entry
            rowIndices.append(idx)
            colIndices.append(idx)
            data.append(diagonalSum)
        
        # Create sparse matrix
        self.laplacianMatrix = sp.csr_matrix(
            (data, (rowIndices, colIndices)),
            shape=(self.nInterior, self.nInterior)
        )
        
        print(f"Built Laplacian matrix: {self.nInterior}×{self.nInterior}, nnz={self.laplacianMatrix.nnz}")
    
    def createInitialCondition(self, heatSources: List[Dict]) -> np.ndarray:
        """
        Create initial temperature distribution from heat sources
        
        Args:
            heatSources: List of heat source dictionaries
        
        Returns:
            u0: Initial condition vector for interior points
        """
        # Initialize on full grid
        u0Grid = np.zeros((self.nX, self.nY, self.nZ))
        
        # Add each heat source
        for source in heatSources:
            x0, y0, z0 = source['position']
            amplitude = source['amplitude']
            sigma = source.get('sigma', 0.05)
            
            # Create Gaussian heat source
            gaussian = amplitude * np.exp(
                -((self.X - x0)**2 + (self.Y - y0)**2 + (self.Z - z0)**2) / (2 * sigma**2)
            )
            u0Grid += gaussian
        
        # Extract interior points
        u0Interior = u0Grid[self.interiorIndices]
        
        return u0Interior
    
    def solveExplicit(self, heatSources: List[Dict], tFinal: float = 1.0, 
                      nSteps: int = 1000, saveInterval: int = 10) -> Dict:
        """
        Solve heat equation using explicit time stepping (Forward Euler)
        
        Args:
            heatSources: List of heat source dictionaries
            tFinal: Final simulation time
            nSteps: Number of time steps
            saveInterval: Save solution every N steps
        
        Returns:
            Dictionary with solution history
        """
        dt = tFinal / nSteps
        
        # Stability check for explicit method
        stabilityLimit = 1 / (2 * self.alpha * (1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2))
        if dt > stabilityLimit:
            print(f"Warning: dt={dt:.6f} may exceed stability limit {stabilityLimit:.6f}")
            print("Consider using implicit solver or reducing dt")
        
        # Initial condition
        u = self.createInitialCondition(heatSources)
        
        # Time stepping matrix: I + dt*L
        timeStepMatrix = sp.identity(self.nInterior) + dt * self.laplacianMatrix
        
        # Storage
        tSave = []
        uSave = []
        saveCounter = 0
        
        print(f"Starting explicit time integration (dt={dt:.6f})...")
        
        for step in range(nSteps + 1):
            t = step * dt
            
            # Save solution
            if step % saveInterval == 0:
                tSave.append(t)
                uSave.append(u.copy())
                saveCounter += 1
                
                if step % (saveInterval * 10) == 0:
                    print(f"  Step {step}/{nSteps}, t={t:.4f}, max_u={np.max(u):.4f}")
            
            # Time step (except on last iteration)
            if step < nSteps:
                u = timeStepMatrix @ u
        
        print(f"Explicit solve complete. Saved {len(tSave)} time points.")
        
        return {
            'times': np.array(tSave),
            'solutions': uSave,
            'method': 'explicit',
            'dt': dt,
            'grid_shape': (self.nX, self.nY, self.nZ),
            'n_interior': self.nInterior
        }
    
    def solveImplicit(self, heatSources: List[Dict], tFinal: float = 1.0,
                      nSteps: int = 100, saveInterval: int = 1) -> Dict:
        """
        Solve heat equation using implicit time stepping (Backward Euler)
        
        Args:
            heatSources: List of heat source dictionaries  
            tFinal: Final simulation time
            nSteps: Number of time steps
            saveInterval: Save solution every N steps
        
        Returns:
            Dictionary with solution history
        """
        dt = tFinal / nSteps
        
        # Initial condition
        u = self.createInitialCondition(heatSources)
        
        # Implicit time stepping matrix: I - dt*L
        implicitMatrix = sp.identity(self.nInterior) - dt * self.laplacianMatrix
        
        # Pre-factorize for efficiency (if possible)
        try:
            from scipy.sparse.linalg import factorized
            solveFactorized = factorized(implicitMatrix.tocsc())
            useFactorized = True
        except:
            useFactorized = False
        
        # Storage
        tSave = []
        uSave = []
        
        print(f"Starting implicit time integration (dt={dt:.6f})...")
        
        for step in range(nSteps + 1):
            t = step * dt
            
            # Save solution
            if step % saveInterval == 0:
                tSave.append(t)
                uSave.append(u.copy())
                
                if step % (saveInterval * 10) == 0:
                    print(f"  Step {step}/{nSteps}, t={t:.4f}, max_u={np.max(u):.4f}")
            
            # Time step (except on last iteration)  
            if step < nSteps:
                if useFactorized:
                    u = solveFactorized(u)
                else:
                    u = spsolve(implicitMatrix, u)
        
        print(f"Implicit solve complete. Saved {len(tSave)} time points.")
        
        return {
            'times': np.array(tSave),
            'solutions': uSave,
            'method': 'implicit',
            'dt': dt,
            'grid_shape': (self.nX, self.nY, self.nZ),
            'n_interior': self.nInterior
        }
    
    def interiorToGrid(self, uInterior: np.ndarray) -> np.ndarray:
        """Convert interior point vector to full 3D grid"""
        uGrid = np.full((self.nX, self.nY, self.nZ), np.nan)
        uGrid[self.interiorIndices] = uInterior
        return uGrid
    
    def saveSolution(self, solutionData: Dict, filename: str):
        """Save solution data to file"""
        # Add grid information
        solutionData['grid_x'] = self.x
        solutionData['grid_y'] = self.y
        solutionData['grid_z'] = self.z
        solutionData['X'] = self.X
        solutionData['Y'] = self.Y
        solutionData['Z'] = self.Z
        solutionData['mask'] = self.mask
        solutionData['interior_indices'] = self.interiorIndices
        solutionData['domain_name'] = self.domain.name
        
        with open(filename, 'wb') as f:
            pickle.dump(solutionData, f)
        
        print(f"Solution saved to {filename}")
    
    def loadSolution(self, filename: str) -> Dict:
        """Load solution data from file"""
        with open(filename, 'rb') as f:
            solutionData = pickle.load(f)
        
        print(f"Solution loaded from {filename}")
        return solutionData

def solveReferenceProblem(domainType: str = 'sphere', heatSources: Optional[List[Dict]] = None,
                          nX: int = 48, alpha: float = 0.01, tFinal: float = 1.0,
                          method: str = 'implicit', saveDir: str = './numerical_solutions') -> Dict:
    """
    Solve reference heat diffusion problem for comparison with PINN
    
    Args:
        domainType: Type of domain ('sphere', 'cube', 'lshape', etc.)
        heatSources: List of heat source dictionaries
        nX: Grid resolution (nX×nX×nX)
        alpha: Thermal diffusivity
        tFinal: Final simulation time
        method: 'explicit' or 'implicit'
        saveDir: Directory to save results
    
    Returns:
        Solution dictionary
    """
    # Create domain
    domain = DomainFactory.createDomain(domainType)
    
    # Default heat sources if not provided
    if heatSources is None:
        np.random.seed(1337)
        bounds = domain.bounds()
        
        heatSources = []
        for i in range(2):
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
    
    print(f"\n=== Numerical Reference Solution ===")
    print(f"Domain: {domainType}")
    print(f"Grid: {nX}×{nX}×{nX}")
    print(f"Method: {method}")
    print(f"Heat sources: {len(heatSources)}")
    
    # Initialize solver
    solver = HeatSolver3D(domain, nX, nX, nX, alpha)
    
    # Solve
    if method == 'explicit':
        nSteps = int(tFinal / (0.1 / (2 * alpha * (nX**2))))  # Conservative time step
        solution = solver.solveExplicit(heatSources, tFinal, nSteps, saveInterval=max(1, nSteps//100))
    else:
        solution = solver.solveImplicit(heatSources, tFinal, nSteps=100, saveInterval=1)
    
    # Save results
    os.makedirs(saveDir, exist_ok=True)
    filename = os.path.join(saveDir, f"numerical_{domainType}_{method}.pkl")
    solver.saveSolution(solution, filename)
    
    return solution

def compareMethodsBenchmark():
    """Benchmark different solution methods"""
    print("\n=== Method Comparison Benchmark ===")
    
    domain = DomainFactory.createDomain('sphere')
    
    # Test heat sources
    heatSources = [{
        'position': (0.5, 0.5, 0.5),
        'amplitude': 1.0,
        'sigma': 0.05
    }]
    
    # Test different grid resolutions
    for nX in [16, 24, 32]:
        print(f"\n--- Grid Resolution {nX}×{nX}×{nX} ---")
        
        solver = HeatSolver3D(domain, nX, nX, nX)
        
        # Implicit method
        import time
        start = time.time()
        solImplicit = solver.solveImplicit(heatSources, tFinal=0.5, nSteps=50, saveInterval=10)
        timeImplicit = time.time() - start
        
        print(f"Implicit: {timeImplicit:.2f}s, final_max={np.max(solImplicit['solutions'][-1]):.4f}")
        
        # Explicit method (if stable)
        stabilityDt = 1 / (2 * 0.01 * (nX**2 + nX**2 + nX**2))
        nExplicit = int(0.5 / stabilityDt) + 1
        
        if nExplicit < 10000:  # Reasonable number of steps
            start = time.time()
            solExplicit = solver.solveExplicit(heatSources, tFinal=0.5, 
                                               nSteps=nExplicit, saveInterval=max(1, nExplicit//10))
            timeExplicit = time.time() - start
            
            print(f"Explicit: {timeExplicit:.2f}s, final_max={np.max(solExplicit['solutions'][-1]):.4f}")
        else:
            print(f"Explicit: Skipped (would need {nExplicit} steps)")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Numerical Heat Diffusion Solver")
    parser.add_argument('--domain', type=str, default='sphere', 
                       choices=['cube', 'sphere', 'lshape', 'torus', 'cylinder_holes'],
                       help='Domain shape')
    parser.add_argument('--nx', type=int, default=32, help='Grid resolution')
    parser.add_argument('--alpha', type=float, default=0.01, help='Thermal diffusivity')
    parser.add_argument('--t_final', type=float, default=1.0, help='Final simulation time')
    parser.add_argument('--method', type=str, default='implicit', choices=['explicit', 'implicit'],
                       help='Time integration method')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparison')
    parser.add_argument('--save_dir', type=str, default='./numerical_solutions', help='Save directory')
    
    args = parser.parse_args()
    
    if args.benchmark:
        compareMethodsBenchmark()
    else:
        solution = solveReferenceProblem(
            domainType=args.domain,
            nX=args.nx,
            alpha=args.alpha,
            tFinal=args.t_final,
            method=args.method,
            saveDir=args.save_dir
        )
        
        print(f"\nSolution computed with {len(solution['times'])} time points")
        print(f"Final maximum temperature: {np.max(solution['solutions'][-1]):.4f}")
