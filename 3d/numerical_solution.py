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
from domain_shapes import Domain3D, DomainFactory

class HeatSolver3D:
    """
    3D finite difference heat equation solver with domain masking
    Supports irregular domains through signed distance functions
    """
    
    def __init__(self, domain: Domain3D, nx: int = 32, ny: int = 32, nz: int = 32,
                 alpha: float = 0.01):
        """
        Initialize 3D heat solver
        
        Args:
            domain: Domain3D object defining geometry
            nx, ny, nz: Grid resolution in each dimension
            alpha: Thermal diffusivity coefficient
        """
        self.domain = domain
        self.nx, self.ny, self.nz = nx, ny, nz
        self.alpha = alpha
        
        # Generate computational grid
        self._setup_grid()
        self._setup_operators()
        
        print(f"Initialized 3D heat solver:")
        print(f"  Domain: {domain.name}")
        print(f"  Grid: {nx}×{ny}×{nz}")
        print(f"  Interior points: {self.n_interior}")
        print(f"  Thermal diffusivity: {alpha}")
    
    def _setup_grid(self):
        """Setup computational grid and domain masks"""
        # Generate domain grid
        grid_data = self.domain.generate_grid(self.nx, self.ny, self.nz)
        
        self.x = grid_data['x']
        self.y = grid_data['y']
        self.z = grid_data['z']
        self.X = grid_data['X']
        self.Y = grid_data['Y']
        self.Z = grid_data['Z']
        self.mask = grid_data['mask']  # True for interior points
        
        # Grid spacing
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        
        # Create index mapping for interior points
        self.interior_indices = np.where(self.mask)
        self.n_interior = len(self.interior_indices[0])
        
        # Create mapping from 3D indices to 1D interior point index
        self.index_map = -np.ones((self.nx, self.ny, self.nz), dtype=int)
        for idx, (i, j, k) in enumerate(zip(*self.interior_indices)):
            self.index_map[i, j, k] = idx
    
    def _setup_operators(self):
        """Setup finite difference operators"""
        # Coefficients for second derivatives
        self.cx = self.alpha / (self.dx**2)
        self.cy = self.alpha / (self.dy**2)
        self.cz = self.alpha / (self.dz**2)
        
        # Build sparse Laplacian matrix for interior points
        self._build_laplacian_matrix()
    
    def _build_laplacian_matrix(self):
        """Build sparse matrix for 3D Laplacian with domain masking"""
        
        # Lists for sparse matrix construction
        row_indices = []
        col_indices = []
        data = []
        
        # Stencil offsets for 6-point finite difference
        offsets = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]
        coeffs = [self.cx, self.cx, self.cy, self.cy, self.cz, self.cz]
        
        # Build matrix row by row
        for idx, (i, j, k) in enumerate(zip(*self.interior_indices)):
            diagonal_sum = 0
            
            # Check each neighbor in 6-point stencil
            for (di, dj, dk), coeff in zip(offsets, coeffs):
                ni, nj, nk = i + di, j + dj, k + dk
                
                # Check bounds
                if (0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz):
                    if self.mask[ni, nj, nk]:  # Neighbor is interior
                        neighbor_idx = self.index_map[ni, nj, nk]
                        row_indices.append(idx)
                        col_indices.append(neighbor_idx)
                        data.append(coeff)
                        diagonal_sum -= coeff
                    else:  # Neighbor is boundary (Dirichlet BC: u=0)
                        diagonal_sum -= coeff
                else:  # Outside domain bounds
                    diagonal_sum -= coeff
            
            # Add diagonal entry
            row_indices.append(idx)
            col_indices.append(idx)
            data.append(diagonal_sum)
        
        # Create sparse matrix
        self.laplacian_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_interior, self.n_interior)
        )
        
        print(f"Built Laplacian matrix: {self.n_interior}×{self.n_interior}, nnz={self.laplacian_matrix.nnz}")
    
    def create_initial_condition(self, heat_sources: List[Dict]) -> np.ndarray:
        """
        Create initial temperature distribution from heat sources
        
        Args:
            heat_sources: List of heat source dictionaries
        
        Returns:
            u0: Initial condition vector for interior points
        """
        # Initialize on full grid
        u0_grid = np.zeros((self.nx, self.ny, self.nz))
        
        # Add each heat source
        for source in heat_sources:
            x0, y0, z0 = source['position']
            amplitude = source['amplitude']
            sigma = source.get('sigma', 0.05)
            
            # Create Gaussian heat source
            gaussian = amplitude * np.exp(
                -((self.X - x0)**2 + (self.Y - y0)**2 + (self.Z - z0)**2) / (2 * sigma**2)
            )
            u0_grid += gaussian
        
        # Extract interior points
        u0_interior = u0_grid[self.interior_indices]
        
        return u0_interior
    
    def solve_explicit(self, heat_sources: List[Dict], t_final: float = 1.0, 
                      n_steps: int = 1000, save_interval: int = 10) -> Dict:
        """
        Solve heat equation using explicit time stepping (Forward Euler)
        
        Args:
            heat_sources: List of heat source dictionaries
            t_final: Final simulation time
            n_steps: Number of time steps
            save_interval: Save solution every N steps
        
        Returns:
            Dictionary with solution history
        """
        dt = t_final / n_steps
        
        # Stability check for explicit method
        stability_limit = 1 / (2 * self.alpha * (1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2))
        if dt > stability_limit:
            print(f"Warning: dt={dt:.6f} may exceed stability limit {stability_limit:.6f}")
            print("Consider using implicit solver or reducing dt")
        
        # Initial condition
        u = self.create_initial_condition(heat_sources)
        
        # Time stepping matrix: I + dt*L
        time_step_matrix = sp.identity(self.n_interior) + dt * self.laplacian_matrix
        
        # Storage
        t_save = []
        u_save = []
        save_counter = 0
        
        print(f"Starting explicit time integration (dt={dt:.6f})...")
        
        for step in range(n_steps + 1):
            t = step * dt
            
            # Save solution
            if step % save_interval == 0:
                t_save.append(t)
                u_save.append(u.copy())
                save_counter += 1
                
                if step % (save_interval * 10) == 0:
                    print(f"  Step {step}/{n_steps}, t={t:.4f}, max_u={np.max(u):.4f}")
            
            # Time step (except on last iteration)
            if step < n_steps:
                u = time_step_matrix @ u
        
        print(f"Explicit solve complete. Saved {len(t_save)} time points.")
        
        return {
            'times': np.array(t_save),
            'solutions': u_save,
            'method': 'explicit',
            'dt': dt,
            'grid_shape': (self.nx, self.ny, self.nz),
            'n_interior': self.n_interior
        }
    
    def solve_implicit(self, heat_sources: List[Dict], t_final: float = 1.0,
                      n_steps: int = 100, save_interval: int = 1) -> Dict:
        """
        Solve heat equation using implicit time stepping (Backward Euler)
        
        Args:
            heat_sources: List of heat source dictionaries  
            t_final: Final simulation time
            n_steps: Number of time steps
            save_interval: Save solution every N steps
        
        Returns:
            Dictionary with solution history
        """
        dt = t_final / n_steps
        
        # Initial condition
        u = self.create_initial_condition(heat_sources)
        
        # Implicit time stepping matrix: I - dt*L
        implicit_matrix = sp.identity(self.n_interior) - dt * self.laplacian_matrix
        
        # Pre-factorize for efficiency (if possible)
        try:
            from scipy.sparse.linalg import factorized
            solve_factorized = factorized(implicit_matrix.tocsc())
            use_factorized = True
        except:
            use_factorized = False
        
        # Storage
        t_save = []
        u_save = []
        
        print(f"Starting implicit time integration (dt={dt:.6f})...")
        
        for step in range(n_steps + 1):
            t = step * dt
            
            # Save solution
            if step % save_interval == 0:
                t_save.append(t)
                u_save.append(u.copy())
                
                if step % (save_interval * 10) == 0:
                    print(f"  Step {step}/{n_steps}, t={t:.4f}, max_u={np.max(u):.4f}")
            
            # Time step (except on last iteration)  
            if step < n_steps:
                if use_factorized:
                    u = solve_factorized(u)
                else:
                    u = spsolve(implicit_matrix, u)
        
        print(f"Implicit solve complete. Saved {len(t_save)} time points.")
        
        return {
            'times': np.array(t_save),
            'solutions': u_save,
            'method': 'implicit',
            'dt': dt,
            'grid_shape': (self.nx, self.ny, self.nz),
            'n_interior': self.n_interior
        }
    
    def interior_to_grid(self, u_interior: np.ndarray) -> np.ndarray:
        """Convert interior point vector to full 3D grid"""
        u_grid = np.full((self.nx, self.ny, self.nz), np.nan)
        u_grid[self.interior_indices] = u_interior
        return u_grid
    
    def save_solution(self, solution_data: Dict, filename: str):
        """Save solution data to file"""
        # Add grid information
        solution_data['grid_x'] = self.x
        solution_data['grid_y'] = self.y
        solution_data['grid_z'] = self.z
        solution_data['mask'] = self.mask
        solution_data['interior_indices'] = self.interior_indices
        solution_data['domain_name'] = self.domain.name
        
        with open(filename, 'wb') as f:
            pickle.dump(solution_data, f)
        
        print(f"Solution saved to {filename}")
    
    def load_solution(self, filename: str) -> Dict:
        """Load solution data from file"""
        with open(filename, 'rb') as f:
            solution_data = pickle.load(f)
        
        print(f"Solution loaded from {filename}")
        return solution_data

def solve_reference_problem(domain_type: str = 'sphere', heat_sources: Optional[List[Dict]] = None,
                          nx: int = 48, alpha: float = 0.01, t_final: float = 1.0,
                          method: str = 'implicit', save_dir: str = './numerical_solutions') -> Dict:
    """
    Solve reference heat diffusion problem for comparison with PINN
    
    Args:
        domain_type: Type of domain ('sphere', 'cube', 'lshape', etc.)
        heat_sources: List of heat source dictionaries
        nx: Grid resolution (nx×nx×nx)
        alpha: Thermal diffusivity
        t_final: Final simulation time
        method: 'explicit' or 'implicit'
        save_dir: Directory to save results
    
    Returns:
        Solution dictionary
    """
    # Create domain
    domain = DomainFactory.create_domain(domain_type)
    
    # Default heat sources if not provided
    if heat_sources is None:
        np.random.seed(1337)
        bounds = domain.bounds()
        
        heat_sources = []
        for i in range(2):
            x_range = bounds[0]
            y_range = bounds[1]
            z_range = bounds[2]
            
            x_pos = np.random.uniform(x_range[0] + 0.1, x_range[1] - 0.1)
            y_pos = np.random.uniform(y_range[0] + 0.1, y_range[1] - 0.1)
            z_pos = np.random.uniform(z_range[0] + 0.1, z_range[1] - 0.1)
            
            heat_sources.append({
                'position': (x_pos, y_pos, z_pos),
                'amplitude': np.random.uniform(0.5, 2.0),
                'sigma': np.random.uniform(0.03, 0.08)
            })
    
    print(f"\n=== Numerical Reference Solution ===")
    print(f"Domain: {domain_type}")
    print(f"Grid: {nx}×{nx}×{nx}")
    print(f"Method: {method}")
    print(f"Heat sources: {len(heat_sources)}")
    
    # Initialize solver
    solver = HeatSolver3D(domain, nx, nx, nx, alpha)
    
    # Solve
    if method == 'explicit':
        n_steps = int(t_final / (0.1 / (2 * alpha * (nx**2))))  # Conservative time step
        solution = solver.solve_explicit(heat_sources, t_final, n_steps, save_interval=max(1, n_steps//100))
    else:
        solution = solver.solve_implicit(heat_sources, t_final, n_steps=100, save_interval=1)
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"numerical_{domain_type}_{method}.pkl")
    solver.save_solution(solution, filename)
    
    return solution

def compare_methods_benchmark():
    """Benchmark different solution methods"""
    print("\n=== Method Comparison Benchmark ===")
    
    domain = DomainFactory.create_domain('sphere')
    
    # Test heat sources
    heat_sources = [{
        'position': (0.5, 0.5, 0.5),
        'amplitude': 1.0,
        'sigma': 0.05
    }]
    
    # Test different grid resolutions
    for nx in [16, 24, 32]:
        print(f"\n--- Grid Resolution {nx}×{nx}×{nx} ---")
        
        solver = HeatSolver3D(domain, nx, nx, nx)
        
        # Implicit method
        import time
        start = time.time()
        sol_implicit = solver.solve_implicit(heat_sources, t_final=0.5, n_steps=50, save_interval=10)
        time_implicit = time.time() - start
        
        print(f"Implicit: {time_implicit:.2f}s, final_max={np.max(sol_implicit['solutions'][-1]):.4f}")
        
        # Explicit method (if stable)
        stability_dt = 1 / (2 * 0.01 * (nx**2 + nx**2 + nx**2))
        n_explicit = int(0.5 / stability_dt) + 1
        
        if n_explicit < 10000:  # Reasonable number of steps
            start = time.time()
            sol_explicit = solver.solve_explicit(heat_sources, t_final=0.5, 
                                               n_steps=n_explicit, save_interval=max(1, n_explicit//10))
            time_explicit = time.time() - start
            
            print(f"Explicit: {time_explicit:.2f}s, final_max={np.max(sol_explicit['solutions'][-1]):.4f}")
        else:
            print(f"Explicit: Skipped (would need {n_explicit} steps)")

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
        compare_methods_benchmark()
    else:
        solution = solve_reference_problem(
            domain_type=args.domain,
            nx=args.nx,
            alpha=args.alpha,
            t_final=args.t_final,
            method=args.method,
            save_dir=args.save_dir
        )
        
        print(f"\nSolution computed with {len(solution['times'])} time points")
        print(f"Final maximum temperature: {np.max(solution['solutions'][-1]):.4f}")