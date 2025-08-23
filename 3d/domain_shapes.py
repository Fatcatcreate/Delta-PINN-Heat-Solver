#!/usr/bin/env python3
"""
3D Domain Shapes using Signed Distance Functions (SDFs)
Supports irregular geometries: spheres, L-shapes, arbitrary polyhedra
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

class Domain3D(ABC):
    """Abstract base class for 3D domains"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Signed distance function: negative inside, positive outside"""
        pass
    
    @abstractmethod
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Return ((x_min, x_max), (y_min, y_max), (z_min, z_max))"""
        pass
    
    def is_inside(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Check if points are inside domain"""
        return self.sdf(x, y, z) <= tol
    
    def is_boundary(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        """Check if points are on boundary"""
        sdf_vals = self.sdf(x, y, z)
        return np.abs(sdf_vals) <= tol
    
    def generate_grid(self, nx: int, ny: int, nz: int) -> Dict[str, np.ndarray]:
        """Generate uniform grid with domain mask"""
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.bounds()
        
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        mask = self.is_inside(X, Y, Z)
        
        return {
            'x': x, 'y': y, 'z': z,
            'X': X, 'Y': Y, 'Z': Z, 
            'mask': mask,
            'sdf': self.sdf(X, Y, Z)
        }

class UnitCube(Domain3D):
    """Unit cube domain [0,1]³"""
    
    def __init__(self):
        super().__init__("UnitCube")
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Distance to cube boundary
        dx = np.maximum(0 - x, x - 1)
        dy = np.maximum(0 - y, y - 1) 
        dz = np.maximum(0 - z, z - 1)
        
        # Outside distance
        outside_dist = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2)
        
        # Inside distance (negative)
        inside_dist = np.maximum(np.maximum(dx, dy), dz)
        
        return np.where(
            (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1),
            inside_dist, outside_dist
        )
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return ((0, 1), (0, 1), (0, 1))

class Sphere(Domain3D):
    """Spherical domain"""
    
    def __init__(self, center: Tuple[float, float, float] = (0.5, 0.5, 0.5), 
                 radius: float = 0.4):
        super().__init__("Sphere")
        self.center = np.array(center)
        self.radius = radius
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        points = np.stack([x, y, z], axis=-1)
        distances = np.linalg.norm(points - self.center, axis=-1)
        return distances - self.radius
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        margin = 0.1
        return (
            (self.center[0] - self.radius - margin, self.center[0] + self.radius + margin),
            (self.center[1] - self.radius - margin, self.center[1] + self.radius + margin),
            (self.center[2] - self.radius - margin, self.center[2] + self.radius + margin)
        )

class LShapedPrism(Domain3D):
    """L-shaped prism domain"""
    
    def __init__(self, thickness: float = 0.3):
        super().__init__("LShapedPrism")
        self.thickness = thickness
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # L-shape is union of two rectangles in xy-plane, extruded in z
        
        # Rectangle 1: [0, 1] x [0, thickness] x [0, 1]
        rect1_x = np.maximum(0 - x, x - 1)
        rect1_y = np.maximum(0 - y, y - self.thickness)
        rect1_z = np.maximum(0 - z, z - 1)
        
        # Rectangle 2: [0, thickness] x [0, 1] x [0, 1]  
        rect2_x = np.maximum(0 - x, x - self.thickness)
        rect2_y = np.maximum(0 - y, y - 1)
        rect2_z = np.maximum(0 - z, z - 1)
        
        # SDF for each rectangle
        sdf1 = self._box_sdf(rect1_x, rect1_y, rect1_z)
        sdf2 = self._box_sdf(rect2_x, rect2_y, rect2_z)
        
        # Union (minimum distance)
        return np.minimum(sdf1, sdf2)
    
    def _box_sdf(self, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
        """SDF for box given edge distances"""
        outside_dist = np.sqrt(
            np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2
        )
        inside_dist = np.maximum(np.maximum(dx, dy), dz)
        
        return np.where(
            (dx <= 0) & (dy <= 0) & (dz <= 0),
            inside_dist, outside_dist
        )
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return ((0, 1), (0, 1), (0, 1))

class TorusSection(Domain3D):
    """Torus section for advanced geometry"""
    
    def __init__(self, R: float = 0.3, r: float = 0.15, 
                 center: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        super().__init__("TorusSection")
        self.R = R  # Major radius
        self.r = r  # Minor radius
        self.center = np.array(center)
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Translate to torus center
        px = x - self.center[0]
        py = y - self.center[1] 
        pz = z - self.center[2]
        
        # Torus SDF
        q = np.sqrt(px**2 + py**2) - self.R
        return np.sqrt(q**2 + pz**2) - self.r
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        total_radius = self.R + self.r + 0.1
        return (
            (self.center[0] - total_radius, self.center[0] + total_radius),
            (self.center[1] - total_radius, self.center[1] + total_radius), 
            (self.center[2] - self.r - 0.1, self.center[2] + self.r + 0.1)
        )

class CylinderWithHoles(Domain3D):
    """Cylinder with cylindrical holes"""
    
    def __init__(self, radius: float = 0.4, height: float = 0.8,
                 center: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 hole_positions: Optional[list] = None, hole_radius: float = 0.1):
        super().__init__("CylinderWithHoles")
        self.radius = radius
        self.height = height
        self.center = np.array(center)
        self.hole_radius = hole_radius
        
        if hole_positions is None:
            self.hole_positions = [(0.3, 0.5), (0.7, 0.5)]  # (x, y) positions
        else:
            self.hole_positions = hole_positions
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Translate to center
        px = x - self.center[0]
        py = y - self.center[1]
        pz = z - self.center[2]
        
        # Main cylinder SDF
        radial_dist = np.sqrt(px**2 + py**2) - self.radius
        height_dist = np.abs(pz) - self.height/2
        
        # Cylinder SDF (union of radial and height constraints)
        cylinder_sdf = np.maximum(radial_dist, height_dist)
        
        # Subtract holes
        for hole_x, hole_y in self.hole_positions:
            hole_px = x - hole_x
            hole_py = y - hole_y
            hole_dist = np.sqrt(hole_px**2 + hole_py**2) - self.hole_radius
            
            # Subtract hole (max with negative hole distance)
            cylinder_sdf = np.maximum(cylinder_sdf, -hole_dist)
        
        return cylinder_sdf
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        margin = 0.1
        return (
            (self.center[0] - self.radius - margin, self.center[0] + self.radius + margin),
            (self.center[1] - self.radius - margin, self.center[1] + self.radius + margin),
            (self.center[2] - self.height/2 - margin, self.center[2] + self.height/2 + margin)
        )

class DomainFactory:
    """Factory for creating domain instances"""
    
    AVAILABLE_DOMAINS = {
        'cube': UnitCube,
        'sphere': Sphere, 
        'lshape': LShapedPrism,
        'torus': TorusSection,
        'cylinder_holes': CylinderWithHoles
    }
    
    @classmethod
    def create_domain(cls, domain_type: str, **kwargs) -> Domain3D:
        """Create domain instance by type"""
        if domain_type not in cls.AVAILABLE_DOMAINS:
            raise ValueError(f"Unknown domain type: {domain_type}")
        
        domain_class = cls.AVAILABLE_DOMAINS[domain_type]
        return domain_class(**kwargs)
    
    @classmethod
    def list_domains(cls) -> list:
        """List available domain types"""
        return list(cls.AVAILABLE_DOMAINS.keys())

def generate_interior_points(domain: Domain3D, n_points: int, 
                           device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random points inside domain for PDE collocation"""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain.bounds()
    
    # Generate more points than needed, then filter
    n_gen = min(n_points * 5, 100000)  # Avoid memory issues
    
    x_rand = np.random.uniform(x_min, x_max, n_gen)
    y_rand = np.random.uniform(y_min, y_max, n_gen)
    z_rand = np.random.uniform(z_min, z_max, n_gen)
    
    # Filter points inside domain
    inside_mask = domain.is_inside(x_rand, y_rand, z_rand)
    
    x_inside = x_rand[inside_mask]
    y_inside = y_rand[inside_mask]
    z_inside = z_rand[inside_mask]
    
    # Take first n_points (or all if less than requested)
    n_actual = min(len(x_inside), n_points)
    
    if n_actual < n_points:
        print(f"Warning: Only generated {n_actual}/{n_points} interior points")
    
    x_points = torch.tensor(x_inside[:n_actual], dtype=torch.float32, device=device, requires_grad=True)
    y_points = torch.tensor(y_inside[:n_actual], dtype=torch.float32, device=device, requires_grad=True)
    z_points = torch.tensor(z_inside[:n_actual], dtype=torch.float32, device=device, requires_grad=True)
    
    return x_points.unsqueeze(1), y_points.unsqueeze(1), z_points.unsqueeze(1)

def generate_boundary_points(domain: Domain3D, n_points: int,
                           device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate points on domain boundary"""
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = domain.bounds()
    
    # Generate many candidates
    n_gen = min(n_points * 10, 200000)
    
    x_rand = np.random.uniform(x_min, x_max, n_gen)
    y_rand = np.random.uniform(y_min, y_max, n_gen)
    z_rand = np.random.uniform(z_min, z_max, n_gen)
    
    # Filter boundary points
    boundary_mask = domain.is_boundary(x_rand, y_rand, z_rand, tol=1e-3)
    
    x_boundary = x_rand[boundary_mask]
    y_boundary = y_rand[boundary_mask]
    z_boundary = z_rand[boundary_mask]
    
    n_actual = min(len(x_boundary), n_points)
    
    if n_actual < n_points:
        print(f"Warning: Only generated {n_actual}/{n_points} boundary points")
    
    x_points = torch.tensor(x_boundary[:n_actual], dtype=torch.float32, device=device, requires_grad=True)
    y_points = torch.tensor(y_boundary[:n_actual], dtype=torch.float32, device=device, requires_grad=True)
    z_points = torch.tensor(z_boundary[:n_actual], dtype=torch.float32, device=device, requires_grad=True)
    
    return x_points.unsqueeze(1), y_points.unsqueeze(1), z_points.unsqueeze(1)

def test_domains():
    """Test domain implementations"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    domains = [
        UnitCube(),
        Sphere(),
        LShapedPrism(),
        CylinderWithHoles()
    ]
    
    fig = plt.figure(figsize=(16, 4))
    
    for i, domain in enumerate(domains):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        
        # Generate test grid  
        grid = domain.generate_grid(32, 32, 32)
        X, Y, Z, mask = grid['X'], grid['Y'], grid['Z'], grid['mask']
        
        # Plot interior points
        interior_idx = np.where(mask)
        if len(interior_idx[0]) > 0:
            sample_idx = np.random.choice(len(interior_idx[0]), 
                                        min(1000, len(interior_idx[0])), replace=False)
            ax.scatter(X[interior_idx][sample_idx], 
                      Y[interior_idx][sample_idx],
                      Z[interior_idx][sample_idx], 
                      s=1, alpha=0.3)
        
        ax.set_title(domain.name)
        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_domains()