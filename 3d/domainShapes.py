#!/usr/bin/env python3
"""
3D Domain Shapes using Signed Distance Functions (SDFs)
Supports irregular geometries: spheres, L-shapes, arbitrary polyhedra
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from scipy.interpolate import interpn

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
        """Return ((xMin, xMax), (yMin, yMax), (zMin, zMax))"""
        pass
    
    def isInside(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """Check if points are inside domain"""
        return self.sdf(x, y, z) <= tol
    
    def isBoundary(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        """Check if points are on boundary"""
        sdfVals = self.sdf(x, y, z)
        return np.abs(sdfVals) <= tol
    
    def generateGrid(self, nX: int, nY: int, nZ: int) -> Dict[str, np.ndarray]:
        """Generate uniform grid with domain mask"""
        (xMin, xMax), (yMin, yMax), (zMin, zMax) = self.bounds()
        
        x = np.linspace(xMin, xMax, nX)
        y = np.linspace(yMin, yMax, nY)
        z = np.linspace(zMin, zMax, nZ)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        mask = self.isInside(X, Y, Z)
        
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
        outsideDist = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2)
        
        # Inside distance (negative)
        insideDist = np.maximum(np.maximum(dx, dy), dz)
        
        return np.where(
            (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1) & (z >= 0) & (z <= 1),
            insideDist, outsideDist
        )
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return ((0, 1), (0, 1), (0, 1))

class Sphere(Domain3D):
    """Spherical domain"""
    
    def __init__(self, centre: Tuple[float, float, float] = (0.5, 0.5, 0.5), 
                 radius: float = 0.4):
        super().__init__("Sphere")
        self.centre = np.array(centre)
        self.radius = radius
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        points = np.stack([x, y, z], axis=-1)
        distances = np.linalg.norm(points - self.centre, axis=-1)
        return distances - self.radius
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        margin = 0.1
        return (
            (self.centre[0] - self.radius - margin, self.centre[0] + self.radius + margin),
            (self.centre[1] - self.radius - margin, self.centre[1] + self.radius + margin),
            (self.centre[2] - self.radius - margin, self.centre[2] + self.radius + margin)
        )

class LShapedPrism(Domain3D):
    """L-shaped prism domain"""
    
    def __init__(self, thickness: float = 0.3):
        super().__init__("LShapedPrism")
        self.thickness = thickness
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # L-shape is union of two rectangles in xy-plane, extruded in z
        
        # Rectangle 1: [0, 1] x [0, thickness] x [0, 1]
        rect1X = np.maximum(0 - x, x - 1)
        rect1Y = np.maximum(0 - y, y - self.thickness)
        rect1Z = np.maximum(0 - z, z - 1)
        
        # Rectangle 2: [0, thickness] x [0, 1] x [0, 1]  
        rect2X = np.maximum(0 - x, x - self.thickness)
        rect2Y = np.maximum(0 - y, y - 1)
        rect2Z = np.maximum(0 - z, z - 1)
        
        # SDF for each rectangle
        sdf1 = self.boxSdf(rect1X, rect1Y, rect1Z)
        sdf2 = self.boxSdf(rect2X, rect2Y, rect2Z)
        
        # Union (minimum distance)
        return np.minimum(sdf1, sdf2)
    
    def boxSdf(self, dx: np.ndarray, dy: np.ndarray, dz: np.ndarray) -> np.ndarray:
        """SDF for box given edge distances"""
        outsideDist = np.sqrt(
            np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2
        )
        insideDist = np.maximum(np.maximum(dx, dy), dz)
        
        return np.where(
            (dx <= 0) & (dy <= 0) & (dz <= 0),
            insideDist, outsideDist
        )
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return ((0, 1), (0, 1), (0, 1))

class TorusSection(Domain3D):
    """Torus section for advanced geometry"""
    
    def __init__(self, R: float = 0.3, r: float = 0.15, 
                 centre: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        super().__init__("TorusSection")
        self.R = R  # Major radius
        self.r = r  # Minor radius
        self.centre = np.array(centre)
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Translate to torus centre
        px = x - self.centre[0]
        py = y - self.centre[1] 
        pz = z - self.centre[2]
        
        # Torus SDF
        q = np.sqrt(px**2 + py**2) - self.R
        return np.sqrt(q**2 + pz**2) - self.r
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        totalRadius = self.R + self.r + 0.1
        return (
            (self.centre[0] - totalRadius, self.centre[0] + totalRadius),
            (self.centre[1] - totalRadius, self.centre[1] + totalRadius), 
            (self.centre[2] - self.r - 0.1, self.centre[2] + self.r + 0.1)
        )

class CylinderWithHoles(Domain3D):
    """Cylinder with cylindrical holes"""
    
    def __init__(self, radius: float = 0.4, height: float = 0.8,
                 centre: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                 holePositions: Optional[list] = None, holeRadius: float = 0.1):
        super().__init__("CylinderWithHoles")
        self.radius = radius
        self.height = height
        self.centre = np.array(centre)
        self.holeRadius = holeRadius
        
        if holePositions is None:
            self.holePositions = [(0.3, 0.5), (0.7, 0.5)]  # (x, y) positions
        else:
            self.holePositions = holePositions
    
    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # Translate to centre
        px = x - self.centre[0]
        py = y - self.centre[1]
        pz = z - self.centre[2]
        
        # Main cylinder SDF
        radialDist = np.sqrt(px**2 + py**2) - self.radius
        heightDist = np.abs(pz) - self.height/2
        
        # Cylinder SDF (union of radial and height constraints)
        cylinderSdf = np.maximum(radialDist, heightDist)
        
        # Subtract holes
        for holeX, holeY in self.holePositions:
            holePx = x - holeX
            holePy = y - holeY
            holeDist = np.sqrt(holePx**2 + holePy**2) - self.holeRadius
            
            # Subtract hole (max with negative hole distance)
            cylinderSdf = np.maximum(cylinderSdf, -holeDist)
        
        return cylinderSdf
    
    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        margin = 0.1
        return (
            (self.centre[0] - self.radius - margin, self.centre[0] + self.radius + margin),
            (self.centre[1] - self.radius - margin, self.centre[1] + self.radius + margin),
            (self.centre[2] - self.height/2 - margin, self.centre[2] + self.height/2 + margin)
        )

class VoxelDomain(Domain3D):
    """Domain defined by a voxelised signed distance function from a .npy file."""

    def __init__(self, npy_path: str, name: str = "VoxelDomain"):
        super().__init__(name)
        print(f"Loading VoxelDomain from: {npy_path}", flush=True)
        print("Attempting to load .npy file...", flush=True)
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            print(".npy file loaded successfully.", flush=True)
            self.sdfData = data["sdf"]
            self._bounds = data["bounds"]
            self.pitch = data["pitch"]
            
            print(f"  - SDF shape: {self.sdfData.shape}", flush=True)
            print(f"  - Bounds: {self._bounds}", flush=True)
            print(f"  - Pitch: {self.pitch}", flush=True)
            
            # Check if the SDF is inverted
            centre_voxel_index = tuple(s // 2 for s in self.sdfData.shape)
            if self.sdfData[centre_voxel_index] > 0:
                print("SDF in .npy file seems to be inverted, flipping the sign.", flush=True)
                self.sdfData = -self.sdfData

            # Create grid coordinates for interpolation
            self.x_coords = np.linspace(self._bounds[0, 0], self._bounds[1, 0], self.sdfData.shape[0])
            self.y_coords = np.linspace(self._bounds[0, 1], self._bounds[1, 1], self.sdfData.shape[1])
            self.z_coords = np.linspace(self._bounds[0, 2], self._bounds[1, 2], self.sdfData.shape[2])
            
            print("VoxelDomain loaded successfully.", flush=True)

        except Exception as e:
            print(f"Error loading VoxelDomain from {npy_path}: {e}", flush=True)
            raise

    def sdf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Interpolate the SDF value from the voxel grid."""
        points = np.stack([x, y, z], axis=-1)
        
        # Interpolate
        return interpn((self.x_coords, self.y_coords, self.z_coords), self.sdfData, points, method='linear', bounds_error=False, fill_value=np.max(self.sdfData))

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return ((self._bounds[0, 0], self._bounds[1, 0]),
                (self._bounds[0, 1], self._bounds[1, 1]),
                (self._bounds[0, 2], self._bounds[1, 2]))

class DomainFactory:
    """Factory for creating domain instances"""
    
    AVAILABLE_DOMAINS = {
        'cube': UnitCube,
        'sphere': Sphere, 
        'lshape': LShapedPrism,
        'torus': TorusSection,
        'cylinder_holes': CylinderWithHoles,
        'voxel': VoxelDomain
    }
    
    @classmethod
    def createDomain(cls, domainType: str, **kwargs) -> Domain3D:
        """Create domain instance by type"""
        if domainType.endswith('.npy'):
            return VoxelDomain(domainType, name=domainType)
        if domainType not in cls.AVAILABLE_DOMAINS:
            # try to load as a npy file
            try:
                return VoxelDomain(domainType, name=domainType)
            except Exception:
                raise ValueError(f"Unknown domain type: {domainType}")
        
        domainClass = cls.AVAILABLE_DOMAINS[domainType]
        return domainClass(**kwargs)
    
    @classmethod
    def listDomains(cls) -> list:
        """List available domain types"""
        return list(cls.AVAILABLE_DOMAINS.keys())

def generateInteriorPoints(domain: Domain3D, nPoints: int, 
                           device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random points inside domain for PDE collocation"""
    (xMin, xMax), (yMin, yMax), (zMin, zMax) = domain.bounds()
    
    # Generate more points than needed, then filter
    nGen = min(nPoints * 5, 100000)  # Avoid memory issues
    
    xRand = np.random.uniform(xMin, xMax, nGen)
    yRand = np.random.uniform(yMin, yMax, nGen)
    zRand = np.random.uniform(zMin, zMax, nGen)
    
    # Filter points inside domain
    insideMask = domain.isInside(xRand, yRand, zRand)
    
    xInside = xRand[insideMask]
    yInside = yRand[insideMask]
    zInside = zRand[insideMask]
    
    # Take first nPoints (or all if less than requested)
    nActual = min(len(xInside), nPoints)
    
    if nActual < nPoints:
        print(f"Warning: Only generated {nActual}/{nPoints} interior points")
    
    xPoints = torch.tensor(xInside[:nActual], dtype=torch.float32, device=device, requires_grad=True)
    yPoints = torch.tensor(yInside[:nActual], dtype=torch.float32, device=device, requires_grad=True)
    zPoints = torch.tensor(zInside[:nActual], dtype=torch.float32, device=device, requires_grad=True)
    
    return xPoints.unsqueeze(1), yPoints.unsqueeze(1), zPoints.unsqueeze(1)

def generateBoundaryPoints(domain: Domain3D, nPoints: int,
                           device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate points on domain boundary"""
    (xMin, xMax), (yMin, yMax), (zMin, zMax) = domain.bounds()
    
    # Generate many candidates
    nGen = min(nPoints * 10, 200000)
    
    xRand = np.random.uniform(xMin, xMax, nGen)
    yRand = np.random.uniform(yMin, yMax, nGen)
    zRand = np.random.uniform(zMin, zMax, nGen)
    
    # Filter boundary points
    boundaryMask = domain.isBoundary(xRand, yRand, zRand, tol=1e-3)
    
    xBoundary = xRand[boundaryMask]
    yBoundary = yRand[boundaryMask]
    zBoundary = zRand[boundaryMask]
    
    nActual = min(len(xBoundary), nPoints)
    
    if nActual < nPoints:
        print(f"Warning: Only generated {nActual}/{nPoints} boundary points")
    
    xPoints = torch.tensor(xBoundary[:nActual], dtype=torch.float32, device=device, requires_grad=True)
    yPoints = torch.tensor(yBoundary[:nActual], dtype=torch.float32, device=device, requires_grad=True)
    zPoints = torch.tensor(zBoundary[:nActual], dtype=torch.float32, device=device, requires_grad=True)
    
    return xPoints.unsqueeze(1), yPoints.unsqueeze(1), zPoints.unsqueeze(1)

def testDomains():
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
        grid = domain.generateGrid(32, 32, 32)
        X, Y, Z, mask = grid['X'], grid['Y'], grid['Z'], grid['mask']
        
        # Plot interior points
        interiorIdx = np.where(mask)
        if len(interiorIdx[0]) > 0:
            sampleIdx = np.random.choice(len(interiorIdx[0]), 
                                        min(1000, len(interiorIdx[0])), replace=False)
            ax.scatter(X[interiorIdx][sampleIdx], 
                      Y[interiorIdx][sampleIdx],
                      Z[interiorIdx][sampleIdx], 
                      s=1, alpha=0.3)
        
        ax.set_title(domain.name)
        ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    testDomains()