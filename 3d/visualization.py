#!/usr/bin/env python3
"""
Advanced 3D Visualization Suite for Delta-PINN Heat Diffusion
Volumetric rendering, isosurfaces, animations, and error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
import pickle
import torch
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')
import sys
from tqdm import tqdm

# Try to import advanced plotting libraries
try:
    import plotly.graph_objects as go
    import plotly.subplots as make_subplots
    from plotly.offline import plot as plotly_plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Using matplotlib only.")

from domain_shapes import Domain3D, DomainFactory
from delta_pinn_3d import DeltaPINN3D, loadTrainedModel, predictSolution3d
from numerical_solution import HeatSolver3D

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Scikit-image not available. Surface plotting disabled.")

from scipy.interpolate import griddata, RBFInterpolator

class HeatVisualization3D:
    """Advanced 3D visualization for heat diffusion"""
    
    def __init__(self, saveDir: str = './visualizations'):
        self.saveDir = saveDir
        os.makedirs(saveDir, exist_ok=True)
        
        # Custom colormap for heat visualization
        self.heatCmap = LinearSegmentedColormap.from_list(
            'heat_custom',
            ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000'],
            N=256
        )
        
        # Error colormap
        self.errorCmap = LinearSegmentedColormap.from_list(
            'error_custom',
            ['#000040', '#0000FF', '#FFFFFF', '#FF0000', '#400000'],
            N=256
        )
    
    def plot3dIsosurfaces(self, solutionData: Dict, timeIdx: int = -1,
                           isoValues: Optional[List[float]] = None,
                           title: str = "3D Heat Isosurfaces", saveName: str = "isosurfaces"):
        """Plot 3D isosurfaces of temperature field"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Using matplotlib slice visualization instead.")
            return self.plotMatplotlibSlices(solutionData, timeIdx, title, saveName)
        
        # Extract data
        X = solutionData.get('X')
        Y = solutionData.get('Y') 
        Z = solutionData.get('Z')
        u = solutionData['u'] if 'u' in solutionData else solutionData['solutions'][timeIdx]
        
        # If u is 1D (from numerical solver), convert to grid
        if len(u.shape) == 1:
            solverData = solutionData
            uGrid = np.full(X.shape, np.nan)
            uGrid[solverData['interior_indices']] = u
            u = uGrid
        
        # Default isovalues
        if isoValues is None:
            uClean = u[~np.isnan(u)]
            print(f"Number of non-NaN points: {len(uClean)}")
            if len(uClean) > 0:
                uMax = np.max(uClean)
                print(f"Max solution value for iso-surface: {uMax}")
                isoValues = [0.1 * uMax, 0.3 * uMax, 0.5 * uMax, 0.7 * uMax]
                print(f"Iso-values: {isoValues}")
            else:
                isoValues = [0.1, 0.3, 0.5, 0.7]
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add isosurfaces
        colors = ['blue', 'green', 'yellow', 'red', 'purple']
        
        plotlyColorscale = [
            [0.0, '#000080'], [1/6, '#0000FF'], [2/6, '#00FFFF'],
            [3/6, '#FFFF00'], [4/6, '#FF8000'], [5/6, '#FF0000'], [1.0, '#800000']
        ]
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"Z shape: {Z.shape}")
        print(f"u shape: {u.shape}")
        print(f"u dtype: {u.dtype}")
        print(f"Number of NaNs in u: {np.isnan(u).sum()}")
        print(f"Number of non-NaNs in u: {np.count_nonzero(~np.isnan(u))}")

        for i, isoVal in enumerate(isoValues):
            if isoVal <= np.nanmax(u):
                fig.add_trace(go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=u.flatten(),
                    isomin=isoVal,
                    isomax=isoVal + 1e-9,
                    surface_count=1,
                    colorscale=plotlyColorscale,
                    showscale=False,
                    opacity=0.7,
                    name=f'T = {isoVal:.3f}'
                ))
        
        # Layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        # Save
        outputPath = os.path.join(self.saveDir, f"{saveName}.html")
        plotly_plot(fig, filename=outputPath, auto_open=False)
        print(f"3D isosurfaces saved to {outputPath}")
        
        return fig, outputPath
    
    def plotMatplotlibSlices(self, solutionData: Dict, timeIdx: int = -1,
                               title: str = "Temperature Slices", saveName: str = "slices"):
        """Fallback matplotlib visualization with 2D slices"""
        
        # Extract data
        X = solutionData.get('X')
        Y = solutionData.get('Y')
        Z = solutionData.get('Z') 
        u = solutionData['u'] if 'u' in solutionData else solutionData['solutions'][timeIdx]
        
        # If u is 1D, convert to grid
        if len(u.shape) == 1:
            uGrid = np.full(X.shape, np.nan)
            uGrid[solutionData['interior_indices']] = u
            u = uGrid
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Middle slice indices
        nx, ny, nz = u.shape
        midX, midY, midZ = nx//2, ny//2, nz//2
        
        # XY slice (middle Z)
        im1 = axes[0,0].contourf(X[midZ], Y[midZ], u[midZ], 
                                levels=20, cmap=self.heatCmap)
        axes[0,0].set_title(f'XY Slice (Z = {Z[midZ,0,midZ]:.2f})')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0,0])
        
        # XZ slice (middle Y) 
        im2 = axes[0,1].contourf(X[:,midY,:], Z[:,midY,:], u[:,midY,:],
                                levels=20, cmap=self.heatCmap)
        axes[0,1].set_title(f'XZ Slice (Y = {Y[0,midY,0]:.2f})')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0,1])
        
        # YZ slice (middle X)
        im3 = axes[1,0].contourf(Y[midX], Z[midX], u[midX],
                                levels=20, cmap=self.heatCmap)
        axes[1,0].set_title(f'YZ Slice (X = {X[midX,0,0]:.2f})')
        axes[1,0].set_xlabel('Y')
        axes[1,0].set_ylabel('Z')
        plt.colorbar(im3, ax=axes[1,0])
        
        # 3D scatter of high-temperature regions
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Show points above threshold
        threshold = 0.3 * np.nanmax(u)
        highTempMask = (u > threshold) & (~np.isnan(u))
        
        if np.any(highTempMask):
            xx, yy, zz = np.where(highTempMask)
            temps = u[highTempMask]
            scatter = ax3d.scatter(X[xx, yy, zz], Y[xx, yy, zz], Z[xx, yy, zz],
                                 c=temps, cmap=self.heatCmap, s=20, alpha=0.6)
            plt.colorbar(scatter, ax=ax3d, shrink=0.8)
        
        ax3d.set_title('High Temperature Regions')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        
        plt.tight_layout()
        
        # Save
        outputPath = os.path.join(self.saveDir, f"{saveName}.png")
        plt.savefig(outputPath, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Temperature slices saved to {outputPath}")
        
        return fig
    
    def plotVolumetricRendering(self, model: DeltaPINN3D, domain: Domain3D,
                                 tMax: float = 1.0, timeSteps: int = 20,
                                 opacityScale: float = 0.1,
                                 title: str = "Volumetric Temperature",
                                 saveName: str = "volumetric"):
        """3D volumetric rendering of temperature field with domain boundary and time slider"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for volumetric rendering.")
            return None, None

        # Create time steps
        timeValues = np.linspace(0, tMax, timeSteps)
        
        # Pre-calculate solutions
        solutions = []
        for t in timeValues:
            print(f"Predicting solution for t={t:.4f}")
            solutionData = predictSolution3d(model, domain, t, nX=48, nY=48, nZ=48, device='cpu')
            solutions.append(solutionData)

        # Find global min and max temperatures
        allTemps = np.concatenate([sol['u'].flatten() for sol in solutions])
        allTemps = allTemps[~np.isnan(allTemps)]
        cmin = np.percentile(allTemps, 1)
        cmax = np.percentile(allTemps, 99)

        # Create the volumetric heatmap plot
        fig = go.Figure()

        # Add a volume trace for each time step
        for i, solData in enumerate(solutions):
            u = solData['u']
            minVal = np.nanmin(u)
            uNoNan = np.nan_to_num(u, nan=minVal)
            
            fig.add_trace(go.Volume(
                x=solData['X'].flatten(),
                y=solData['Y'].flatten(),
                z=solData['Z'].flatten(),
                value=uNoNan.flatten(),
                opacity=0.2,
                surface_count=15,
                colorscale='Hot',
                cmin=cmin,
                cmax=cmax,
                opacityscale=[[0, 0], [0.2, 0.1], [0.5, 0.3], [0.8, 0.7], [1, 1]],
                name=f't = {timeValues[i]:.2f}',
                visible=(i == 0) # Make only the first trace visible
            ))

        # Create slider
        steps = []
        for i in range(len(solutions)):
            step = dict(
                method="update",
                args=[{"visible": [False] * (len(solutions) + 1)}, # +1 for the isosurface
                      {"title": f"{title} (t={timeValues[i]:.2f})"}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps
        )]
        
        # Generate domain boundary for visualization
        gridData = domain.generateGrid(48, 48, 48)
        XDom, YDom, ZDom = gridData['X'], gridData['Y'], gridData['Z']
        sdfVals = gridData['sdf']

        # Add domain boundary as a mesh
        fig.add_trace(go.Isosurface(
            x=XDom.flatten(),
            y=YDom.flatten(),
            z=ZDom.flatten(),
            value=sdfVals.flatten(),
            isomin=0,
            isomax=0,
            surface_count=1,
            colorscale=[[0, 'grey'], [1, 'grey']],
            showscale=False,
            opacity=0.1,
            name='Domain Boundary'
        ))
        
        # Make the isosurface visible for all steps
        for step in steps:
            step["args"][0]["visible"][-1] = True

        fig.update_layout(
            title=f"{title} (t={timeValues[0]:.2f})",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700,
            sliders=sliders
        )
        
        # Save
        outputPath = os.path.join(self.saveDir, f"{saveName}.html")
        plotly_plot(fig, filename=outputPath, auto_open=False)
        print(f"Volumetric rendering with domain boundary saved to {outputPath}")
        
        return fig, outputPath

    def plotSurfaceHeatmap(self, model: DeltaPINN3D, domain: Domain3D,
                             tMax: float = 1.0, timeSteps: int = 20,
                             title: str = "Surface Temperature Heatmap",
                             saveName: str = "surface_heatmap",
                             smoothing: float = 1.0):
        """Plot temperature heatmap on the 3D domain surface with a time slider."""
        
        if not PLOTLY_AVAILABLE or not SKIMAGE_AVAILABLE:
            print("Plotly or Scikit-image not available for surface heatmap.")
            return None, None, None, None

        # Generate a high-resolution grid for surface extraction
        gridData = domain.generateGrid(256, 256, 256)
        XDom, YDom, ZDom = gridData['X'], gridData['Y'], gridData['Z']
        sdfVals = gridData['sdf']

        # Use marching cubes to find the surface mesh
        try:
            verts, faces, _, _ = measure.marching_cubes(sdfVals, level=0)
            # Scale vertices to the domain bounds
            bounds = domain.bounds()
            verts[:, 0] = verts[:, 0] * (bounds[0][1] - bounds[0][0]) / (256 - 1) + bounds[0][0]
            verts[:, 1] = verts[:, 1] * (bounds[1][1] - bounds[1][0]) / (256 - 1) + bounds[1][0]
            verts[:, 2] = verts[:, 2] * (bounds[2][1] - bounds[2][0]) / (256 - 1) + bounds[2][0]
        except (ValueError, RuntimeError) as e:
            print(f"Marching cubes failed: {e}. Cannot generate surface plot.")
            return None, None, None, None

        if len(verts) == 0:
            print("Marching cubes did not find a surface.")
            return None, None, None, None

        # Create time steps
        timeValues = np.linspace(0, tMax, timeSteps)
        
        # Pre-calculate solutions and interpolate temperatures
        interpolatedTempsList = []
        for t in tqdm(timeValues, desc="Calculating surface plot", file=sys.stdout):
            solutionData = predictSolution3d(model, domain, t, nX=48, nY=48, nZ=48, device='cpu')
            
            uSol = solutionData['u']
            solGridPoints = np.array([solutionData['X'].flatten(), solutionData['Y'].flatten(), solutionData['Z'].flatten()]).T
            solValues = uSol.flatten()
            
            validIndices = ~np.isnan(solValues)
            if not np.any(validIndices):
                print(f"No valid solution data to interpolate for t={t:.4f}")
                interpolatedTempsList.append(np.zeros(len(verts)))
                continue

            rbfInterp = RBFInterpolator(solGridPoints[validIndices], solValues[validIndices], kernel='thin_plate_spline', epsilon=smoothing)
            interpolatedTemps = rbfInterp(verts)

            if np.isnan(interpolatedTemps).any():
                print(f"Could not interpolate temperatures for all surface vertices for t={t:.4f}. Filling with 0.")
                interpolatedTemps = np.nan_to_num(interpolatedTemps, nan=0.0)
            
            interpolatedTempsList.append(interpolatedTemps)

        # Find global min and max temperatures
        allTemps = np.concatenate(interpolatedTempsList)
        cmin = np.percentile(allTemps, 10)
        cmax = np.percentile(allTemps, 90)

        # Create Plotly figure
        fig = go.Figure()

        # Add a mesh trace for each time step
        for i, interpolatedTemps in enumerate(interpolatedTempsList):
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=interpolatedTemps,
                colorscale='Jet',
                cmin=cmin,
                cmax=cmax,
                colorbar_title='Temperature',
                name=f't = {timeValues[i]:.2f}',
                visible=(i == 0)
            ))

        # Create slider
        steps = []
        for i in range(len(interpolatedTempsList)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(interpolatedTempsList)},
                      {"title": f"{title} (t={timeValues[i]:.2f})"}],
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Time: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            title=f"{title} (t={timeValues[0]:.2f})",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700,
            sliders=sliders
        )

        # Save
        outputPath = os.path.join(self.saveDir, f"{saveName}.html")
        plotly_plot(fig, filename=outputPath, auto_open=False)
        print(f"Surface heatmap saved to {outputPath}")

        return fig, outputPath, outputPath, outputPath
    
    def createComparisonPlot(self, pinnData: Dict, numericalData: Dict, 
                              timeIdx: int = -1, saveName: str = "comparison"):
        """Create side-by-side comparison of PINN vs numerical solution"""
        
        # Extract PINN data
        pinnU = pinnData['u']
        X = pinnData['X']
        Y = pinnData['Y'] 
        Z = pinnData['Z']
        
        # Extract numerical data and convert to grid
        numU = numericalData['solutions'][timeIdx]
        if len(numU.shape) == 1:
            numUGrid = np.full(X.shape, np.nan)
            numUGrid[numericalData['interior_indices']] = numU
            numU = numUGrid
        
        # Compute error where both solutions exist
        maskValid = ~(np.isnan(pinnU) | np.isnan(numU))
        error = np.full_like(pinnU, np.nan)
        error[maskValid] = np.abs(pinnU[maskValid] - numU[maskValid])
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PINN vs Numerical Solution Comparison', fontsize=16)
        
        # Middle slice indices
        nx, ny, nz = pinnU.shape
        midX, midY, midZ = nx//2, ny//2, nz//2
        
        # PINN solution slices
        im1 = axes[0,0].contourf(X[midZ], Y[midZ], pinnU[midZ], 
                                levels=20, cmap=self.heatCmap)
        axes[0,0].set_title('PINN - XY Slice')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].contourf(X[:,midY,:], Z[:,midY,:], pinnU[:,midY,:],
                                levels=20, cmap=self.heatCmap)
        axes[0,1].set_title('PINN - XZ Slice')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0,1])
        
        im3 = axes[0,2].contourf(Y[midX], Z[midX], pinnU[midX],
                                levels=20, cmap=self.heatCmap)
        axes[0,2].set_title('PINN - YZ Slice')
        axes[0,2].set_xlabel('Y')
        axes[0,2].set_ylabel('Z')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Numerical solution slices
        im4 = axes[1,0].contourf(X[midZ], Y[midZ], numU[midZ],
                                levels=20, cmap=self.heatCmap)
        axes[1,0].set_title('Numerical - XY Slice')
        axes[1,0].set_xlabel('X')
        axes[1,0].set_ylabel('Y')
        plt.colorbar(im4, ax=axes[1,0])
        
        im5 = axes[1,1].contourf(X[:,midY,:], Z[:,midY,:], numU[:,midY,:],
                                levels=20, cmap=self.heatCmap)
        axes[1,1].set_title('Numerical - XZ Slice')
        axes[1,1].set_xlabel('X')
        axes[1,1].set_ylabel('Z')
        plt.colorbar(im5, ax=axes[1,1])
        
        im6 = axes[1,2].contourf(Y[midX], Z[midX], numU[midX],
                                levels=20, cmap=self.heatCmap)
        axes[1,2].set_title('Numerical - YZ Slice')
        axes[1,2].set_xlabel('Y')
        axes[1,2].set_ylabel('Z')
        plt.colorbar(im6, ax=axes[1,2])
        
        plt.tight_layout()
        
        # Save comparison
        outputPath = os.path.join(self.saveDir, f"{saveName}_solutions.png")
        plt.savefig(outputPath, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create error analysis plot
        self.plotErrorAnalysis(error, X, Y, Z, saveName)
        
        print(f"Comparison plots saved with prefix {saveName}")
        
        return fig
    
    def plotErrorAnalysis(self, error: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                           saveName: str):
        """Plot detailed error analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Error Analysis: |PINN - Numerical|', fontsize=16)
        
        # Middle slice indices
        nx, ny, nz = error.shape
        midX, midY, midZ = nx//2, ny//2, nz//2
        
        # Error slices
        im1 = axes[0,0].contourf(X[midZ], Y[midZ], error[midZ],
                                levels=20, cmap=self.errorCmap)
        axes[0,0].set_title(f'Error - XY Slice (Z = {Z[midZ,0,midZ]:.2f})')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].contourf(X[:,midY,:], Z[:,midY,:], error[:,midY,:],
                                levels=20, cmap=self.errorCmap)
        axes[0,1].set_title(f'Error - XZ Slice (Y = {Y[0,midY,0]:.2f})')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Error histogram
        errorValid = error[~np.isnan(error)]
        if len(errorValid) > 0:
            axes[1,0].hist(errorValid, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[1,0].set_xlabel('Absolute Error')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Error Distribution')
            axes[1,0].grid(True, alpha=0.3)
            
            # Error statistics
            meanError = np.mean(errorValid)
            maxError = np.max(errorValid)
            stdError = np.std(errorValid)
            
            axes[1,0].axvline(meanError, color='blue', linestyle='--', 
                             label=f'Mean: {meanError:.4f}')
            axes[1,0].axvline(maxError, color='red', linestyle='--',
                             label=f'Max: {maxError:.4f}')
            axes[1,0].legend()
        
        # Error statistics text
        if len(errorValid) > 0:
            statsText = f"""Error Statistics:
Mean Error: {np.mean(errorValid):.4e}
Max Error: {np.max(errorValid):.4e}
Std Error: {np.std(errorValid):.4e}
L2 Error: {np.sqrt(np.mean(errorValid**2)):.4e}
Valid Points: {len(errorValid)}"""
            
            axes[1,1].text(0.1, 0.9, statsText, transform=axes[1,1].transAxes,
                          fontsize=12, verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            axes[1,1].set_title('Error Statistics')
            axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Save error plot
        outputPath = os.path.join(self.saveDir, f"{saveName}_error.png")
        plt.savefig(outputPath, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Error analysis saved to {outputPath}")
    
    def createAnimation(self, solutionData: Dict, fps: int = 10,
                        title: str = "Heat Diffusion Animation",
                        saveName: str = "animation"):
        """Create animated visualization of heat diffusion over time"""
        
        times = solutionData['times']
        solutions = solutionData['solutions']
        
        if len(solutions) < 2:
            print("Need at least 2 time points for animation")
            return None
        
        # Get grid data
        X = solutionData.get('X')
        Y = solutionData.get('Y')
        Z = solutionData.get('Z')
        
        # Setup figure for animation
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Middle slice indices
        if X is not None:
            nx, ny, nz = X.shape
        else:
            # Estimate shape from solution data
            nx = ny = nz = int(round(len(solutions[0])**(1/3)))
            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)
            z = np.linspace(0, 1, nz)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        midX, midY, midZ = nx//2, ny//2, nz//2
        
        # Initialize plots
        def initFrame():
            for ax in axes.flat:
                ax.clear()
            return []
        
        # Animation function
        def animate(frame):
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            # Get current solution
            u = solutions[frame]
            
            # Convert to grid if needed
            if len(u.shape) == 1:
                uGrid = np.full(X.shape, np.nan)
                if 'interior_indices' in solutionData:
                    uGrid[solutionData['interior_indices']] = u
                u = uGrid
            
            # Determine global colormap range
            uMin = np.nanmin([np.nanmin(sol) for sol in solutions if len(sol) > 0])
            uMax = np.nanmax([np.nanmax(sol) for sol in solutions if len(sol) > 0])
            
            # XY slice
            im1 = axes[0,0].contourf(X[midZ], Y[midZ], u[midZ],
                                    levels=20, cmap=self.heatCmap, 
                                    vmin=uMin, vmax=uMax)
            axes[0,0].set_title(f'XY Slice - t = {times[frame]:.3f}')
            axes[0,0].set_xlabel('X')
            axes[0,0].set_ylabel('Y')
            
            # XZ slice
            im2 = axes[0,1].contourf(X[:,midY,:], Z[:,midY,:], u[:,midY,:],
                                    levels=20, cmap=self.heatCmap,
                                    vmin=uMin, vmax=uMax)
            axes[0,1].set_title(f'XZ Slice - t = {times[frame]:.3f}')
            axes[0,1].set_xlabel('X')
            axes[0,1].set_ylabel('Z')
            
            # Temperature evolution plot
            maxTemps = [np.nanmax(sol) for sol in solutions[:frame+1]]
            axes[1,0].plot(times[:frame+1], maxTemps, 'r-', linewidth=2)
            axes[1,0].set_xlabel('Time')
            axes[1,0].set_ylabel('Max Temperature')
            axes[1,0].set_title('Temperature Evolution')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_xlim(times[0], times[-1])
            axes[1,0].set_ylim(0, uMax * 1.1)
            
            # Current time marker
            axes[1,1].clear()
            axes[1,1].text(0.5, 0.7, f'Time: {times[frame]:.3f}', 
                          transform=axes[1,1].transAxes, fontsize=20, 
                          ha='center', va='center')
            axes[1,1].text(0.5, 0.3, f'Max Temp: {np.nanmax(u):.3f}',
                          transform=axes[1,1].transAxes, fontsize=16,
                          ha='center', va='center')
            axes[1,1].set_title('Current State')
            axes[1,1].axis('off')
            
            plt.tight_layout()
            return []
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=initFrame,
                           frames=len(solutions), interval=1000//fps, blit=False)
        
        # Save as GIF
        gifPath = os.path.join(self.saveDir, f"{saveName}.gif")
        anim.save(gifPath, writer=PillowWriter(fps=fps), dpi=100)
        print(f"Animation saved to {gifPath}")
        
        plt.show()
        return anim
    
    def plotHeatSources(self, heatSources: List[Dict], domain: Domain3D,
                         title: str = "Heat Source Configuration",
                         saveName: str = "heat_sources"):
        """Visualize heat source locations and domain"""
        
        fig = plt.figure(figsize=(12, 8))
        
        # 3D plot
        ax3d = fig.add_subplot(121, projection='3d')
        
        # Plot domain boundary (sample points)
        gridData = domain.generateGrid(32, 32, 32)
        X, Y, Z = gridData['X'], gridData['Y'], gridData['Z']
        mask = gridData['mask']
        
        # Plot domain boundary
        boundaryPoints = []
        for i in range(0, X.shape[0], 4):
            for j in range(0, X.shape[1], 4):
                for k in range(0, X.shape[2], 4):
                    if mask[i,j,k]:
                        boundaryPoints.append([X[i,j,k], Y[i,j,k], Z[i,j,k]])
        
        if boundaryPoints:
            boundaryPoints = np.array(boundaryPoints)
            ax3d.scatter(boundaryPoints[:,0], boundaryPoints[:,1], boundaryPoints[:,2],
                        c='lightblue', alpha=0.1, s=1)
        
        # Plot heat sources
        colors = ['red', 'orange', 'yellow', 'purple', 'green']
        for i, source in enumerate(heatSources):
            x0, y0, z0 = source['position']
            amplitude = source['amplitude']
            sigma = source.get('sigma', 0.05)
            
            # Plot source location
            ax3d.scatter([x0], [y0], [z0], c=colors[i % len(colors)], 
                       s=200 * amplitude, alpha=0.8, 
                       label=f'Source {i+1} (A={amplitude:.2f})')
            
            # Plot Gaussian influence region
            theta = np.linspace(0, 2*np.pi, 20)
            phi = np.linspace(0, np.pi, 20)
            r = 3 * sigma  # 3-sigma region
            
            xSphere = x0 + r * np.outer(np.cos(theta), np.sin(phi))
            ySphere = y0 + r * np.outer(np.sin(theta), np.sin(phi))
            zSphere = z0 + r * np.outer(np.ones(np.size(theta)), np.cos(phi))
            
            ax3d.plot_wireframe(xSphere, ySphere, zSphere, 
                               alpha=0.3, color=colors[i % len(colors)])
        
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title('3D Heat Sources')
        ax3d.legend()
        
        # 2D projection (XY plane)
        ax2d = fig.add_subplot(122)
        
        # Plot domain outline in XY
        xBounds, yBounds, zBounds = domain.bounds()
        
        # Sample domain boundary in XY plane
        xTest = np.linspace(xBounds[0], xBounds[1], 100)
        yTest = np.linspace(yBounds[0], yBounds[1], 100)
        XTest, YTest = np.meshgrid(xTest, yTest)
        ZTest = np.full_like(XTest, (zBounds[0] + zBounds[1]) / 2)
        
        insideMask = domain.isInside(XTest, YTest, ZTest)
        ax2d.contour(XTest, YTest, insideMask.astype(float), levels=[0.5], colors='black')
        
        # Plot heat sources
        for i, source in enumerate(heatSources):
            x0, y0, z0 = source['position']
            amplitude = source['amplitude']
            sigma = source.get('sigma', 0.05)
            
            ax2d.scatter([x0], [y0], c=colors[i % len(colors)], 
                       s=200 * amplitude, alpha=0.8, 
                       label=f'Source {i+1}')
            
            # Gaussian influence circle
            circle = patches.Circle((x0, y0), 3*sigma, fill=False, 
                                  color=colors[i % len(colors)], alpha=0.5)
            ax2d.add_patch(circle)
        
        ax2d.set_xlabel('X')
        ax2d.set_ylabel('Y')
        ax2d.set_title('XY Projection')
        ax2d.legend()
        ax2d.set_aspect('equal')
        ax2d.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        outputPath = os.path.join(self.saveDir, f"{saveName}.png")
        plt.savefig(outputPath, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Heat source visualization saved to {outputPath}")
        
        return fig

def visualizeTrainingResults(modelPath: str, numericalPath: Optional[str] = None,
                             tEval: float = 1.0, device: str = 'cpu'):
    """Complete visualization pipeline for trained model"""
    
    print(f"\n=== Visualization Pipeline ===")
    print(f"Model: {modelPath}")
    if numericalPath:
        print(f"Numerical: {numericalPath}")
    print(f"Evaluation time: {tEval}")
    
    # Load trained model
    model, checkpoint = loadTrainedModel(modelPath, device)
    domainName = checkpoint['domain_name']
    heatSources = checkpoint['heat_sources']
    
    print(f"Loaded model for domain: {domainName}")
    print(f"Heat sources: {len(heatSources)}")
    
    # Create domain
    domain = DomainFactory.createDomain(domainName.lower())
    
    # Initialize visualizer
    viz = HeatVisualization3D()
    
    # Plot heat source configuration
    viz.plotHeatSources(heatSources, domain, saveName=f"sources_{domainName}")
    
    # Predict PINN solution
    print("Predicting PINN solution...")
    pinnSolution = predictSolution3d(model, domain, tEval, nX=48, nY=48, nZ=48, device=device)
    
    # Create PINN visualizations
    print("Creating PINN visualizations...")
    viz.plot3dIsosurfaces(pinnSolution, title=f"PINN Solution - {domainName}", 
                           saveName=f"pinn_iso_{domainName}")
    
    viz.plotMatplotlibSlices(pinnSolution, title=f"PINN Solution - {domainName}",
                               saveName=f"pinn_slices_{domainName}")
    
    # Load and visualize numerical solution if available
    if numericalPath and os.path.exists(numericalPath):
        print("Loading numerical solution...")
        with open(numericalPath, 'rb') as f:
            numericalData = pickle.load(f)
        
        # Find closest time point
        timeIdx = np.argmin(np.abs(numericalData['times'] - tEval))
        actualTime = numericalData['times'][timeIdx]
        print(f"Using numerical solution at t = {actualTime:.4f}")
        
        # Create comparison
        print("Creating comparison plots...")
        viz.createComparisonPlot(pinnSolution, numericalData, timeIdx, 
                                  saveName=f"comparison_{domainName}")
        
        # Create animation if multiple time points
        if len(numericalData['times']) > 5:
            print("Creating animation...")
            viz.createAnimation(numericalData, fps=5, 
                               title=f"Heat Diffusion - {domainName}",
                               saveName=f"animation_{domainName}")
    
    print("Visualization complete!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Heat Diffusion Visualization")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--numerical_path', type=str, help='Path to numerical solution')
    parser.add_argument('--t_eval', type=float, default=1.0, help='Evaluation time')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference')
    
    args = parser.parse_args()
    
    visualizeTrainingResults(args.model_path, args.numerical_path, args.t_eval, args.device)