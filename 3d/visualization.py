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
from delta_pinn_3d import DeltaPINN3D, load_trained_model, predict_solution_3d
from numerical_solution import HeatSolver3D

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Scikit-image not available. Surface plotting disabled.")

from scipy.interpolate import griddata

class HeatVisualization3D:
    """Advanced 3D visualization for heat diffusion"""
    
    def __init__(self, save_dir: str = './visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Custom colormap for heat visualization
        self.heat_cmap = LinearSegmentedColormap.from_list(
            'heat_custom',
            ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800000'],
            N=256
        )
        
        # Error colormap
        self.error_cmap = LinearSegmentedColormap.from_list(
            'error_custom',
            ['#000040', '#0000FF', '#FFFFFF', '#FF0000', '#400000'],
            N=256
        )
    
    def plot_3d_isosurfaces(self, solution_data: Dict, time_idx: int = -1,
                           iso_values: Optional[List[float]] = None,
                           title: str = "3D Heat Isosurfaces", save_name: str = "isosurfaces"):
        """Plot 3D isosurfaces of temperature field"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Using matplotlib slice visualization instead.")
            return self._plot_matplotlib_slices(solution_data, time_idx, title, save_name)
        
        # Extract data
        X = solution_data.get('X')
        Y = solution_data.get('Y') 
        Z = solution_data.get('Z')
        u = solution_data['u'] if 'u' in solution_data else solution_data['solutions'][time_idx]
        
        # If u is 1D (from numerical solver), convert to grid
        if len(u.shape) == 1:
            solver_data = solution_data
            u_grid = np.full(X.shape, np.nan)
            u_grid[solver_data['interior_indices']] = u
            u = u_grid
        
        # Default isovalues
        if iso_values is None:
            u_clean = u[~np.isnan(u)]
            print(f"Number of non-NaN points: {len(u_clean)}")
            if len(u_clean) > 0:
                u_max = np.max(u_clean)
                print(f"Max solution value for iso-surface: {u_max}")
                iso_values = [0.1 * u_max, 0.3 * u_max, 0.5 * u_max, 0.7 * u_max]
                print(f"Iso-values: {iso_values}")
            else:
                iso_values = [0.1, 0.3, 0.5, 0.7]
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add isosurfaces
        colors = ['blue', 'green', 'yellow', 'red', 'purple']
        
        plotly_colorscale = [
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

        for i, iso_val in enumerate(iso_values):
            if iso_val <= np.nanmax(u):
                fig.add_trace(go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=u.flatten(),
                    isomin=iso_val,
                    isomax=iso_val + 1e-9,
                    surface_count=1,
                    colorscale=plotly_colorscale,
                    showscale=False,
                    opacity=0.7,
                    name=f'T = {iso_val:.3f}'
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
        output_path = os.path.join(self.save_dir, f"{save_name}.html")
        plotly_plot(fig, filename=output_path, auto_open=False)
        print(f"3D isosurfaces saved to {output_path}")
        
        return fig
    
    def _plot_matplotlib_slices(self, solution_data: Dict, time_idx: int = -1,
                               title: str = "Temperature Slices", save_name: str = "slices"):
        """Fallback matplotlib visualization with 2D slices"""
        
        # Extract data
        X = solution_data.get('X')
        Y = solution_data.get('Y')
        Z = solution_data.get('Z') 
        u = solution_data['u'] if 'u' in solution_data else solution_data['solutions'][time_idx]
        
        # If u is 1D, convert to grid
        if len(u.shape) == 1:
            u_grid = np.full(X.shape, np.nan)
            u_grid[solution_data['interior_indices']] = u
            u = u_grid
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # Middle slice indices
        nx, ny, nz = u.shape
        mid_x, mid_y, mid_z = nx//2, ny//2, nz//2
        
        # XY slice (middle Z)
        im1 = axes[0,0].contourf(X[mid_z], Y[mid_z], u[mid_z], 
                                levels=20, cmap=self.heat_cmap)
        axes[0,0].set_title(f'XY Slice (Z = {Z[mid_z,0,mid_z]:.2f})')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0,0])
        
        # XZ slice (middle Y) 
        im2 = axes[0,1].contourf(X[:,mid_y,:], Z[:,mid_y,:], u[:,mid_y,:],
                                levels=20, cmap=self.heat_cmap)
        axes[0,1].set_title(f'XZ Slice (Y = {Y[0,mid_y,0]:.2f})')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0,1])
        
        # YZ slice (middle X)
        im3 = axes[1,0].contourf(Y[mid_x], Z[mid_x], u[mid_x],
                                levels=20, cmap=self.heat_cmap)
        axes[1,0].set_title(f'YZ Slice (X = {X[mid_x,0,0]:.2f})')
        axes[1,0].set_xlabel('Y')
        axes[1,0].set_ylabel('Z')
        plt.colorbar(im3, ax=axes[1,0])
        
        # 3D scatter of high-temperature regions
        ax3d = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Show points above threshold
        threshold = 0.3 * np.nanmax(u)
        high_temp_mask = (u > threshold) & (~np.isnan(u))
        
        if np.any(high_temp_mask):
            xx, yy, zz = np.where(high_temp_mask)
            temps = u[high_temp_mask]
            scatter = ax3d.scatter(X[xx, yy, zz], Y[xx, yy, zz], Z[xx, yy, zz],
                                 c=temps, cmap=self.heat_cmap, s=20, alpha=0.6)
            plt.colorbar(scatter, ax=ax3d, shrink=0.8)
        
        ax3d.set_title('High Temperature Regions')
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        
        plt.tight_layout()
        
        # Save
        output_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Temperature slices saved to {output_path}")
        
        return fig
    
    def plot_volumetric_rendering(self, solution_data: Dict, domain: Domain3D, time_idx: int = -1,
                                 opacity_scale: float = 0.1,
                                 title: str = "Volumetric Temperature",
                                 save_name: str = "volumetric"):
        """3D volumetric rendering of temperature field with domain boundary"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for volumetric rendering. Using slice view.")
            return self._plot_matplotlib_slices(solution_data, time_idx, title, save_name)
        
        # Extract solution data
        X_sol, Y_sol, Z_sol = solution_data.get('X'), solution_data.get('Y'), solution_data.get('Z')
        u = solution_data['u'] if 'u' in solution_data else solution_data['solutions'][time_idx]
        
        if len(u.shape) == 1:
            u_grid = np.full(X_sol.shape, np.nan)
            u_grid[solution_data['interior_indices']] = u
            u = u_grid
        
        u_clean = u[~np.isnan(u)]
        if len(u_clean) == 0:
            print("No valid temperature data found.")
            return None
        
        min_val = np.nanmin(u)
        u_no_nan = np.nan_to_num(u, nan=min_val)

        # Create the volumetric heatmap plot
        fig = go.Figure()

        fig.add_trace(go.Volume(
            x=X_sol.flatten(),
            y=Y_sol.flatten(),
            z=Z_sol.flatten(),
            value=u_no_nan.flatten(),
            opacity=0.2,
            surface_count=15,
            colorscale='Hot',
            opacityscale=[[0, 0], [0.2, 0.1], [0.5, 0.3], [0.8, 0.7], [1, 1]],
            name='Heatmap'
        ))

        # Generate domain boundary for visualization
        grid_data = domain.generate_grid(48, 48, 48)
        X_dom, Y_dom, Z_dom = grid_data['X'], grid_data['Y'], grid_data['Z']
        sdf_vals = grid_data['sdf']

        # Add domain boundary as a mesh
        fig.add_trace(go.Isosurface(
            x=X_dom.flatten(),
            y=Y_dom.flatten(),
            z=Z_dom.flatten(),
            value=sdf_vals.flatten(),
            isomin=0,
            isomax=0,
            surface_count=1,
            colorscale=[[0, 'grey'], [1, 'grey']],
            showscale=False,
            opacity=0.1,
            name='Domain Boundary'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        # Save
        output_path = os.path.join(self.save_dir, f"{save_name}.html")
        plotly_plot(fig, filename=output_path, auto_open=False)
        print(f"Volumetric rendering with domain boundary saved to {output_path}")
        
        return fig

    def plot_surface_heatmap(self, solution_data: Dict, domain: Domain3D,
                             title: str = "Surface Temperature Heatmap",
                             save_name: str = "surface_heatmap"):
        """Plot temperature heatmap on the 3D domain surface"""
        
        if not PLOTLY_AVAILABLE or not SKIMAGE_AVAILABLE:
            print("Plotly or Scikit-image not available for surface heatmap.")
            return None

        # Generate a high-resolution grid for surface extraction
        grid_data = domain.generate_grid(64, 64, 64)
        X_dom, Y_dom, Z_dom = grid_data['X'], grid_data['Y'], grid_data['Z']
        sdf_vals = grid_data['sdf']

        # Use marching cubes to find the surface mesh
        try:
            verts, faces, _, _ = measure.marching_cubes(sdf_vals, level=0)
            # Scale vertices to the domain bounds
            bounds = domain.bounds()
            verts[:, 0] = verts[:, 0] * (bounds[0][1] - bounds[0][0]) / (64 - 1) + bounds[0][0]
            verts[:, 1] = verts[:, 1] * (bounds[1][1] - bounds[1][0]) / (64 - 1) + bounds[1][0]
            verts[:, 2] = verts[:, 2] * (bounds[2][1] - bounds[2][0]) / (64 - 1) + bounds[2][0]
        except (ValueError, RuntimeError) as e:
            print(f"Marching cubes failed: {e}. Cannot generate surface plot.")
            return None

        if len(verts) == 0:
            print("Marching cubes did not find a surface.")
            return None

        # Interpolate the solution onto the surface vertices
        u_sol = solution_data['u']
        sol_grid_points = np.array([solution_data['X'].flatten(), solution_data['Y'].flatten(), solution_data['Z'].flatten()]).T
        sol_values = u_sol.flatten()
        
        # Filter out NaN values from solution for interpolation
        valid_indices = ~np.isnan(sol_values)
        if not np.any(valid_indices):
            print("No valid solution data to interpolate.")
            return None
            
        interpolated_temps = griddata(sol_grid_points[valid_indices], sol_values[valid_indices], verts, method='linear')

        # Handle vertices where interpolation failed (e.g., outside the convex hull)
        if np.isnan(interpolated_temps).any():
            # Fallback to nearest neighbor for NaNs
            nan_indices = np.isnan(interpolated_temps)
            nearest_temps = griddata(sol_grid_points[valid_indices], sol_values[valid_indices], verts[nan_indices], method='nearest')
            interpolated_temps[nan_indices] = nearest_temps

        if np.isnan(interpolated_temps).any():
            print("Could not interpolate temperatures for all surface vertices. Filling with 0.")
            interpolated_temps = np.nan_to_num(interpolated_temps, nan=0.0)

        # Create Plotly Mesh3d plot
        fig = go.Figure(data=[go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=interpolated_temps,
            colorscale='Jet',
            colorbar_title='Temperature',
            name='Surface Heatmap'
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )

        # Save
        output_path = os.path.join(self.save_dir, f"{save_name}.html")
        plotly_plot(fig, filename=output_path, auto_open=False)
        print(f"Surface heatmap saved to {output_path}")

        return fig
    
    def create_comparison_plot(self, pinn_data: Dict, numerical_data: Dict, 
                              time_idx: int = -1, save_name: str = "comparison"):
        """Create side-by-side comparison of PINN vs numerical solution"""
        
        # Extract PINN data
        pinn_u = pinn_data['u']
        X = pinn_data['X']
        Y = pinn_data['Y'] 
        Z = pinn_data['Z']
        
        # Extract numerical data and convert to grid
        num_u = numerical_data['solutions'][time_idx]
        if len(num_u.shape) == 1:
            num_u_grid = np.full(X.shape, np.nan)
            num_u_grid[numerical_data['interior_indices']] = num_u
            num_u = num_u_grid
        
        # Compute error where both solutions exist
        mask_valid = ~(np.isnan(pinn_u) | np.isnan(num_u))
        error = np.full_like(pinn_u, np.nan)
        error[mask_valid] = np.abs(pinn_u[mask_valid] - num_u[mask_valid])
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PINN vs Numerical Solution Comparison', fontsize=16)
        
        # Middle slice indices
        nx, ny, nz = pinn_u.shape
        mid_x, mid_y, mid_z = nx//2, ny//2, nz//2
        
        # PINN solution slices
        im1 = axes[0,0].contourf(X[mid_z], Y[mid_z], pinn_u[mid_z], 
                                levels=20, cmap=self.heat_cmap)
        axes[0,0].set_title('PINN - XY Slice')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].contourf(X[:,mid_y,:], Z[:,mid_y,:], pinn_u[:,mid_y,:],
                                levels=20, cmap=self.heat_cmap)
        axes[0,1].set_title('PINN - XZ Slice')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0,1])
        
        im3 = axes[0,2].contourf(Y[mid_x], Z[mid_x], pinn_u[mid_x],
                                levels=20, cmap=self.heat_cmap)
        axes[0,2].set_title('PINN - YZ Slice')
        axes[0,2].set_xlabel('Y')
        axes[0,2].set_ylabel('Z')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Numerical solution slices
        im4 = axes[1,0].contourf(X[mid_z], Y[mid_z], num_u[mid_z],
                                levels=20, cmap=self.heat_cmap)
        axes[1,0].set_title('Numerical - XY Slice')
        axes[1,0].set_xlabel('X')
        axes[1,0].set_ylabel('Y')
        plt.colorbar(im4, ax=axes[1,0])
        
        im5 = axes[1,1].contourf(X[:,mid_y,:], Z[:,mid_y,:], num_u[:,mid_y,:],
                                levels=20, cmap=self.heat_cmap)
        axes[1,1].set_title('Numerical - XZ Slice')
        axes[1,1].set_xlabel('X')
        axes[1,1].set_ylabel('Z')
        plt.colorbar(im5, ax=axes[1,1])
        
        im6 = axes[1,2].contourf(Y[mid_x], Z[mid_x], num_u[mid_x],
                                levels=20, cmap=self.heat_cmap)
        axes[1,2].set_title('Numerical - YZ Slice')
        axes[1,2].set_xlabel('Y')
        axes[1,2].set_ylabel('Z')
        plt.colorbar(im6, ax=axes[1,2])
        
        plt.tight_layout()
        
        # Save comparison
        output_path = os.path.join(self.save_dir, f"{save_name}_solutions.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Create error analysis plot
        self._plot_error_analysis(error, X, Y, Z, save_name)
        
        print(f"Comparison plots saved with prefix {save_name}")
        
        return fig
    
    def _plot_error_analysis(self, error: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                           save_name: str):
        """Plot detailed error analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Error Analysis: |PINN - Numerical|', fontsize=16)
        
        # Middle slice indices
        nx, ny, nz = error.shape
        mid_x, mid_y, mid_z = nx//2, ny//2, nz//2
        
        # Error slices
        im1 = axes[0,0].contourf(X[mid_z], Y[mid_z], error[mid_z],
                                levels=20, cmap=self.error_cmap)
        axes[0,0].set_title(f'Error - XY Slice (Z = {Z[mid_z,0,mid_z]:.2f})')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0,0])
        
        im2 = axes[0,1].contourf(X[:,mid_y,:], Z[:,mid_y,:], error[:,mid_y,:],
                                levels=20, cmap=self.error_cmap)
        axes[0,1].set_title(f'Error - XZ Slice (Y = {Y[0,mid_y,0]:.2f})')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Error histogram
        error_valid = error[~np.isnan(error)]
        if len(error_valid) > 0:
            axes[1,0].hist(error_valid, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[1,0].set_xlabel('Absolute Error')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Error Distribution')
            axes[1,0].grid(True, alpha=0.3)
            
            # Error statistics
            mean_error = np.mean(error_valid)
            max_error = np.max(error_valid)
            std_error = np.std(error_valid)
            
            axes[1,0].axvline(mean_error, color='blue', linestyle='--', 
                             label=f'Mean: {mean_error:.4f}')
            axes[1,0].axvline(max_error, color='red', linestyle='--',
                             label=f'Max: {max_error:.4f}')
            axes[1,0].legend()
        
        # Error statistics text
        if len(error_valid) > 0:
            stats_text = f"""Error Statistics:
Mean Error: {np.mean(error_valid):.4e}
Max Error: {np.max(error_valid):.4e}
Std Error: {np.std(error_valid):.4e}
L2 Error: {np.sqrt(np.mean(error_valid**2)):.4e}
Valid Points: {len(error_valid)}"""
            
            axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes,
                          fontsize=12, verticalalignment='top', 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            axes[1,1].set_title('Error Statistics')
            axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Save error plot
        output_path = os.path.join(self.save_dir, f"{save_name}_error.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Error analysis saved to {output_path}")
    
    def create_animation(self, solution_data: Dict, fps: int = 10,
                        title: str = "Heat Diffusion Animation",
                        save_name: str = "animation"):
        """Create animated visualization of heat diffusion over time"""
        
        times = solution_data['times']
        solutions = solution_data['solutions']
        
        if len(solutions) < 2:
            print("Need at least 2 time points for animation")
            return None
        
        # Get grid data
        X = solution_data.get('X')
        Y = solution_data.get('Y')
        Z = solution_data.get('Z')
        
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
        
        mid_x, mid_y, mid_z = nx//2, ny//2, nz//2
        
        # Initialize plots
        def init_frame():
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
                u_grid = np.full(X.shape, np.nan)
                if 'interior_indices' in solution_data:
                    u_grid[solution_data['interior_indices']] = u
                u = u_grid
            
            # Determine global colormap range
            u_min = np.nanmin([np.nanmin(sol) for sol in solutions if len(sol) > 0])
            u_max = np.nanmax([np.nanmax(sol) for sol in solutions if len(sol) > 0])
            
            # XY slice
            im1 = axes[0,0].contourf(X[mid_z], Y[mid_z], u[mid_z],
                                    levels=20, cmap=self.heat_cmap, 
                                    vmin=u_min, vmax=u_max)
            axes[0,0].set_title(f'XY Slice - t = {times[frame]:.3f}')
            axes[0,0].set_xlabel('X')
            axes[0,0].set_ylabel('Y')
            
            # XZ slice
            im2 = axes[0,1].contourf(X[:,mid_y,:], Z[:,mid_y,:], u[:,mid_y,:],
                                    levels=20, cmap=self.heat_cmap,
                                    vmin=u_min, vmax=u_max)
            axes[0,1].set_title(f'XZ Slice - t = {times[frame]:.3f}')
            axes[0,1].set_xlabel('X')
            axes[0,1].set_ylabel('Z')
            
            # Temperature evolution plot
            max_temps = [np.nanmax(sol) for sol in solutions[:frame+1]]
            axes[1,0].plot(times[:frame+1], max_temps, 'r-', linewidth=2)
            axes[1,0].set_xlabel('Time')
            axes[1,0].set_ylabel('Max Temperature')
            axes[1,0].set_title('Temperature Evolution')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_xlim(times[0], times[-1])
            axes[1,0].set_ylim(0, u_max * 1.1)
            
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
        anim = FuncAnimation(fig, animate, init_func=init_frame,
                           frames=len(solutions), interval=1000//fps, blit=False)
        
        # Save as GIF
        gif_path = os.path.join(self.save_dir, f"{save_name}.gif")
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=100)
        print(f"Animation saved to {gif_path}")
        
        plt.show()
        return anim
    
    def plot_heat_sources(self, heat_sources: List[Dict], domain: Domain3D,
                         title: str = "Heat Source Configuration",
                         save_name: str = "heat_sources"):
        """Visualize heat source locations and domain"""
        
        fig = plt.figure(figsize=(12, 8))
        
        # 3D plot
        ax3d = fig.add_subplot(121, projection='3d')
        
        # Plot domain boundary (sample points)
        grid_data = domain.generate_grid(32, 32, 32)
        X, Y, Z = grid_data['X'], grid_data['Y'], grid_data['Z']
        mask = grid_data['mask']
        
        # Plot domain boundary
        boundary_points = []
        for i in range(0, X.shape[0], 4):
            for j in range(0, X.shape[1], 4):
                for k in range(0, X.shape[2], 4):
                    if mask[i,j,k]:
                        boundary_points.append([X[i,j,k], Y[i,j,k], Z[i,j,k]])
        
        if boundary_points:
            boundary_points = np.array(boundary_points)
            ax3d.scatter(boundary_points[:,0], boundary_points[:,1], boundary_points[:,2],
                        c='lightblue', alpha=0.1, s=1)
        
        # Plot heat sources
        colors = ['red', 'orange', 'yellow', 'purple', 'green']
        for i, source in enumerate(heat_sources):
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
            
            x_sphere = x0 + r * np.outer(np.cos(theta), np.sin(phi))
            y_sphere = y0 + r * np.outer(np.sin(theta), np.sin(phi))
            z_sphere = z0 + r * np.outer(np.ones(np.size(theta)), np.cos(phi))
            
            ax3d.plot_wireframe(x_sphere, y_sphere, z_sphere, 
                               alpha=0.3, color=colors[i % len(colors)])
        
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title('3D Heat Sources')
        ax3d.legend()
        
        # 2D projection (XY plane)
        ax2d = fig.add_subplot(122)
        
        # Plot domain outline in XY
        x_bounds, y_bounds, z_bounds = domain.bounds()
        
        # Sample domain boundary in XY plane
        x_test = np.linspace(x_bounds[0], x_bounds[1], 100)
        y_test = np.linspace(y_bounds[0], y_bounds[1], 100)
        X_test, Y_test = np.meshgrid(x_test, y_test)
        Z_test = np.full_like(X_test, (z_bounds[0] + z_bounds[1]) / 2)
        
        inside_mask = domain.is_inside(X_test, Y_test, Z_test)
        ax2d.contour(X_test, Y_test, inside_mask.astype(float), levels=[0.5], colors='black')
        
        # Plot heat sources
        for i, source in enumerate(heat_sources):
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
        output_path = os.path.join(self.save_dir, f"{save_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Heat source visualization saved to {output_path}")
        
        return fig

def visualize_training_results(model_path: str, numerical_path: Optional[str] = None,
                             t_eval: float = 1.0, device: str = 'cpu'):
    """Complete visualization pipeline for trained model"""
    
    print(f"\n=== Visualization Pipeline ===")
    print(f"Model: {model_path}")
    if numerical_path:
        print(f"Numerical: {numerical_path}")
    print(f"Evaluation time: {t_eval}")
    
    # Load trained model
    model, checkpoint = load_trained_model(model_path, device)
    domain_name = checkpoint['domain_name']
    heat_sources = checkpoint['heat_sources']
    
    print(f"Loaded model for domain: {domain_name}")
    print(f"Heat sources: {len(heat_sources)}")
    
    # Create domain
    domain = DomainFactory.create_domain(domain_name.lower())
    
    # Initialize visualizer
    viz = HeatVisualization3D()
    
    # Plot heat source configuration
    viz.plot_heat_sources(heat_sources, domain, save_name=f"sources_{domain_name}")
    
    # Predict PINN solution
    print("Predicting PINN solution...")
    pinn_solution = predict_solution_3d(model, domain, t_eval, nx=48, ny=48, nz=48, device=device)
    
    # Create PINN visualizations
    print("Creating PINN visualizations...")
    viz.plot_3d_isosurfaces(pinn_solution, title=f"PINN Solution - {domain_name}", 
                           save_name=f"pinn_iso_{domain_name}")
    
    viz._plot_matplotlib_slices(pinn_solution, title=f"PINN Solution - {domain_name}",
                               save_name=f"pinn_slices_{domain_name}")
    
    # Load and visualize numerical solution if available
    if numerical_path and os.path.exists(numerical_path):
        print("Loading numerical solution...")
        with open(numerical_path, 'rb') as f:
            numerical_data = pickle.load(f)
        
        # Find closest time point
        time_idx = np.argmin(np.abs(numerical_data['times'] - t_eval))
        actual_time = numerical_data['times'][time_idx]
        print(f"Using numerical solution at t = {actual_time:.4f}")
        
        # Create comparison
        print("Creating comparison plots...")
        viz.create_comparison_plot(pinn_solution, numerical_data, time_idx, 
                                  save_name=f"comparison_{domain_name}")
        
        # Create animation if multiple time points
        if len(numerical_data['times']) > 5:
            print("Creating animation...")
            viz.create_animation(numerical_data, fps=5, 
                               title=f"Heat Diffusion - {domain_name}",
                               save_name=f"animation_{domain_name}")
    
    print("Visualization complete!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Heat Diffusion Visualization")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--numerical_path', type=str, help='Path to numerical solution')
    parser.add_argument('--t_eval', type=float, default=1.0, help='Evaluation time')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference')
    
    args = parser.parse_args()
    
    visualize_training_results(args.model_path, args.numerical_path, args.t_eval, args.device)