#!/usr/bin/env python3
"""
Interactive 3D Heat Diffusion Demo
Real-time PINN vs Numerical solver comparison with GUI controls
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. PINN functionality disabled.")

from domain_shapes import Domain3D, DomainFactory
from visualization import HeatVisualization3D

if TORCH_AVAILABLE:
    from delta_pinn_3d import DeltaPINN3D, train_delta_pinn_3d, predict_solution_3d
    from numerical_solution import HeatSolver3D, solve_reference_problem

class InteractiveHeatDemo:
    """Interactive 3D heat diffusion demonstration"""
    
    def __init__(self):
        self.heat_sources = []
        self.domain_type = 'sphere'
        self.domain = DomainFactory.create_domain(self.domain_type)
        self.alpha = 0.01
        self.t_current = 0.0
        self.t_max = 1.0
        self.is_running = False
        self.is_training = False
        
        # Solution data
        self.pinn_model = None
        self.numerical_data = None
        self.current_pinn_solution = None
        
        # Visualization
        self.viz = HeatVisualization3D('./demo_output')
        
        # Threading
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("3D Heat Diffusion Interactive Demo")
        self.root.geometry("1200x900")
        self.root.resizable(True, True)
        
        # Create main frames
        control_container = ttk.Frame(self.root, width=300)
        control_container.pack(side='left', fill='y', padx=10, pady=10)

        canvas = tk.Canvas(control_container)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            if event.num == 5 or event.delta == -120:
                canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta == 120:
                canvas.yview_scroll(-1, "units")

        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", _on_mousewheel)
        self.root.bind_all("<Button-5>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        viz_frame = ttk.Frame(self.root)
        viz_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.setup_controls(scrollable_frame)
        self.setup_visualization(viz_frame)
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self.background_worker, daemon=True)
        self.worker_thread.start()
        
        # Start GUI update timer
        self.root.after(100, self.update_gui)
    
    def setup_controls(self, parent):
        """Setup control panel"""
        
        # Title
        title_label = ttk.Label(parent, text="3D Heat Diffusion Demo", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Domain selection
        domain_frame = ttk.LabelFrame(parent, text="Domain Configuration")
        domain_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(domain_frame, text="Domain Shape:").pack(anchor='w')
        self.domain_var = tk.StringVar(value=self.domain_type)
        domain_combo = ttk.Combobox(domain_frame, textvariable=self.domain_var,
                                   values=['cube', 'sphere', 'lshape', 'torus', 'cylinder_holes'],
                                   state='readonly')
        domain_combo.pack(fill='x', pady=(0, 5))
        domain_combo.bind('<<ComboboxSelected>>', self.on_domain_change)
        
        # Heat source controls
        source_frame = ttk.LabelFrame(parent, text="Heat Sources")
        source_frame.pack(fill='x', pady=(0, 10))
        
        # Add source button
        # add_source_btn = ttk.Button(source_frame, text="Add Random Source", 
        #                            command=self.add_random_source)
        # add_source_btn.pack(fill='x', pady=(0, 5))
        
        # Manual source entry
        manual_frame = ttk.Frame(source_frame)
        manual_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(manual_frame, text="Position (x,y,z):").pack(anchor='w')
        pos_frame = ttk.Frame(manual_frame)
        pos_frame.pack(fill='x')
        
        self.x_entry = ttk.Entry(pos_frame, width=8)
        self.x_entry.pack(side='left', padx=(0, 2))
        self.x_entry.insert(0, "0.5")
        
        self.y_entry = ttk.Entry(pos_frame, width=8)
        self.y_entry.pack(side='left', padx=2)
        self.y_entry.insert(0, "0.5")
        
        self.z_entry = ttk.Entry(pos_frame, width=8)
        self.z_entry.pack(side='left', padx=(2, 0))
        self.z_entry.insert(0, "0.5")
        
        ttk.Label(manual_frame, text="Amplitude:").pack(anchor='w', pady=(5, 0))
        self.amp_entry = ttk.Entry(manual_frame, width=10)
        self.amp_entry.pack(fill='x')
        self.amp_entry.insert(0, "1.0")
        
        add_manual_btn = ttk.Button(manual_frame, text="Add Manual Source", 
                                   command=self.add_manual_source)
        add_manual_btn.pack(fill='x', pady=(5, 0))
        
        # Source list
        ttk.Label(source_frame, text="Current Sources:").pack(anchor='w', pady=(10, 0))
        self.source_listbox = tk.Listbox(source_frame, height=4)
        self.source_listbox.pack(fill='x', pady=(0, 5))
        
        # Remove source button
        remove_source_btn = ttk.Button(source_frame, text="Remove Selected", 
                                      command=self.remove_source)
        remove_source_btn.pack(fill='x', pady=(0, 5))
        
        clear_sources_btn = ttk.Button(source_frame, text="Clear All Sources", 
                                      command=self.clear_sources)
        clear_sources_btn.pack(fill='x')
        
        # Simulation parameters
        sim_frame = ttk.LabelFrame(parent, text="Simulation Parameters")
        sim_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(sim_frame, text="Thermal Diffusivity (α):").pack(anchor='w')
        self.alpha_var = tk.DoubleVar(value=self.alpha)
        alpha_scale = ttk.Scale(sim_frame, from_=0.001, to=0.1, variable=self.alpha_var,
                               orient='horizontal')
        alpha_scale.pack(fill='x', pady=(0, 5))
        alpha_scale.bind('<ButtonRelease-1>', self.on_alpha_change)
        
        self.alpha_label = ttk.Label(sim_frame, text=f"α = {self.alpha:.3f}")
        self.alpha_label.pack(anchor='w')
        
        ttk.Label(sim_frame, text="Max Time:").pack(anchor='w', pady=(5, 0))
        self.t_max_var = tk.DoubleVar(value=self.t_max)
        t_max_scale = ttk.Scale(sim_frame, from_=0.1, to=5.0, variable=self.t_max_var,
                               orient='horizontal')
        t_max_scale.pack(fill='x', pady=(0, 5))
        t_max_scale.bind('<ButtonRelease-1>', self.on_t_max_change)
        
        self.t_max_label = ttk.Label(sim_frame, text=f"t_max = {self.t_max:.1f}")
        self.t_max_label.pack(anchor='w')

        ttk.Label(sim_frame, text="Surface Smoothing:").pack(anchor='w', pady=(5, 0))
        self.smoothing_var = tk.DoubleVar(value=1.0)
        smoothing_scale = ttk.Scale(sim_frame, from_=1, to=100, variable=self.smoothing_var,
                               orient='horizontal')
        smoothing_scale.pack(fill='x', pady=(0, 5))
        self.smoothing_label = ttk.Label(sim_frame, text=f"Smoothing = {self.smoothing_var.get():.1f}")
        smoothing_scale.bind('<Motion>', self.on_smoothing_change)
        self.smoothing_label.pack(anchor='w')
        
        # Time control
        time_frame = ttk.LabelFrame(parent, text="Time Control")
        time_frame.pack(fill='x', pady=(0, 10))
        
        self.t_var = tk.DoubleVar(value=self.t_current)
        self.time_scale = ttk.Scale(time_frame, from_=0.0, to=self.t_max, 
                                   variable=self.t_var, orient='horizontal')
        self.time_scale.pack(fill='x', pady=(0, 5))
        self.time_scale.bind('<Motion>', self.on_time_change)
        
        self.time_label = ttk.Label(time_frame, text=f"t = {self.t_current:.3f}")
        self.time_label.pack(anchor='w')
        
        # Control buttons
        button_frame = ttk.LabelFrame(parent, text="Simulation Control")
        button_frame.pack(fill='x', pady=(0, 10))
        
        self.train_btn = ttk.Button(button_frame, text="Train PINN", 
                                   command=self.start_training)
        self.train_btn.pack(fill='x', pady=(0, 5))
        
        self.solve_btn = ttk.Button(button_frame, text="Solve Numerical", 
                                   command=self.start_numerical_solve)
        self.solve_btn.pack(fill='x', pady=(0, 5))
        
        self.compare_btn = ttk.Button(button_frame, text="Compare Solutions", 
                                     command=self.compare_solutions, state='disabled')
        self.compare_btn.pack(fill='x', pady=(0, 5))
        
        self.animate_btn = ttk.Button(button_frame, text="Animate", 
                                     command=self.start_animation, state='disabled')
        self.animate_btn.pack(fill='x', pady=(0, 5))
        
        self.show_3d_btn = ttk.Button(button_frame, text="Show 3D Plot",
                                     command=self.show_3d_plot, state='disabled')
        self.show_3d_btn.pack(fill='x', pady=(0, 5))

        self.show_surface_3d_btn = ttk.Button(button_frame, text="Show Surface 3D Plot",
                                             command=self.show_surface_3d_plot, state='disabled')
        self.show_surface_3d_btn.pack(fill='x', pady=(0, 5))
        
        # Export buttons
        export_frame = ttk.LabelFrame(parent, text="Export")
        export_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(export_frame, text="Save Configuration", 
                  command=self.save_config).pack(fill='x', pady=(0, 2))
        ttk.Button(export_frame, text="Load Configuration", 
                  command=self.load_config).pack(fill='x', pady=(0, 2))
        ttk.Button(export_frame, text="Export Visualization", 
                  command=self.export_visualization).pack(fill='x')
        
        # Status
        status_frame = ttk.LabelFrame(parent, text="Status")
        status_frame.pack(fill='both', expand=True)
        
        self.status_text = tk.Text(status_frame, height=8, wrap='word')
        status_scroll = ttk.Scrollbar(status_frame, orient='vertical', command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        self.status_text.pack(side='left', fill='both', expand=True)
        status_scroll.pack(side='right', fill='y')
        
        self.log_message("Interactive Heat Diffusion Demo initialized.")
        if not TORCH_AVAILABLE:
            self.log_message("WARNING: PyTorch not available. PINN functionality disabled.")
    
    def setup_visualization(self, parent):
        """Setup visualization panel with matplotlib"""
        
        # Create matplotlib figure
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        
        # Initialize plots
        self.update_visualization()
        
        # Mouse interaction for adding sources
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
    
    def on_canvas_click(self, event):
        """Handle mouse clicks on visualization for adding heat sources"""
        if event.inaxes and event.button == 1 and event.dblclick:  # Double-click
            # Get click coordinates
            x_click, y_click = event.xdata, event.ydata
            
            # Estimate z coordinate (middle of domain)
            bounds = self.domain.bounds()
            z_click = (bounds[2][0] + bounds[2][1]) / 2
            
            # Check if point is inside domain
            if self.domain.is_inside(np.array([x_click]), np.array([y_click]), np.array([z_click]))[0]:
                # Add heat source
                self.heat_sources.append({
                    'position': (x_click, y_click, z_click),
                    'amplitude': 1.0,
                    'sigma': 0.05
                })
                
                self.update_source_list()
                self.update_visualization()
                self.log_message(f"Added heat source at ({x_click:.3f}, {y_click:.3f}, {z_click:.3f})")
            else:
                self.log_message("Cannot add heat source outside domain.")
    
    def on_domain_change(self, event=None):
        """Handle domain type change"""
        new_domain = self.domain_var.get()
        if new_domain != self.domain_type:
            self.domain_type = new_domain
            self.domain = DomainFactory.create_domain(self.domain_type)
            self.log_message(f"Changed domain to: {self.domain_type}")
            
            # Clear solutions
            self.pinn_model = None
            self.numerical_data = None
            self.compare_btn.config(state='disabled')
            self.animate_btn.config(state='disabled')
            
            self.update_visualization()
    
    def on_alpha_change(self, event=None):
        """Handle thermal diffusivity change"""
        self.alpha = self.alpha_var.get()
        self.alpha_label.config(text=f"α = {self.alpha:.3f}")
        
        # Invalidate numerical solution
        self.numerical_data = None
        self.compare_btn.config(state='disabled')
        self.animate_btn.config(state='disabled')
    
    def on_t_max_change(self, event=None):
        """Handle max time change"""
        self.t_max = self.t_max_var.get()
        self.t_max_label.config(text=f"t_max = {self.t_max:.1f}")
        self.time_scale.config(to=self.t_max)
        
        # Invalidate solutions
        self.numerical_data = None
        self.compare_btn.config(state='disabled')
        self.animate_btn.config(state='disabled')
    
    def on_time_change(self, event=None):
        """Handle time slider change"""
        self.t_current = self.t_var.get()
        self.time_label.config(text=f"t = {self.t_current:.3f}")
        
        # Update visualization if solutions exist
        if self.pinn_model or self.numerical_data:
            self.update_solution_visualization()

    def on_smoothing_change(self, event=None):
        """Handle smoothing slider change"""
        self.smoothing_label.config(text=f"Smoothing = {self.smoothing_var.get():.1f}")
    
    def add_random_source(self):
        """Add randomly positioned heat source"""
        bounds = self.domain.bounds()
        
        # Generate random position within domain bounds (with margin)
        x_pos = np.random.uniform(bounds[0][0] + 0.1, bounds[0][1] - 0.1)
        y_pos = np.random.uniform(bounds[1][0] + 0.1, bounds[1][1] - 0.1)
        z_pos = np.random.uniform(bounds[2][0] + 0.1, bounds[2][1] - 0.1)
        
        # Verify position is inside domain
        attempts = 0
        while not self.domain.is_inside(np.array([x_pos]), np.array([y_pos]), np.array([z_pos]))[0] and attempts < 20:
            x_pos = np.random.uniform(bounds[0][0] + 0.1, bounds[0][1] - 0.1)
            y_pos = np.random.uniform(bounds[1][0] + 0.1, bounds[1][1] - 0.1)
            z_pos = np.random.uniform(bounds[2][0] + 0.1, bounds[2][1] - 0.1)
            attempts += 1
        
        if attempts >= 20:
            self.log_message("Could not find valid position for random source.")
            return
        
        self.heat_sources.append({
            'position': (x_pos, y_pos, z_pos),
            'amplitude': np.random.uniform(0.5, 2.0),
            'sigma': np.random.uniform(0.03, 0.08)
        })
        
        self.update_source_list()
        self.update_visualization()
        self.log_message(f"Added random heat source at ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f})")
    
    def add_manual_source(self):
        """Add manually specified heat source"""
        try:
            x_pos = float(self.x_entry.get())
            y_pos = float(self.y_entry.get())
            z_pos = float(self.z_entry.get())
            amplitude = float(self.amp_entry.get())
            
            # Check if position is inside domain
            if self.domain.is_inside(np.array([x_pos]), np.array([y_pos]), np.array([z_pos]))[0]:
                self.heat_sources.append({
                    'position': (x_pos, y_pos, z_pos),
                    'amplitude': amplitude,
                    'sigma': 0.05
                })
                
                self.update_source_list()
                self.update_visualization()
                self.log_message(f"Added manual heat source at ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f})")
            else:
                messagebox.showerror("Error", "Position is outside the selected domain.")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
    
    def remove_source(self):
        """Remove selected heat source"""
        selection = self.source_listbox.curselection()
        if selection:
            idx = selection[0]
            removed_source = self.heat_sources.pop(idx)
            self.update_source_list()
            self.update_visualization()
            pos = removed_source['position']
            self.log_message(f"Removed heat source at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    def clear_sources(self):
        """Clear all heat sources"""
        if self.heat_sources:
            self.heat_sources.clear()
            self.update_source_list()
            self.update_visualization()
            self.log_message("Cleared all heat sources.")
    
    def update_source_list(self):
        """Update the source listbox"""
        self.source_listbox.delete(0, tk.END)
        for i, source in enumerate(self.heat_sources):
            pos = source['position']
            amp = source['amplitude']
            self.source_listbox.insert(tk.END, f"Source {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) A={amp:.2f}")
    
    def update_visualization(self):
        """Update the main visualization"""
        self.fig.clear()
        
        if not self.heat_sources:
            # Show empty domain
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Add heat sources to begin simulation\nDouble-click to add source', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Domain: {self.domain_type}")
            
        else:
            # Show domain with heat sources
            ax = self.fig.add_subplot(111)
            
            # Plot domain boundary (2D projection)
            bounds = self.domain.bounds()
            x_test = np.linspace(bounds[0][0], bounds[0][1], 100)
            y_test = np.linspace(bounds[1][0], bounds[1][1], 100)
            X_test, Y_test = np.meshgrid(x_test, y_test)
            Z_test = np.full_like(X_test, (bounds[2][0] + bounds[2][1]) / 2)
            
            inside_mask = self.domain.is_inside(X_test, Y_test, Z_test)
            ax.contour(X_test, Y_test, inside_mask.astype(float), levels=[0.5], colors='black', linewidths=2)
            ax.contourf(X_test, Y_test, inside_mask.astype(float), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.3)
            
            # Plot heat sources
            colors = ['red', 'orange', 'yellow', 'purple', 'green', 'cyan', 'magenta']
            for i, source in enumerate(self.heat_sources):
                x0, y0, z0 = source['position']
                amplitude = source['amplitude']
                sigma = source.get('sigma', 0.05)
                
                ax.scatter([x0], [y0], c=colors[i % len(colors)], 
                          s=200 * amplitude, alpha=0.8, 
                          label=f'Source {i+1}')
                
                # Influence circle
                circle = patches.Circle((x0, y0), 3*sigma, fill=False, 
                                      color=colors[i % len(colors)], alpha=0.5, linestyle='--')
                ax.add_patch(circle)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Heat Sources - {self.domain_type}')
            ax.legend()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def update_solution_visualization(self):
        """Update visualization with current solution"""
        if not (self.pinn_model or self.numerical_data):
            return
        
        self.fig.clear()
        
        # Create subplot layout
        if self.pinn_model and self.numerical_data:
            # Comparison view
            ax1 = self.fig.add_subplot(131)
            ax2 = self.fig.add_subplot(132)
            ax3 = self.fig.add_subplot(133)
            axes = [ax1, ax2, ax3]
            titles = ['PINN Solution', 'Numerical Solution', 'Absolute Error']
        elif self.pinn_model:
            # PINN only
            ax1 = self.fig.add_subplot(111)
            axes = [ax1]
            titles = ['PINN Solution']
        else:
            # Numerical only
            ax1 = self.fig.add_subplot(111)
            axes = [ax1]
            titles = ['Numerical Solution']
        
        # Get solutions at current time
        solutions = []
        
        if self.pinn_model:
            try:
                device = 'cpu'  # Force CPU for GUI responsiveness
                pinn_sol = predict_solution_3d(self.pinn_model, self.domain, self.t_current,
                                             nx=32, ny=32, nz=32, device=device)
                solutions.append(pinn_sol['u'])
            except Exception as e:
                self.log_message(f"Error predicting PINN solution: {e}")
                return
        
        if self.numerical_data:
            # Find closest time
            time_idx = np.argmin(np.abs(self.numerical_data['times'] - self.t_current))
            num_sol = self.numerical_data['solutions'][time_idx]
            
            # Convert to grid
            if len(num_sol.shape) == 1:
                grid_shape = (32, 32, 32)  # Match PINN resolution
                X = np.linspace(0, 1, 32)
                Y = np.linspace(0, 1, 32)
                Z = np.linspace(0, 1, 32)
                XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing='ij')
                
                num_sol_grid = np.full(grid_shape, np.nan)
                # Interpolate or use available data
                if 'interior_indices' in self.numerical_data:
                    # Map to new grid (simplified)
                    interior_idx = self.numerical_data['interior_indices']
                    for i, val in enumerate(num_sol):
                        if i < len(interior_idx[0]):
                            ii, jj, kk = interior_idx[0][i], interior_idx[1][i], interior_idx[2][i]
                            # Scale indices to match new grid
                            scale_x = (grid_shape[0] - 1) / (self.numerical_data['grid_shape'][0] - 1)
                            scale_y = (grid_shape[1] - 1) / (self.numerical_data['grid_shape'][1] - 1)
                            scale_z = (grid_shape[2] - 1) / (self.numerical_data['grid_shape'][2] - 1)
                            
                            new_i = int(ii * scale_x)
                            new_j = int(jj * scale_y)
                            new_k = int(kk * scale_z)
                            
                            if (0 <= new_i < grid_shape[0] and 
                                0 <= new_j < grid_shape[1] and 
                                0 <= new_k < grid_shape[2]):
                                num_sol_grid[new_i, new_j, new_k] = val
                
                solutions.append(num_sol_grid)
            else:
                solutions.append(num_sol)
        
        # Plot solutions
        for i, (ax, title, sol) in enumerate(zip(axes, titles, solutions)):
            if i == 2 and len(solutions) >= 2:  # Error plot
                sol = np.abs(solutions[0] - solutions[1])
                cmap = 'Reds'
            else:
                cmap = 'hot'
            
            # Plot middle XY slice
            mid_z = sol.shape[2] // 2
            slice_data = sol[:, :, mid_z]
            
            # Create coordinate arrays
            x_coords = np.linspace(0, 1, sol.shape[0])
            y_coords = np.linspace(0, 1, sol.shape[1])
            
            im = ax.contourf(x_coords, y_coords, slice_data.T, levels=20, cmap=cmap)
            ax.set_title(f'{title}\nt = {self.t_current:.3f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            
            # Add colorbar
            self.fig.colorbar(im, ax=ax, shrink=0.8)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def start_training(self):
        """Start PINN training in background"""
        if not TORCH_AVAILABLE:
            messagebox.showerror("Error", "PyTorch not available. Cannot train PINN.")
            return
        
        if not self.heat_sources:
            messagebox.showerror("Error", "Please add at least one heat source before training.")
            return
        
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress.")
            return
        
        self.log_message("Starting PINN training...")
        self.train_btn.config(state='disabled', text='Training...')
        self.is_training = True
        
        # Add training task to queue
        self.task_queue.put({
            'type': 'train_pinn',
            'domain_type': self.domain_type,
            'heat_sources': self.heat_sources.copy(),
            'alpha': self.alpha
        })
    
    def start_numerical_solve(self):
        """Start numerical solving in background"""
        if not self.heat_sources:
            messagebox.showerror("Error", "Please add at least one heat source before solving.")
            return
        
        self.log_message("Starting numerical solution...")
        self.solve_btn.config(state='disabled', text='Solving...')
        
        # Add numerical task to queue
        self.task_queue.put({
            'type': 'solve_numerical',
            'domain_type': self.domain_type,
            'heat_sources': self.heat_sources.copy(),
            'alpha': self.alpha,
            't_final': self.t_max
        })
    
    def compare_solutions(self):
        """Compare PINN and numerical solutions"""
        if not (self.pinn_model and self.numerical_data):
            messagebox.showwarning("Warning", "Both PINN and numerical solutions are required for comparison.")
            return
        
        self.log_message("Creating detailed comparison...")
        
        # Use visualization module for detailed comparison
        try:
            # Create temporary PINN solution data
            device = 'cpu'
            pinn_data = predict_solution_3d(self.pinn_model, self.domain, self.t_current,
                                          nx=48, ny=48, nz=48, device=device)
            
            # Find closest numerical time
            time_idx = np.argmin(np.abs(self.numerical_data['times'] - self.t_current))
            
            # Create comparison plot
            self.viz.create_comparison_plot(pinn_data, self.numerical_data, time_idx,
                                          save_name=f"comparison_{self.domain_type}")
            
            self.log_message("Comparison saved to ./demo_output/")
            
        except Exception as e:
            self.log_message(f"Error creating comparison: {e}")
    
    def show_3d_plot(self):
        """Generate and show an interactive 3D plot"""
        if not self.pinn_model:
            messagebox.showwarning("Warning", "A trained PINN model is required for 3D visualization.")
            return

        self.log_message("Generating interactive 3D plot with volumetric rendering...")

        try:
            # Predict solution
            device = 'cpu'
            pinn_data = predict_solution_3d(self.pinn_model, self.domain, self.t_current,
                                          nx=48, ny=48, nz=48, device=device)
            
            u_sol = pinn_data['u']
            self.log_message(f"Predicted solution min: {np.nanmin(u_sol)}, max: {np.nanmax(u_sol)}")

            # Generate plot
            save_name = f"interactive_3d_{self.domain_type}"
            fig = self.viz.plot_volumetric_rendering(pinn_data, self.domain, title=f"PINN 3D Solution - {self.domain_type}",
                                                     save_name=save_name)

            if fig:
                # Open in web browser
                import webbrowser
                import os
                output_path = os.path.join(self.viz.save_dir, f"{save_name}.html")
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                self.log_message(f"3D plot opened in web browser.")
            else:
                self.log_message("Failed to generate 3D plot (Plotly may not be available).")

        except Exception as e:
            self.log_message(f"Error creating 3D plot: {e}")

    def show_surface_3d_plot(self):
        """Generate and show an interactive 3D surface plot"""
        if not self.pinn_model:
            messagebox.showwarning("Warning", "A trained PINN model is required for 3D visualization.")
            return

        self.log_message("Generating interactive 3D surface plot...")

        try:
            # Predict solution
            device = 'cpu'
            pinn_data = predict_solution_3d(self.pinn_model, self.domain, self.t_current,
                                          nx=48, ny=48, nz=48, device=device)
            
            # Get smoothing value
            smoothing_val = self.smoothing_var.get()

            # Generate plot
            save_name = f"interactive_3d_surface_{self.domain_type}"
            fig = self.viz.plot_surface_heatmap(pinn_data, self.domain, title=f"PINN 3D Surface Solution - {self.domain_type}",
                                                  save_name=save_name, smoothing=smoothing_val)

            if fig:
                # Open in web browser
                import webbrowser
                import os
                output_path = os.path.join(self.viz.save_dir, f"{save_name}.html")
                webbrowser.open(f"file://{os.path.abspath(output_path)}")
                self.log_message(f"3D surface plot opened in web browser.")
            else:
                self.log_message("Failed to generate 3D surface plot.")

        except Exception as e:
            self.log_message(f"Error creating 3D surface plot: {e}")

    def start_animation(self):
        """Start animation of time evolution"""
        if not self.numerical_data:
            messagebox.showwarning("Warning", "Numerical solution required for animation.")
            return
        
        self.log_message("Creating animation...")
        
        try:
            self.viz.create_animation(self.numerical_data, fps=5,
                                    title=f"Heat Diffusion - {self.domain_type}",
                                    save_name=f"animation_{self.domain_type}")
            
            self.log_message("Animation saved to ./demo_output/")
            
        except Exception as e:
            self.log_message(f"Error creating animation: {e}")
    
    def background_worker(self):
        """Background worker thread for computationally intensive tasks"""
        while True:
            try:
                task = self.task_queue.get(timeout=1.0)
                
                if task['type'] == 'train_pinn':
                    self.execute_training(task)
                elif task['type'] == 'solve_numerical':
                    self.execute_numerical_solve(task)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.result_queue.put({'type': 'error', 'message': str(e)})
    
    def execute_training(self, task):
        """Execute PINN training"""
        try:
            # Import here to avoid GUI thread issues
            import argparse
            
            # Create minimal args for training
            class Args:
                def __init__(self):
                    self.hidden_size = 64  # Smaller for speed
                    self.num_layers = 4
                    self.use_fourier = True
                    self.fourier_scale = 1.0
                    self.use_residual = True
                    self.epochs = 1000  # Fewer epochs for demo
                    self.lr = 1e-3
                    self.warmup_epochs = 200
                    self.alpha = task['alpha']
                    self.n_pde = 2000  # Smaller for speed
                    self.n_bc = 500
                    self.n_ic = 500
                    self.device = 'cpu'  # Force CPU for stability
                    self.seed = 1337
                    self.save_dir = './demo_output'
                    self.log_interval = 200
            
            args = Args()
            
            # Create domain
            domain = DomainFactory.create_domain(task['domain_type'])
            
            # Train model
            model = train_delta_pinn_3d(args, domain, task['heat_sources'])
            
            self.result_queue.put({
                'type': 'training_complete',
                'model': model
            })
            
        except Exception as e:
            self.result_queue.put({
                'type': 'training_error',
                'message': str(e)
            })
    
    def execute_numerical_solve(self, task):
        """Execute numerical solution"""
        try:
            # Solve numerical problem
            solution = solve_reference_problem(
                domain_type=task['domain_type'],
                heat_sources=task['heat_sources'],
                nx=24,  # Smaller grid for speed
                alpha=task['alpha'],
                t_final=task['t_final'],
                method='implicit',
                save_dir='./demo_output'
            )
            
            self.result_queue.put({
                'type': 'numerical_complete',
                'solution': solution
            })
            
        except Exception as e:
            self.result_queue.put({
                'type': 'numerical_error',
                'message': str(e)
            })
    
    def save_config(self):
        """Save current configuration"""
        import json
        
        config = {
            'domain_type': self.domain_type,
            'heat_sources': self.heat_sources,
            'alpha': self.alpha,
            't_max': self.t_max
        }
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            self.log_message(f"Configuration saved to {filename}")
    
    def load_config(self):
        """Load configuration from file"""
        import json
        
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                # Apply configuration
                self.domain_type = config.get('domain_type', 'sphere')
                self.domain_var.set(self.domain_type)
                self.domain = DomainFactory.create_domain(self.domain_type)
                
                self.heat_sources = config.get('heat_sources', [])
                self.alpha = config.get('alpha', 0.01)
                self.t_max = config.get('t_max', 1.0)
                
                # Update GUI
                self.alpha_var.set(self.alpha)
                self.t_max_var.set(self.t_max)
                self.alpha_label.config(text=f"α = {self.alpha:.3f}")
                self.t_max_label.config(text=f"t_max = {self.t_max:.1f}")
                self.time_scale.config(to=self.t_max)
                
                self.update_source_list()
                self.update_visualization()
                
                self.log_message(f"Configuration loaded from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def export_visualization(self):
        """Export current visualization"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            self.log_message(f"Visualization exported to {filename}")
    
    def update_gui(self):
        """Periodic GUI update from background tasks"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                
                if result['type'] == 'training_complete':
                    self.pinn_model = result['model']
                    self.train_btn.config(state='normal', text='Train PINN')
                    self.is_training = False
                    self.log_message("PINN training completed successfully!")
                    if self.numerical_data:
                        self.compare_btn.config(state='normal')
                        self.animate_btn.config(state='normal')
                    self.show_3d_btn.config(state='normal')
                    self.show_surface_3d_btn.config(state='normal')
                
                elif result['type'] == 'training_error':
                    self.train_btn.config(state='normal', text='Train PINN')
                    self.is_training = False
                    self.log_message(f"PINN training failed: {result['message']}")
                
                elif result['type'] == 'numerical_complete':
                    self.numerical_data = result['solution']
                    self.solve_btn.config(state='normal', text='Solve Numerical')
                    self.log_message("Numerical solution completed successfully!")
                    if self.pinn_model:
                        self.compare_btn.config(state='normal')
                        self.show_3d_btn.config(state='normal')
                        self.show_surface_3d_btn.config(state='normal')
                    self.animate_btn.config(state='normal')
                
                elif result['type'] == 'numerical_error':
                    self.solve_btn.config(state='normal', text='Solve Numerical')
                    self.log_message(f"Numerical solution failed: {result['message']}")
                
                elif result['type'] == 'error':
                    self.log_message(f"Background task error: {result['message']}")
                    
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.update_gui)
    
    def log_message(self, message):
        """Add message to status log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.status_text.insert(tk.END, formatted_message)
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def run(self):
        """Run the interactive demo"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nInteractive demo interrupted by user.")
        except Exception as e:
            print(f"Demo error: {e}")
        finally:
            print("Interactive demo finished.")


def main():
    """Main entry point for interactive demo"""
    import os
    import sys
    
    # Create output directory
    os.makedirs('./demo_output', exist_ok=True)
    
    # Check dependencies
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append('matplotlib')
    
    try:
        import scipy
    except ImportError:
        missing_deps.append('scipy')
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Please install with: pip install torch matplotlib scipy")
        return
    
    print("Starting 3D Heat Diffusion Interactive Demo...")
    print("Features:")
    print("- Multiple domain shapes (cube, sphere, L-shape, torus, cylinder with holes)")
    print("- Click-to-add heat sources")
    print("- Real-time PINN vs Numerical comparison")
    print("- 3D visualization and animation")
    print("- Export capabilities")
    print("\nDouble-click on visualization to add heat sources!")
    
    demo = InteractiveHeatDemo()
    demo.run()


if __name__ == "__main__":
    main()