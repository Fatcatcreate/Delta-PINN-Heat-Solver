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
import os
warnings.filterwarnings('ignore')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. PINN functionality disabled.")

from domainShapes import Domain3D, DomainFactory
from visualisation import HeatVisualisation3D
from convertObj import convertObjToSdf

if TORCH_AVAILABLE:
    from deltaPinn3d import DeltaPINN3D, trainDeltaPinn3d, predictSolution3d, ResidualBlock
    from numericalSolution import HeatSolver3D, solveReferenceProblem

class InteractiveHeatDemo:
    """Interactive 3D heat diffusion demonstration"""
    
    def __init__(self):
        self.heatSources = []
        self.domainType = 'sphere'
        self.domain = DomainFactory.createDomain(self.domainType)
        self.alpha = 0.01
        self.tCurrent = 0.0
        self.tMax = 1.0
        self.isRunning = False
        self.isTraining = False
        
        self.pinnModel = None
        self.numericalData = None
        self.currentPinnSolution = None
        
        self.viz = HeatVisualisation3D('./demo_output')
        
        self.taskQueue = queue.Queue()
        self.resultQueue = queue.Queue()
        
        self.setupGui()

    def getSanitisedDomainName(self):
        """Returns a sanitised version of the domain name suitable for filenames."""
        if self.domainType.endswith('.npy'):
            return os.path.splitext(os.path.basename(self.domainType))[0]
        return self.domainType
    
    def setupGui(self):
        """Setup the main GUI interface"""
        self.root = tk.Tk()
        self.root.title("3D Heat Diffusion Interactive Demo")
        self.root.geometry("1200x900")
        self.root.resizable(True, True)
        
        controlContainer = ttk.Frame(self.root, width=300)
        controlContainer.pack(side='left', fill='y', padx=10, pady=10)

        canvas = tk.Canvas(controlContainer)
        scrollbar = ttk.Scrollbar(controlContainer, orient="vertical", command=canvas.yview)
        scrollableFrame = ttk.Frame(canvas)

        scrollableWindow = canvas.create_window((0, 0), window=scrollableFrame, anchor="nw")

        scrollableFrame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(scrollableWindow, width=e.width)
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        def onMousewheel(event):
            if event.num == 5 or event.delta == -120:
                canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta == 120:
                canvas.yview_scroll(-1, "units")

        self.root.bind_all("<MouseWheel>", onMousewheel)
        self.root.bind_all("<Button-4>", onMousewheel)
        self.root.bind_all("<Button-5>", onMousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        vizFrame = ttk.Frame(self.root)
        vizFrame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        self.setupControls(scrollableFrame)
        self.setupVisualisation(vizFrame)
        
        self.workerThread = threading.Thread(target=self.backgroundWorker, daemon=True)
        self.workerThread.start()
        
        self.root.after(100, self.updateGui)
    
    def setupControls(self, parent):
        """Setup control panel"""
        
        titleLabel = ttk.Label(parent, text="3D Heat Diffusion Demo", font=('Arial', 16, 'bold'))
        titleLabel.pack(pady=(0, 20))
        
        domainFrame = ttk.LabelFrame(parent, text="Domain Configuration")
        domainFrame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(domainFrame, text="Domain Shape:").pack(anchor='w')
        self.domainVar = tk.StringVar(value=self.domainType)
        domainCombo = ttk.Combobox(domainFrame, textvariable=self.domainVar,
                                   values=['cube', 'sphere', 'lshape', 'torus', 'cylinder_holes'],
                                   state='readonly')
        domainCombo.pack(fill='x', pady=(0, 5))
        domainCombo.bind('<<ComboboxSelected>>', self.onDomainChange)

        loadObjBtn = ttk.Button(domainFrame, text="Load .obj File", 
                                command=self.load_obj_file)
        loadObjBtn.pack(fill='x', pady=(5, 0))
        
        ttk.Label(domainFrame, text="Voxelisation Resolution:").pack(anchor='w', pady=(5, 0))
        self.resolutionVar = tk.IntVar(value=128)
        resolutionScale = ttk.Scale(domainFrame, from_=8, to=512, variable=self.resolutionVar,
                               orient='horizontal')
        resolutionScale.pack(fill='x', pady=(0, 5))
        self.resolutionLabel = ttk.Label(domainFrame, text=f"Resolution = {self.resolutionVar.get()}")
        resolutionScale.bind('<Motion>', lambda e: self.resolutionLabel.config(text=f"Resolution = {self.resolutionVar.get()}"))
        self.resolutionLabel.pack(anchor='w')

        sourceFrame = ttk.LabelFrame(parent, text="Heat Sources")
        sourceFrame.pack(fill='x', pady=(0, 10))
        
        manualFrame = ttk.Frame(sourceFrame)
        manualFrame.pack(fill='x', pady=(0, 5))
        
        ttk.Label(manualFrame, text="Position (x,y,z):").pack(anchor='w')
        posFrame = ttk.Frame(manualFrame)
        posFrame.pack(fill='x')
        
        self.xEntry = ttk.Entry(posFrame, width=8)
        self.xEntry.pack(side='left', padx=(0, 2))
        self.xEntry.insert(0, "0.5")
        
        self.yEntry = ttk.Entry(posFrame, width=8)
        self.yEntry.pack(side='left', padx=2)
        self.yEntry.insert(0, "0.5")
        
        self.zEntry = ttk.Entry(posFrame, width=8)
        self.zEntry.pack(side='left', padx=(2, 0))
        self.zEntry.insert(0, "0.5")
        
        ttk.Label(manualFrame, text="Amplitude:").pack(anchor='w', pady=(5, 0))
        self.ampEntry = ttk.Entry(manualFrame, width=10)
        self.ampEntry.pack(fill='x')
        self.ampEntry.insert(0, "1.0")
        
        addManualBtn = ttk.Button(manualFrame, text="Add Manual Source", 
                                   command=self.addManualSource)
        addManualBtn.pack(fill='x', pady=(5, 0))
        
        ttk.Label(sourceFrame, text="Current Sources:").pack(anchor='w', pady=(10, 0))
        self.sourceListbox = tk.Listbox(sourceFrame, height=4)
        self.sourceListbox.pack(fill='x', pady=(0, 5))
        
        removeSourceBtn = ttk.Button(sourceFrame, text="Remove Selected", 
                                      command=self.removeSource)
        removeSourceBtn.pack(fill='x', pady=(0, 5))
        
        clearSourcesBtn = ttk.Button(sourceFrame, text="Clear All Sources", 
                                      command=self.clearSources)
        clearSourcesBtn.pack(fill='x')
        
        simFrame = ttk.LabelFrame(parent, text="Simulation Parameters")
        simFrame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(simFrame, text="Thermal Diffusivity (α):").pack(anchor='w')
        self.alphaVar = tk.DoubleVar(value=self.alpha)
        alphaScale = ttk.Scale(simFrame, from_=0.001, to=0.1, variable=self.alphaVar,
                               orient='horizontal')
        alphaScale.pack(fill='x', pady=(0, 5))
        alphaScale.bind('<ButtonRelease-1>', self.onAlphaChange)
        
        self.alphaLabel = ttk.Label(simFrame, text=f"α = {self.alpha:.3f}")
        self.alphaLabel.pack(anchor='w')
        
        ttk.Label(simFrame, text="Max Time:").pack(anchor='w', pady=(5, 0))
        self.tMaxVar = tk.DoubleVar(value=self.tMax)
        tMaxScale = ttk.Scale(simFrame, from_=0.1, to=5.0, variable=self.tMaxVar,
                               orient='horizontal')
        tMaxScale.pack(fill='x', pady=(0, 5))
        tMaxScale.bind('<ButtonRelease-1>', self.onTMaxChange)
        
        self.tMaxLabel = ttk.Label(simFrame, text=f"t_max = {self.tMax:.1f}")
        self.tMaxLabel.pack(anchor='w')

        ttk.Label(simFrame, text="Surface Smoothing:").pack(anchor='w', pady=(5, 0))
        self.smoothingVar = tk.DoubleVar(value=1.0)
        smoothingScale = ttk.Scale(simFrame, from_=1, to=100, variable=self.smoothingVar,
                               orient='horizontal')
        smoothingScale.pack(fill='x', pady=(0, 5))
        self.smoothingLabel = ttk.Label(simFrame, text=f"Smoothing = {self.smoothingVar.get():.1f}")
        smoothingScale.bind('<Motion>', self.onSmoothingChange)
        self.smoothingLabel.pack(anchor='w')
        
        timeFrame = ttk.LabelFrame(parent, text="Time Control")
        timeFrame.pack(fill='x', pady=(0, 10))
        
        self.tVar = tk.DoubleVar(value=self.tCurrent)
        self.timeScale = ttk.Scale(timeFrame, from_=0.0, to=self.tMax, 
                                   variable=self.tVar, orient='horizontal')
        self.timeScale.pack(fill='x', pady=(0, 5))
        self.timeScale.bind('<Motion>', self.onTimeChange)
        
        self.timeLabel = ttk.Label(timeFrame, text=f"t = {self.tCurrent:.3f}")
        self.timeLabel.pack(anchor='w')
        
        buttonFrame = ttk.LabelFrame(parent, text="Simulation Control")
        buttonFrame.pack(fill='x', pady=(0, 10))
        
        self.trainBtn = ttk.Button(buttonFrame, text="Train PINN", 
                                   command=self.startTraining)
        self.trainBtn.pack(fill='x', pady=(0, 5))

        self.trainFullBtn = ttk.Button(buttonFrame, text="Train PINN (Full Quality)", 
                                   command=self.startTrainingFull)
        self.trainFullBtn.pack(fill='x', pady=(0, 5))
        
        self.solveBtn = ttk.Button(buttonFrame, text="Solve Numerical", 
                                   command=self.startNumericalSolve)
        self.solveBtn.pack(fill='x', pady=(0, 5))
        
        self.compareBtn = ttk.Button(buttonFrame, text="Compare Solutions", 
                                     command=self.compareSolutions, state='disabled')
        self.compareBtn.pack(fill='x', pady=(0, 5))
        
        self.animateBtn = ttk.Button(buttonFrame, text="Animate", 
                                     command=self.startAnimation, state='disabled')
        self.animateBtn.pack(fill='x', pady=(0, 5))
        
        self.show3dBtn = ttk.Button(buttonFrame, text="Show 3D Plot",
                                     command=self.show3dPlot, state='disabled')
        self.show3dBtn.pack(fill='x', pady=(0, 5))

        self.showSurface3dBtn = ttk.Button(buttonFrame, text="Show Surface 3D Plot",
                                             command=self.showSurface3dPlot, state='disabled')
        self.showSurface3dBtn.pack(fill='x', pady=(0, 5))

        manageFrame = ttk.LabelFrame(parent, text="Model Management & Visualisation")
        manageFrame.pack(fill='x', pady=(0, 10))

        self.loadBtn = ttk.Button(manageFrame, text="Load Model",
                                    command=self.loadModel)
        self.loadBtn.pack(fill='x', pady=(0, 5))

        self.saveBtn = ttk.Button(manageFrame, text="Save Current Model",
                                    command=self.saveModel, state='disabled')
        self.saveBtn.pack(fill='x', pady=(0, 5))

        self.genVizBtn = ttk.Button(manageFrame, text="Generate All Visualisations",
                                    command=self.generateAllVisualisations, state='disabled')
        self.genVizBtn.pack(fill='x', pady=(0, 5))
        
        exportFrame = ttk.LabelFrame(parent, text="Export")
        exportFrame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(exportFrame, text="Save Configuration", 
                  command=self.saveConfig).pack(fill='x', pady=(0, 2))
        ttk.Button(exportFrame, text="Load Configuration", 
                  command=self.loadConfig).pack(fill='x', pady=(0, 2))
        ttk.Button(exportFrame, text="Export Visualisation", 
                  command=self.exportVisualisation).pack(fill='x')
        
        statusFrame = ttk.LabelFrame(parent, text="Status")
        statusFrame.pack(fill='both', expand=True)
        
        self.statusText = tk.Text(statusFrame, height=8, wrap='word')
        statusScroll = ttk.Scrollbar(statusFrame, orient='vertical', command=self.statusText.yview)
        self.statusText.configure(yscrollcommand=statusScroll.set)
        
        self.statusText.pack(side='left', fill='both', expand=True)
        statusScroll.pack(side='right', fill='y')
        
        self.progressBar = ttk.Progressbar(statusFrame, orient='horizontal', mode='determinate')
        self.progressBar.pack(side='bottom', fill='x')
        
        self.logMessage("Interactive Heat Diffusion Demo initialised.")
        if not TORCH_AVAILABLE:
            self.logMessage("WARNING: PyTorch not available. PINN functionality disabled.")
    
    def setupVisualisation(self, parent):
        """Setup visualisation panel with matplotlib"""
        
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        
        self.updateVisualisation()
        
        self.canvas.mpl_connect('button_press_event', self.onCanvasClick)
    
    def onCanvasClick(self, event):
        """Handle mouse clicks on visualisation for adding heat sources"""
        if event.inaxes and event.button == 1 and event.dblclick:
            xClick, yClick = event.xdata, event.ydata
            
            bounds = self.domain.bounds()
            zClick = (bounds[2][0] + bounds[2][1]) / 2
            
            if self.domain.isInside(np.array([xClick]), np.array([yClick]), np.array([zClick]))[0]:
                self.heatSources.append({
                    'position': (xClick, yClick, zClick),
                    'amplitude': 1.0,
                    'sigma': 0.05
                })
                
                self.updateSourceList()
                self.updateVisualisation()
                self.logMessage(f"Added heat source at ({xClick:.3f}, {yClick:.3f}, {zClick:.3f})")
            else:
                self.logMessage("Cannot add heat source outside domain.")
    
    def onDomainChange(self, event=None):
        """Handle domain type change"""
        newDomain = self.domainVar.get()
        if newDomain != self.domainType:
            self.domainType = newDomain
            self.domain = DomainFactory.createDomain(self.domainType)
            self.logMessage(f"Changed domain to: {self.domainType}")
            
            self.pinnModel = None
            self.numericalData = None
            self.compareBtn.config(state='disabled')
            self.animateBtn.config(state='disabled')
            
            self.updateVisualisation()

    def load_obj_file(self):
        """Load an .obj file, convert it to a voxelised domain, and set it as the current domain."""
        filepath = filedialog.askopenfilename(
            title="Select .obj or .npy File",
            filetypes=(("3D Files", "*.obj *.npy"), ("All files", "*.*" ))
        )
        if not filepath:
            return

        if filepath.endswith('.npy'):
            self.resultQueue.put({'type': 'domain_change', 'domain_type': filepath})
        elif filepath.endswith('.obj'):
            output_npy_path = filepath.replace('.obj', '.npy')
            resolution = self.resolutionVar.get()

            def convert_and_load():
                try:
                    self.logMessage(f"Converting {filepath} to .npy with resolution {resolution}...")
                    convertObjToSdf(filepath, output_npy_path, resolution=resolution)
                    self.logMessage(f"Conversion complete. Saved to {output_npy_path}")
                    self.resultQueue.put({'type': 'domain_change', 'domain_type': output_npy_path})
                except Exception as e:
                    self.resultQueue.put({'type': 'error', 'message': f"Failed to convert .obj file: {e}"})

            threading.Thread(target=convert_and_load, daemon=True).start()
    
    def onAlphaChange(self, event=None):
        """Handle thermal diffusivity change"""
        self.alpha = self.alphaVar.get()
        self.alphaLabel.config(text=f"α = {self.alpha:.3f}")
        
        self.numericalData = None
        self.compareBtn.config(state='disabled')
        self.animateBtn.config(state='disabled')
    
    def onTMaxChange(self, event=None):
        """Handle max time change"""
        self.tMax = self.tMaxVar.get()
        self.tMaxLabel.config(text=f"t_max = {self.tMax:.1f}")
        self.timeScale.config(to=self.tMax)
        
        self.numericalData = None
        self.compareBtn.config(state='disabled')
        self.animateBtn.config(state='disabled')
    
    def onTimeChange(self, event=None):
        """Handle time slider change"""
        self.tCurrent = self.tVar.get()
        self.timeLabel.config(text=f"t = {self.tCurrent:.3f}")
        
        if self.pinnModel or self.numericalData:
            self.updateSolutionVisualisation()

    def onSmoothingChange(self, event=None):
        """Handle smoothing slider change"""
        self.smoothingLabel.config(text=f"Smoothing = {self.smoothingVar.get():.1f}")
    
    def addRandomSource(self):
        """Add randomly positioned heat source"""
        bounds = self.domain.bounds()
        
        xPos = np.random.uniform(bounds[0][0] + 0.1, bounds[0][1] - 0.1)
        yPos = np.random.uniform(bounds[1][0] + 0.1, bounds[1][1] - 0.1)
        zPos = np.random.uniform(bounds[2][0] + 0.1, bounds[2][1] - 0.1)
        
        attempts = 0
        while not self.domain.isInside(np.array([xPos]), np.array([yPos]), np.array([zPos]))[0] and attempts < 20:
            xPos = np.random.uniform(bounds[0][0] + 0.1, bounds[0][1] - 0.1)
            yPos = np.random.uniform(bounds[1][0] + 0.1, bounds[1][1] - 0.1)
            zPos = np.random.uniform(bounds[2][0] + 0.1, bounds[2][1] - 0.1)
            attempts += 1
        
        if attempts >= 20:
            self.logMessage("Could not find valid position for random source.")
            return
        
        self.heatSources.append({
            'position': (xPos, yPos, zPos),
            'amplitude': np.random.uniform(0.5, 2.0),
            'sigma': np.random.uniform(0.03, 0.08)
        })
        
        self.updateSourceList()
        self.updateVisualisation()
        self.logMessage(f"Added random heat source at ({xPos:.3f}, {yPos:.3f}, {zPos:.3f})")
    
    def addManualSource(self):
        """Add manually specified heat source"""
        try:
            xPos = float(self.xEntry.get())
            yPos = float(self.yEntry.get())
            zPos = float(self.zEntry.get())
            amplitude = float(self.ampEntry.get())
            
            if self.domain.isInside(np.array([xPos]), np.array([yPos]), np.array([zPos]))[0]:
                self.heatSources.append({
                    'position': (xPos, yPos, zPos),
                    'amplitude': amplitude,
                    'sigma': 0.05
                })
                
                self.updateSourceList()
                self.updateVisualisation()
                self.logMessage(f"Added manual heat source at ({xPos:.3f}, {yPos:.3f}, {zPos:.3f})")
            else:
                messagebox.showerror("Error", "Position is outside the selected domain.")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
    
    def removeSource(self):
        """Remove selected heat source"""
        selection = self.sourceListbox.curselection()
        if selection:
            idx = selection[0]
            removedSource = self.heatSources.pop(idx)
            self.updateSourceList()
            self.updateVisualisation()
            pos = removedSource['position']
            self.logMessage(f"Removed heat source at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    def clearSources(self):
        """Clear all heat sources"""
        if self.heatSources:
            self.heatSources.clear()
            self.updateSourceList()
            self.updateVisualisation()
            self.logMessage("Cleared all heat sources.")
    
    def updateSourceList(self):
        """Update the source listbox"""
        self.sourceListbox.delete(0, tk.END)
        for i, source in enumerate(self.heatSources):
            pos = source['position']
            amp = source['amplitude']
            self.sourceListbox.insert(tk.END, f"Source {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) A={amp:.2f}")
    
    def updateVisualisation(self):
        """Update the main visualisation"""
        self.fig.clear()
        
        ax = self.fig.add_subplot(111)
        
        bounds = self.domain.bounds()
        xTest = np.linspace(bounds[0][0], bounds[0][1], 100)
        yTest = np.linspace(bounds[1][0], bounds[1][1], 100)
        XTest, YTest = np.meshgrid(xTest, yTest)
        ZTest = np.full_like(XTest, (bounds[2][0] + bounds[2][1]) / 2)
        
        insideMask = self.domain.isInside(XTest, YTest, ZTest)
        ax.contour(XTest, YTest, insideMask.astype(float), levels=[0.5], colors='black', linewidths=2)
        ax.contourf(XTest, YTest, insideMask.astype(float), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.3)
        
        if not self.heatSources:
            ax.set_title(f"Domain: {self.getSanitisedDomainName()} (Double-click to add heat source)")
        else:
            colours = ['red', 'orange', 'yellow', 'purple', 'green', 'cyan', 'magenta']
            for i, source in enumerate(self.heatSources):
                x0, y0, z0 = source['position']
                amplitude = source['amplitude']
                sigma = source.get('sigma', 0.05)
                
                ax.scatter([x0], [y0], c=colours[i % len(colours)], 
                          s=200 * amplitude, alpha=0.8, 
                          label=f'Source {i+1}')
                
                circle = patches.Circle((x0, y0), 3*sigma, fill=False, 
                                      color=colours[i % len(colours)], alpha=0.5, linestyle='--')
                ax.add_patch(circle)
            
            ax.set_title(f'Heat Sources - {self.getSanitisedDomainName()}')
            ax.legend()

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def updateSolutionVisualisation(self):
        """Update visualisation with current solution"""
        if not (self.pinnModel or self.numericalData):
            return
        
        self.fig.clear()
        
        if self.pinnModel and self.numericalData:
            ax1 = self.fig.add_subplot(131)
            ax2 = self.fig.add_subplot(132)
            ax3 = self.fig.add_subplot(133)
            axes = [ax1, ax2, ax3]
            titles = ['PINN Solution', 'Numerical Solution', 'Absolute Error']
        elif self.pinnModel:
            ax1 = self.fig.add_subplot(111)
            axes = [ax1]
            titles = ['PINN Solution']
        else:
            ax1 = self.fig.add_subplot(111)
            axes = [ax1]
            titles = ['Numerical Solution']
        
        solutions = []
        
        if self.pinnModel:
            try:
                device = 'cpu'
                pinnSol = predictSolution3d(self.pinnModel, self.domain, self.tCurrent,
                                             nX=32, nY=32, nZ=32, device=device)
                solutions.append(pinnSol['u'])
            except Exception as e:
                self.logMessage(f"Error predicting PINN solution: {e}")
                return
        
        if self.numericalData:
            timeIdx = np.argmin(np.abs(self.numericalData['times'] - self.tCurrent))
            numSol = self.numericalData['solutions'][timeIdx]
            
            if len(numSol.shape) == 1:
                gridShape = (32, 32, 32)
                X = np.linspace(0, 1, 32)
                Y = np.linspace(0, 1, 32)
                Z = np.linspace(0, 1, 32)
                XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing='ij')
                
                numSolGrid = np.full(gridShape, np.nan)
                if 'interior_indices' in self.numericalData:
                    interiorIdx = self.numericalData['interior_indices']
                    for i, val in enumerate(numSol):
                        if i < len(interiorIdx[0]):
                            ii, jj, kk = interiorIdx[0][i], interiorIdx[1][i], interiorIdx[2][i]
                            scaleX = (gridShape[0] - 1) / (self.numericalData['grid_shape'][0] - 1)
                            scaleY = (gridShape[1] - 1) / (self.numericalData['grid_shape'][1] - 1)
                            scaleZ = (gridShape[2] - 1) / (self.numericalData['grid_shape'][2] - 1)
                            
                            newI = int(ii * scaleX)
                            newJ = int(jj * scaleY)
                            newK = int(kk * scaleZ)
                            
                            if (0 <= newI < gridShape[0] and 
                                0 <= newJ < gridShape[1] and 
                                0 <= newK < gridShape[2]):
                                numSolGrid[newI, newJ, newK] = val
                
                solutions.append(numSolGrid)
            else:
                solutions.append(numSol)
        
        for i, (ax, title, sol) in enumerate(zip(axes, titles, solutions)):
            if i == 2 and len(solutions) >= 2:
                sol = np.abs(solutions[0] - solutions[1])
                cmap = 'Reds'
            else:
                cmap = 'hot'
            
            midZ = sol.shape[2] // 2
            sliceData = sol[:, :, midZ]
            
            xCoords = np.linspace(0, 1, sol.shape[0])
            yCoords = np.linspace(0, 1, sol.shape[1])
            
            im = ax.contourf(xCoords, yCoords, sliceData.T, levels=20, cmap=cmap)
            ax.set_title(f'{title}\nt = {self.tCurrent:.3f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            
            self.fig.colorbar(im, ax=ax, shrink=0.8)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def startTraining(self):
        """Start PINN training in background"""
        if not TORCH_AVAILABLE:
            messagebox.showerror("Error", "PyTorch not available. Cannot train PINN.")
            return
        
        if not self.heatSources:
            messagebox.showerror("Error", "Please add at least one heat source before training.")
            return
        
        if self.isTraining:
            messagebox.showwarning("Warning", "Training is already in progress.")
            return
        
        self.logMessage("Starting PINN training (1000 epochs)...")
        self.trainBtn.config(state='disabled', text='Training...')
        self.trainFullBtn.config(state='disabled')
        self.isTraining = True
        
        self.taskQueue.put({
            'type': 'train_pinn',
            'quality': 'quick',
            'domain_type': self.domainType,
            'heat_sources': self.heatSources.copy(),
            'alpha': self.alpha,
            'progress_callback': self.updateProgress
        })

    def startTrainingFull(self):
        """Start a full-quality PINN training"""
        if not TORCH_AVAILABLE:
            messagebox.showerror("Error", "PyTorch not available. Cannot train PINN.")
            return
        
        if not self.heatSources:
            messagebox.showerror("Error", "Please add at least one heat source before training.")
            return

        if self.isTraining:
            messagebox.showwarning("Warning", "A training process is already running.")
            return

        self.logMessage("Confirmation dialog for full training initiated.")
        if not messagebox.askyesno("Confirmation", "This will start a full-quality training (20,000 epochs) that may take a long time. Are you sure?"):
            self.logMessage("Full training cancelled by user.")
            return

        self.logMessage("Starting PINN training (Full Quality)... This will take a while.")
        self.trainBtn.config(state='disabled')
        self.trainFullBtn.config(state='disabled', text='Training...')
        self.isTraining = True
        
        self.taskQueue.put({
            'type': 'train_pinn',
            'quality': 'full',
            'domain_type': self.domainType,
            'heat_sources': self.heatSources.copy(),
            'alpha': self.alpha,
            'progress_callback': self.updateProgress
        })
    
    def startNumericalSolve(self):
        """Start numerical solving in background"""
        if not self.heatSources:
            messagebox.showerror("Error", "Please add at least one heat source before solving.")
            return
        
        self.logMessage("Starting numerical solution...")
        self.solveBtn.config(state='disabled', text='Solving...')
        
        self.taskQueue.put({
            'type': 'solve_numerical',
            'domain_type': self.domainType,
            'heat_sources': self.heatSources.copy(),
            'alpha': self.alpha,
            't_final': self.tMax
        })
    
    def compareSolutions(self):
        """Compare PINN and numerical solutions"""
        if not (self.pinnModel and self.numericalData):
            messagebox.showwarning("Warning", "Both PINN and numerical solutions are required for comparison.")
            return
        
        self.logMessage("Creating detailed comparison...")
        
        try:
            device = 'cpu'
            pinnData = predictSolution3d(self.pinnModel, self.domain, self.tCurrent,
                                          nX=48, nY=48, nZ=48, device=device)
            
            timeIdx = np.argmin(np.abs(self.numericalData['times'] - self.tCurrent))
            
            self.viz.createComparisonPlot(pinnData, self.numericalData, timeIdx,
                                          saveName=f"comparison_{self.getSanitisedDomainName()}")
            
            self.logMessage("Comparison saved to ./demo_output/")
            
        except Exception as e:
            self.logMessage(f"Error creating comparison: {e}")
    
    def show3dPlot(self):
        """Generate and show an interactive 3D plot"""
        if not self.pinnModel:
            messagebox.showwarning("Warning", "A trained PINN model is required for 3D visualisation.")
            return

        self.logMessage("Generating interactive 3D plot with volumetric rendering...")

        try:
            saveName = f"interactive_3d_{self.getSanitisedDomainName()}"
            fig = self.viz.plotVolumetricRendering(self.pinnModel, self.domain, 
                                                     tMax=self.tMax, timeSteps=20,
                                                     title=f"PINN 3D Solution - {self.domainType}",
                                                     saveName=saveName)

            if fig:
                import webbrowser
                import os
                outputPath = os.path.join(self.viz.saveDir, f"{saveName}.html")
                webbrowser.open(f"file://{os.path.abspath(outputPath)}")
                self.logMessage(f"3D plot opened in web browser.")
            else:
                self.logMessage("Failed to generate 3D plot (Plotly may not be available).")

        except Exception as e:
            self.logMessage(f"Error creating 3D plot: {e}")

    def showSurface3dPlot(self):
        """Generate and show an interactive 3D surface plot"""
        if not self.pinnModel:
            messagebox.showwarning("Warning", "A trained PINN model is required for 3D visualisation.")
            return

        self.logMessage("Generating interactive 3D surface plot...")

        try:
            smoothingVal = self.smoothingVar.get()

            saveName = f"interactive_3d_surface_{self.getSanitisedDomainName()}"
            fig = self.viz.plotSurfaceHeatmap(self.pinnModel, self.domain, 
                                                  tMax=self.tMax, timeSteps=20,
                                                  title=f"PINN 3D Surface Solution - {self.domainType}",
                                                  saveName=saveName, smoothing=smoothingVal)

            if fig:
                import webbrowser
                import os
                outputPath = os.path.join(self.viz.saveDir, f"{saveName}.html")
                webbrowser.open(f"file://{os.path.abspath(outputPath)}")
                self.logMessage(f"3D surface plot opened in web browser.")
            else:
                self.logMessage("Failed to generate 3D surface plot.")

        except Exception as e:
            self.logMessage(f"Error creating 3D surface plot: {e}")

    def startAnimation(self):
        """Start animation of time evolution"""
        if not self.numericalData:
            messagebox.showwarning("Warning", "Numerical solution required for animation.")
            return
        
        self.logMessage("Creating animation...")
        
        try:
            self.viz.createAnimation(self.numericalData, fps=5,
                                    title=f"Heat Diffusion - {self.domainType}",
                                    saveName=f"animation_{self.getSanitisedDomainName()}")
            
            self.logMessage("Animation saved to ./demo_output/")
            
        except Exception as e:
            self.logMessage(f"Error creating animation: {e}")
    
    def backgroundWorker(self):
        """Background worker thread for computationally intensive tasks"""
        while True:
            try:
                task = self.taskQueue.get(timeout=1.0)
                
                if task['type'] == 'train_pinn':
                    self.executeTraining(task)
                elif task['type'] == 'solve_numerical':
                    self.executeNumericalSolve(task)
                elif task['type'] == 'domain_change':
                    self.domainType = task['domain_type']
                    self.domain = DomainFactory.createDomain(self.domainType)
                    self.logMessage(f"Changed domain to: {self.domainType}")
                    
                    self.pinnModel = None
                    self.numericalData = None
                    self.compareBtn.config(state='disabled')
                    self.animateBtn.config(state='disabled')
                    
                    self.updateVisualisation()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.resultQueue.put({'type': 'error', 'message': str(e)})
    
    def executeTraining(self, task):
        """Execute PINN training based on quality setting"""
        try:
            import argparse
            
            class Args:
                def __init__(self):
                    quality = task.get('quality', 'quick')
                    
                    if quality == 'full':
                        self.hiddenSize = 128
                        self.numLayers = 6
                        self.epochs = 20000
                        self.nPde = 10000
                        self.nBc = 2500
                        self.nIc = 2500
                        self.logInterval = 500
                    else:
                        self.hiddenSize = 64
                        self.numLayers = 4
                        self.epochs = 1000
                        self.nPde = 2000
                        self.nBc = 500
                        self.nIc = 500
                        self.logInterval = 200

                    self.useFourier = True
                    self.fourierScale = 1.0
                    self.useResidual = True
                    self.lr = 1e-3
                    self.warmupEpochs = 200
                    self.alpha = task['alpha']
                    self.device = 'cpu'
                    self.seed = 1337
                    self.saveDir = './demo_output'
            
            args = Args()
            
            domain = DomainFactory.createDomain(task['domain_type'])
            
            model = trainDeltaPinn3d(args, domain, task['heat_sources'], progressCallback=task.get('progress_callback'))
            
            self.resultQueue.put({
                'type': 'training_complete',
                'model': model,
                'quality': task.get('quality', 'quick')
            })
            
        except Exception as e:
            self.resultQueue.put({
                'type': 'training_error',
                'message': str(e)
            })
    
    def executeNumericalSolve(self, task):
        """Execute numerical solution"""
        try:
            solution = solveReferenceProblem(
                domainType=task['domain_type'],
                heatSources=task['heat_sources'],
                nX=48,
                alpha=task['alpha'],
                tFinal=task['t_final'],
                method='implicit',
                saveDir='./demo_output'
            )
            
            self.resultQueue.put({
                'type': 'numerical_complete',
                'solution': solution
            })
            
        except Exception as e:
            self.resultQueue.put({
                'type': 'numerical_error',
                'message': str(e)
            })
    
    def saveConfig(self):
        """Save current configuration"""
        import json
        
        config = {
            'domain_type': self.domainType,
            'heat_sources': self.heatSources,
            'alpha': self.alpha,
            't_max': self.tMax
        }
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*" )])
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            self.logMessage(f"Configuration saved to {filename}")
    
    def loadConfig(self):
        """Load configuration from file"""
        import json
        
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*" )])
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                self.domainType = config.get('domain_type', 'sphere')
                self.domainVar.set(self.domainType)
                self.domain = DomainFactory.createDomain(self.domainType)
                
                self.heatSources = config.get('heat_sources', [])
                self.alpha = config.get('alpha', 0.01)
                self.tMax = config.get('t_max', 1.0)
                
                self.alphaVar.set(self.alpha)
                self.tMaxVar.set(self.tMax)
                self.alphaLabel.config(text=f"α = {self.alpha:.3f}")
                self.tMaxLabel.config(text=f"t_max = {self.tMax:.1f}")
                self.timeScale.config(to=self.tMax)
                
                self.updateSourceList()
                self.updateVisualisation()
                
                self.logMessage(f"Configuration loaded from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def exportVisualisation(self):
        """Export current visualisation"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*" )])
        
        if filename:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            self.logMessage(f"Visualisation exported to {filename}")

    def loadModel(self):
        """Load a pre-trained model from a file."""
        if self.isTraining:
            messagebox.showwarning("Warning", "Cannot load a model while training is in progress.")
            return

        filename = filedialog.askopenfilename(
            title="Select a Model File",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*" )])

        if not filename:
            return

        try:
            self.logMessage(f"Loading model from {filename}...")
            from deltaPinn3d import loadTrainedModel
            model, checkpoint = loadTrainedModel(filename, device='cpu')

            self.pinnModel = model
            
            self.domainType = checkpoint.get('domain_name', 'sphere')
            self.heatSources = checkpoint.get('heat_sources', [])
            args = checkpoint.get('args', {})
            self.alpha = args.get('alpha', 0.01)

            self.domainVar.set(self.domainType)
            self.domain = DomainFactory.createDomain(self.domainType)
            self.alphaVar.set(self.alpha)
            self.alphaLabel.config(text=f"α = {self.alpha:.3f}")
            self.updateSourceList()
            self.updateVisualisation()
            self.updateSolutionVisualisation()

            self.saveBtn.config(state='normal')
            self.genVizBtn.config(state='normal')
            self.show3dBtn.config(state='normal')
            self.showSurface3dBtn.config(state='normal')
            if self.numericalData:
                self.compareBtn.config(state='normal')
                self.animateBtn.config(state='normal')

            self.logMessage("Model loaded successfully.")

        except ImportError:
            self.logMessage("Error: Could not import ResidualBlock. The model may be from an older version.")
            messagebox.showerror("Error", "Could not import ResidualBlock. The model may be from an older version.")
        except Exception as e:
            self.logMessage(f"Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model file: {e}")

    def saveModel(self):
        """Save the current PINN model to a file."""
        if not self.pinnModel:
            messagebox.showwarning("Warning", "No active model to save.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Model As...",
            defaultextension=".pt",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*" )])

        if not filename:
            return

        try:
            self.logMessage(f"Saving model to {filename}...")
            from deltaPinn3d import ResidualBlock
            argsToSave = {
                'hidden_size': self.pinnModel.hidden_layers[0].in_features if not self.pinnModel.use_fourier else self.pinnModel.fourier_embed.embed_dim,
                'num_layers': len([l for l in self.pinnModel.hidden_layers if isinstance(l, (torch.nn.Linear, ResidualBlock))]),
                'use_fourier': self.pinnModel.use_fourier,
                'fourier_scale': self.pinnModel.fourier_embed.B.std().item() if self.pinnModel.use_fourier else 1.0,
                'use_residual': self.pinnModel.use_residual,
                'alpha': self.alpha
            }
            torch.save({
                'model_state_dict': self.pinnModel.state_dict(),
                'history': None,
                'args': argsToSave,
                'domain_name': self.domain.name,
                'heat_sources': self.heatSources
            }, filename)
            self.logMessage("Model saved successfully.")
        except Exception as e:
            self.logMessage(f"Error saving model: {e}")
            messagebox.showerror("Error", f"Failed to save model: {e}")

    def generateAllVisualisations(self):
        """Generate and save a standard set of visualisation files, then open them."""
        if not self.pinnModel:
            messagebox.showwarning("Warning", "A trained or loaded PINN model is required.")
            return

        self.logMessage("Generating all standard visualisations... This may take a moment.")
        
        htmlFilesToOpen = []
        try:
            domainName = self.getSanitisedDomainName()

            self.viz.plotHeatSources(self.heatSources, self.domain, save_name=f"sources_{domainName}")
            pinnSolution = predictSolution3d(self.pinnModel, self.domain, self.tCurrent, nX=48, nY=48, nZ=48, device='cpu')
            self.viz.plotMatplotlibSlices(pinnSolution, title=f"PINN Solution - {domainName}", save_name=f"pinn_slices_{domainName}")

            self.logMessage("Generating volumetric plot...")
            _, volPath = self.viz.plotVolumetricRendering(self.pinnModel, self.domain, tMax=self.tMax, saveName=f"volumetric_{domainName}")
            if volPath: htmlFilesToOpen.append(volPath)

            self.logMessage("Generating surface plot...")
            _, surfPath = self.viz.plotSurfaceHeatmap(self.pinnModel, self.domain, tMax=self.tMax, saveName=f"surface_{domainName}")
            if surfPath: htmlFilesToOpen.append(surfPath)
            
            self.logMessage("Generating isosurface plot...")
            _, isoPath = self.viz.plot3dIsosurfaces(pinnSolution, title=f"PINN Solution - {domainName}", save_name=f"pinn_iso_{domainName}")
            if isoPath: htmlFilesToOpen.append(isoPath)

            self.logMessage(f"All visualisations saved to ./demo_output/")
            messagebox.showinfo("Success", "All visualisations generated. Opening interactive plots in browser...")

        except Exception as e:
            self.logMessage(f"Error generating visualisations: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")
            return

        import webbrowser
        import os
        for path in htmlFilesToOpen:
            if path:
                webbrowser.open(f"file://{os.path.abspath(path)}")
    
    def updateGui(self):
        """Periodic GUI update from background tasks"""
        try:
            while True:
                result = self.resultQueue.get_nowait()
                
                if result['type'] == 'training_complete':
                    self.pinnModel = result['model']
                    self.trainBtn.config(state='normal', text='Train PINN')
                    self.trainFullBtn.config(state='normal', text='Train PINN (Full Quality)')
                    self.isTraining = False
                    self.logMessage(f"PINN training ({result['quality']}) completed successfully!")
                    
                    self.saveBtn.config(state='normal')
                    self.genVizBtn.config(state='normal')
                    self.show3dBtn.config(state='normal')
                    self.showSurface3dBtn.config(state='normal')

                    if self.numericalData:
                        self.compareBtn.config(state='normal')
                        self.animateBtn.config(state='normal')
                
                elif result['type'] == 'training_error':
                    self.trainBtn.config(state='normal', text='Train PINN')
                    self.trainFullBtn.config(state='normal', text='Train PINN (Full Quality)')
                    self.isTraining = False
                    self.logMessage(f"PINN training failed: {result['message']}")
                
                elif result['type'] == 'numerical_complete':
                    self.numericalData = result['solution']
                    self.solveBtn.config(state='normal', text='Solve Numerical')
                    self.logMessage("Numerical solution completed successfully!")
                    if self.pinnModel:
                        self.compareBtn.config(state='normal')
                        self.show3dBtn.config(state='normal')
                        self.showSurface3dBtn.config(state='normal')
                    self.animateBtn.config(state='normal')
                
                elif result['type'] == 'numerical_error':
                    self.solveBtn.config(state='normal', text='Solve Numerical')
                    self.logMessage(f"Numerical solution failed: {result['message']}")
                
                elif result['type'] == 'domain_change':
                    self.domainType = result['domain_type']
                    self.domain = DomainFactory.createDomain(self.domainType)
                    self.domainVar.set(self.domainType)
                    self.logMessage(f"Changed domain to: {self.domainType}")
                    
                    self.pinnModel = None
                    self.numericalData = None
                    self.compareBtn.config(state='disabled')
                    self.animateBtn.config(state='disabled')
                    
                    self.updateVisualisation()

                elif result['type'] == 'error':
                    self.logMessage(f"Background task error: {result['message']}")
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self.updateGui)
    
    def logMessage(self, message):
        """Add message to status log"""
        timestamp = time.strftime("%H:%M:%S")
        formattedMessage = f"[{timestamp}] {message}\n"
        
        self.statusText.insert(tk.END, formattedMessage)
        self.statusText.see(tk.END)
        self.root.update_idletasks()

    def updateProgress(self, epoch, totalEpochs, loss):
        """Update progress bar and log"""
        if totalEpochs > 0:
            progress = int((epoch / totalEpochs) * 100)
            self.progressBar['value'] = progress
            if epoch % 100 == 0:
                self.logMessage(f"Training epoch {epoch}/{totalEpochs}, Loss: {loss:.4e}")
        self.root.update_idletasks()
    
    def run(self):
        """Run the interactive demo"""
        try:
            print("Starting mainloop...", flush=True)
            self.root.mainloop()
            print("Mainloop finished.", flush=True)
        except KeyboardInterrupt:
            print("\nInteractive demo interrupted by user.", flush=True)
        except Exception as e:
            print(f"Demo error: {e}", flush=True)
        finally:
            print("Interactive demo finished.", flush=True)


def main():
    """Main entry point for interactive demo"""
    import os
    import sys
    
    os.makedirs('./demo_output', exist_ok=True)
    
    missingDeps = []
    try:
        import torch
    except ImportError:
        missingDeps.append('torch')
    
    try:
        import matplotlib
    except ImportError:
        missingDeps.append('matplotlib')
    
    try:
        import scipy
    except ImportError:
        missingDeps.append('scipy')
    
    if missingDeps:
        print(f"Missing dependencies: {', '.join(missingDeps)}")
        print("Please install with: pip install torch matplotlib scipy")
        return
    
    print("Starting 3D Heat Diffusion Interactive Demo...")
    print("Features:")
    print("- Multiple domain shapes (cube, sphere, L-shape, torus, cylinder with holes)")
    print("- Click-to-add heat sources")
    print("- Real-time PINN vs Numerical comparison")
    print("- 3D visualisation and animation")
    print("- Export capabilities")
    print("\nDouble-click on visualisation to add heat sources!")
    
    demo = InteractiveHeatDemo()
    demo.run()


if __name__ == "__main__":
    main()