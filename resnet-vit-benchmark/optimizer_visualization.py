"""
Optimizer Trajectory Visualization on a Saddle Point Surface

This script visualizes how different optimizers navigate a saddle point,
showing their distinct behaviors on a challenging loss landscape.

Output: optimizer_saddle_point.png
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.optim as optim


def saddle_function(x, y):
    """Saddle point function: f(x,y) = x^2 - y^2"""
    return x**2 - y**2


def saddle_function_torch(params):
    """Saddle point function for torch tensors"""
    x, y = params[0], params[1]
    return x**2 - y**2


def run_optimizer(optimizer_class, optimizer_kwargs, start_point, n_steps=100):
    """
    Run an optimizer on the saddle function and record the trajectory.
    
    Args:
        optimizer_class: The optimizer class (e.g., torch.optim.SGD)
        optimizer_kwargs: Dict of kwargs for the optimizer
        start_point: Tuple (x, y) starting coordinates
        n_steps: Number of optimization steps
    
    Returns:
        List of (x, y, z) coordinates along the trajectory
    """
    # Initialize parameters
    params = torch.tensor(start_point, dtype=torch.float32, requires_grad=True)
    
    # Create optimizer
    optimizer = optimizer_class([params], **optimizer_kwargs)
    
    trajectory = []
    
    for _ in range(n_steps):
        x, y = params[0].item(), params[1].item()
        z = saddle_function(x, y)
        trajectory.append((x, y, z))
        
        optimizer.zero_grad()
        loss = saddle_function_torch(params)
        loss.backward()
        optimizer.step()
    
    # Add final point
    x, y = params[0].item(), params[1].item()
    z = saddle_function(x, y)
    trajectory.append((x, y, z))
    
    return trajectory


def main():
    # Starting point (on the saddle)
    start = [0.7, 0.01]  # Slightly off y=0 to break symmetry
    n_steps = 150
    
    # Define optimizers to compare
    optimizers = {
        'SGD': (optim.SGD, {'lr': 0.01}),
        'Momentum': (optim.SGD, {'lr': 0.01, 'momentum': 0.9}),
        'NAG': (optim.SGD, {'lr': 0.01, 'momentum': 0.9, 'nesterov': True}),
        'Adagrad': (optim.Adagrad, {'lr': 0.5}),
        'Adadelta': (optim.Adadelta, {'lr': 10.0}),
        'RMSprop': (optim.RMSprop, {'lr': 0.01}),
        'Adam': (optim.Adam, {'lr': 0.05}),
        'AdamW': (optim.AdamW, {'lr': 0.05}),
    }
    
    # Colors matching the reference image style
    colors = {
        'SGD': '#FF0000',        # Red
        'Momentum': '#00FF00',   # Green
        'NAG': '#FF00FF',        # Magenta
        'Adagrad': '#0000FF',    # Blue
        'Adadelta': '#FFFF00',   # Yellow
        'RMSprop': '#000000',    # Black
        'Adam': '#00FFFF',       # Cyan
        'AdamW': '#FF8C00',      # Orange
    }
    
    # Run all optimizers
    trajectories = {}
    for name, (opt_class, opt_kwargs) in optimizers.items():
        trajectories[name] = run_optimizer(opt_class, opt_kwargs, start.copy(), n_steps)
    
    # Create the 3D surface
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate surface mesh
    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = saddle_function(X, Y)
    
    # Plot surface with color gradient (red=high, blue=low)
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7, 
                           linewidth=0, antialiased=True,
                           rstride=2, cstride=2)
    
    # Plot optimizer trajectories
    for name, traj in trajectories.items():
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        zs = [p[2] for p in traj]
        
        # Plot trajectory line
        ax.plot(xs, ys, zs, color=colors[name], linewidth=2.5, label=name)
        
        # Mark start point
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=colors[name], s=50, marker='o')
        
        # Mark end point
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=colors[name], s=80, marker='*')
    
    # Labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('f(x,y) = x² - y²', fontsize=12)
    ax.set_title('Optimizer Trajectories on Saddle Point Surface', fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-4, 4)
    
    # Adjust view angle to match reference
    ax.view_init(elev=20, azim=45)
    
    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'results_optimizer_saddle.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    # Also create a 2D contour version for comparison
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Contour plot
    contour = ax2.contour(X, Y, Z, levels=20, cmap='coolwarm')
    ax2.clabel(contour, inline=True, fontsize=8)
    
    # Plot trajectories on 2D
    for name, traj in trajectories.items():
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax2.plot(xs, ys, color=colors[name], linewidth=2, label=name, marker='', markersize=3)
        ax2.scatter([xs[0]], [ys[0]], color=colors[name], s=100, marker='o', zorder=5)
        ax2.scatter([xs[-1]], [ys[-1]], color=colors[name], s=150, marker='*', zorder=5)
    
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_title('Optimizer Trajectories (Top View)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    output_path_2d = 'results_optimizer_contour.png'
    plt.savefig(output_path_2d, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path_2d}")
    
    plt.show()


if __name__ == "__main__":
    main()
