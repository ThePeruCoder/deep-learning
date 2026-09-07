import numpy as np
import matplotlib.pyplot as plt
import os

# Automatically create the outputs folder if it doesn't exist
os.makedirs('outputs', exist_ok=True)

def save_plot(title):
    """Helper function to save plots automatically without blocking the terminal."""
    safe_title = title.replace(' ', '_').replace('/', '_')
    file_path = os.path.join('outputs', f'{safe_title}.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()  # Close the figure to free up memory

def plot_loss(train_loss, val_loss=None, title="Average Error vs Epochs"):
    """Requirement: Plot of average error (y-axis) vs epochs (x-axis)"""
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss', linewidth=2)
    if val_loss is not None:
        plt.plot(val_loss, label='Validation Loss', linewidth=2, linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Average Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_plot(title)

def plot_decision_region(X, y, model, title="Decision Region"):
    """Requirement: Decision region plot superimposed by training data"""
    if X.shape[1] != 2:
        print(f"Skipping Decision Region for '{title}' (requires 2D input data).")
        return

    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)
    Z = preds.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.title(title)
    plt.xlabel('Feature 1 (x1)')
    plt.ylabel('Feature 2 (x2)')
    save_plot(title)

def plot_3d_activations(model, X, layer_idx, title="Node Activations"):
    """
    Requirement: Plots of outputs for each hidden node/output node.
    Handles both 1D (Univariate) and 2D (Bivariate) input data.
    """
    # Handle Univariate Data (1D Input)
    
    if X.shape[1] == 1:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        
        _ = model.forward(grid)
        layer_activations = model.activations[layer_idx]
        num_nodes = layer_activations.shape[1]
        
        fig = plt.figure(figsize=(15, 4 * ((num_nodes + 1) // 2)))
        for i in range(num_nodes):
            ax = fig.add_subplot((num_nodes + 1) // 2, 2, i + 1)
            ax.plot(grid, layer_activations[:, i], color='b', lw=2)
            ax.set_title(f'{title} - Node {i+1}')
            ax.set_xlabel('Input Feature (x)')
            ax.set_ylabel('Activation (z)')
            ax.grid(True)
        
        plt.tight_layout()
        save_plot(title)
        return


    # Handle Bivariate Data (2D Input) -> Generates 3D plots

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    grid = np.c_[xx.ravel(), yy.ravel()]
    
    _ = model.forward(grid) 
    layer_activations = model.activations[layer_idx]
    
    num_nodes = layer_activations.shape[1]
    
    fig = plt.figure(figsize=(15, 6 * ((num_nodes + 1) // 2)))
    for i in range(num_nodes):
        ax = fig.add_subplot((num_nodes + 1) // 2, 2, i + 1, projection='3d')
        Z = layer_activations[:, i].reshape(xx.shape)
        
        ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.8)
        ax.set_title(f'{title} - Node {i+1}')
        ax.set_xlabel('Input Feature 1')
        ax.set_ylabel('Input Feature 2')
        ax.set_zlabel('Activation (z)')
    
    plt.tight_layout()
    save_plot(title)

def plot_regression_scatter(y_true, y_pred, title="Target vs Model Output"):
    """Requirement: Scatter plot with target on x-axis and model output on y-axis"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y=x)')
    
    plt.xlabel('Target Output (True)')
    plt.ylabel('Model Output (Predicted)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_plot(title)
    
def plot_regression_overlay(X, y_true, model, title="Model Overlay"):
    """Requirement: Plots of model output superimposed on the target output"""
    y_true = np.asarray(y_true).ravel()
    
    # 1D Input: Line plot
    if X.shape[1] == 1:
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], y_true, color='blue', alpha=0.5, label='Target Output')
        
        # Sort X for a smooth line
        sorted_idx = np.argsort(X[:, 0])
        X_sorted = X[sorted_idx]
        preds = model.predict(X_sorted).ravel()
        
        plt.plot(X_sorted[:, 0], preds, color='red', linewidth=2, label='Model Output')
        plt.xlabel('x values')
        plt.ylabel('y value (target / model output)')
        plt.title(title)
        plt.legend()
        save_plot(title)
        
    # 2D Input: 3D Surface Plot
    elif X.shape[1] == 2:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter target points
        ax.scatter(X[:, 0], X[:, 1], y_true, color='blue', alpha=0.3, label='Target')
        
        # Surface for model predictions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = model.predict(grid).reshape(xx.shape)
        
        ax.plot_surface(xx, yy, preds, color='red', alpha=0.4)
        ax.set_xlabel('x1 values')
        ax.set_ylabel('x2 values')
        ax.set_zlabel('y value')
        ax.set_title(title)
        save_plot(title)
