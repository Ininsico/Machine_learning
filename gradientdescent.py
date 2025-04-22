import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from ctypes import windll  # For DPI awareness on Windows

# Enable DPI awareness
try:
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# Gradient Descent Implementation
def gradient_descent(X, y, lr=0.01, epochs=100, batch_size=None):
    m = len(y)
    theta = np.random.randn(2, 1)  # Random init
    losses = []
    
    for epoch in range(epochs):
        if batch_size:  # Mini-batch SGD
            batch_size = min(batch_size, m)  # Safety check
            indices = np.random.randint(0, m, batch_size)
            X_batch, y_batch = X[indices], y[indices]
        else:  # Batch GD
            X_batch, y_batch = X, y
        
        gradients = 2/m * X_batch.T.dot(X_batch.dot(theta) - y_batch)
        theta -= lr * gradients
        loss = np.mean((X.dot(theta) - y) ** 2)
        losses.append(loss)
    
    return theta, losses

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]  # Add bias term

# GUI Setup
root = tk.Tk()
root.title("Gradient Descent Visualizer")

# Make window larger
root.geometry("1000x600")

# Controls Frame
controls = ttk.Frame(root, padding="10")
controls.pack(side=tk.LEFT, fill=tk.Y)

# Learning Rate Slider
ttk.Label(controls, text="Learning Rate (α):").pack()
lr_slider = ttk.Scale(controls, from_=0.001, to=0.1, value=0.01)
lr_slider.pack()

# Epochs Slider
ttk.Label(controls, text="Epochs:").pack()
epochs_slider = ttk.Scale(controls, from_=10, to=500, value=100)
epochs_slider.pack()

# Batch Size Dropdown
ttk.Label(controls, text="Batch Size:").pack()
batch_options = ["Full Batch", "Mini-Batch (32)", "Stochastic (1)"]
batch_var = tk.StringVar(value=batch_options[0])
batch_dropdown = ttk.Combobox(controls, textvariable=batch_var, values=batch_options, state="readonly")
batch_dropdown.pack()

# Plot Frame
plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Update Function
def update_plot():
    lr = max(lr_slider.get(), 0.001)  # Prevent too small values
    epochs = int(epochs_slider.get())
    batch_text = batch_var.get()
    
    if batch_text == "Full Batch":
        batch_size = None
    elif batch_text == "Mini-Batch (32)":
        batch_size = 32
    else:
        batch_size = 1
    
    theta, losses = gradient_descent(X_b, y, lr, epochs, batch_size)
    
    ax1.clear()
    ax1.scatter(X, y, alpha=0.5)
    ax1.plot(X, X_b.dot(theta), 'r-', label=f"θ0={theta[0][0]:.2f}, θ1={theta[1][0]:.2f}")
    ax1.set_title("Linear Regression Fit")
    ax1.legend()
    
    ax2.clear()
    ax2.plot(losses)
    ax2.set_title("Loss Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    
    canvas.draw()

ttk.Button(controls, text="Run Gradient Descent", command=update_plot).pack(pady=10)
update_plot()  # Initial run

root.mainloop()