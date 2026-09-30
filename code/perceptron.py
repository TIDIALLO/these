import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def gelu(x):
    """GELU exact"""
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_derivative(x):
    """Dérivée de GELU"""
    # TODO: Implémenter
    pass

def gelu_approx(x):
    """Approximation GELU utilisée en pratique"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    """SiLU = Swish avec beta=1"""
    return x * sigmoid(x)

def silu_derivative(x):
    """Dérivée de SiLU"""
    s = sigmoid(x)
    return s + x * s * (1 - s)

# Visualisation
x = np.linspace(-5, 5, 1000)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

activations = [
    ('ReLU', relu, relu_derivative),
    ('Sigmoid', sigmoid, sigmoid_derivative),
    ('Tanh', tanh, tanh_derivative),
    ('GELU', gelu, gelu_derivative),
    ('SiLU/Swish', silu, silu_derivative),
]

for ax, (name, func, deriv) in zip(axes.flat, activations):
    ax.plot(x, func(x), 'b-', label=f'{name}(x)', linewidth=2)
    if deriv is not None:
        ax.plot(x, deriv(x), 'r--', label=f"{name}'(x)", linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activations_comparison.png', dpi=150)
plt.show()
