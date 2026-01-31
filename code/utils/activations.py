"""
Fonctions d'activation et leurs dérivées
Pour l'analyse dans le cadre de la thèse sur l'extraction de modèles
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.special import erf


# =============================================================================
# Implémentations NumPy (pour analyse mathématique)
# =============================================================================

def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(x):
    """Dérivée de ReLU: 1 si x > 0, 0 sinon"""
    return (x > 0).astype(float)


def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    """Dérivée de sigmoid: σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)


def tanh_activation(x):
    """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return np.tanh(x)


def tanh_derivative(x):
    """Dérivée de tanh: 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2


def gelu_exact(x):
    """
    GELU exact: x * Φ(x) où Φ est la CDF normale standard
    GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    """
    return x * 0.5 * (1 + erf(x / np.sqrt(2)))


def gelu_approx(x):
    """
    Approximation GELU utilisée dans GPT/BERT:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def gelu_derivative(x):
    """
    Dérivée de GELU exact:
    GELU'(x) = Φ(x) + x * φ(x)
    où Φ est la CDF et φ est la PDF normale standard
    """
    phi_cdf = 0.5 * (1 + erf(x / np.sqrt(2)))  # CDF
    phi_pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)  # PDF
    return phi_cdf + x * phi_pdf


def silu(x):
    """
    SiLU (Swish avec β=1): x * σ(x)
    """
    return x * sigmoid(x)


def silu_derivative(x):
    """
    Dérivée de SiLU:
    SiLU'(x) = σ(x) + x * σ(x) * (1 - σ(x))
             = σ(x) * (1 + x * (1 - σ(x)))
    """
    s = sigmoid(x)
    return s * (1 + x * (1 - s))


# =============================================================================
# Analyse des propriétés
# =============================================================================

def is_piecewise_linear(activation_name):
    """Indique si l'activation est linéaire par morceaux"""
    piecewise_linear = ['relu', 'leaky_relu', 'prelu', 'elu']
    return activation_name.lower() in piecewise_linear


def second_derivative(x, activation_fn, eps=1e-5):
    """Calcul numérique de la dérivée seconde"""
    return (activation_fn(x + eps) - 2 * activation_fn(x) + activation_fn(x - eps)) / (eps ** 2)


def find_inflection_points(activation_fn, x_range=(-5, 5), resolution=10000, threshold=1e-3):
    """
    Trouve les points d'inflexion (où la dérivée seconde change de signe)
    Utile pour les activations non piecewise-linear
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    d2 = second_derivative(x, activation_fn)

    # Trouver les changements de signe
    sign_changes = np.where(np.diff(np.sign(d2)))[0]

    inflection_points = x[sign_changes]
    return inflection_points


# =============================================================================
# Comparaison visuelle
# =============================================================================

def plot_activations_comparison(save_path=None):
    """Génère une figure comparant toutes les activations"""
    import matplotlib.pyplot as plt

    x = np.linspace(-5, 5, 1000)

    activations = {
        'ReLU': (relu, relu_derivative),
        'Sigmoid': (sigmoid, sigmoid_derivative),
        'Tanh': (tanh_activation, tanh_derivative),
        'GELU': (gelu_exact, gelu_derivative),
        'SiLU/Swish': (silu, silu_derivative),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, (func, deriv)) in enumerate(activations.items()):
        ax = axes[idx]
        ax.plot(x, func(x), 'b-', label=f'{name}(x)', linewidth=2)
        ax.plot(x, deriv(x), 'r--', label=f"{name}'(x)", linewidth=1.5)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(name, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 5)

    # Cacher le dernier subplot inutilisé
    axes[-1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée: {save_path}")

    return fig


# =============================================================================
# PyTorch wrappers
# =============================================================================

class ActivationAnalyzer(nn.Module):
    """
    Module pour analyser les activations dans un réseau PyTorch
    """
    def __init__(self, activation_type='relu'):
        super().__init__()

        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
        }

        if activation_type not in activations:
            raise ValueError(f"Activation inconnue: {activation_type}")

        self.activation = activations[activation_type]
        self.activation_type = activation_type

    def forward(self, x):
        return self.activation(x)

    def is_piecewise_linear(self):
        return self.activation_type in ['relu']

    def count_active_neurons(self, x):
        """Pour ReLU: compte les neurones actifs"""
        if self.activation_type == 'relu':
            with torch.no_grad():
                pre_activation = x
                return (pre_activation > 0).float().mean().item()
        return None


if __name__ == "__main__":
    # Test des fonctions
    x = np.array([-2, -1, 0, 1, 2])

    print("Tests des activations:")
    print(f"ReLU({x}) = {relu(x)}")
    print(f"GELU({x}) = {gelu_exact(x)}")
    print(f"SiLU({x}) = {silu(x)}")

    print("\nPoints d'inflexion GELU:", find_inflection_points(gelu_exact))
    print("Points d'inflexion SiLU:", find_inflection_points(silu))

    # Générer la figure de comparaison
    plot_activations_comparison('activations_comparison.png')
