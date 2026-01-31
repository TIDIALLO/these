"""
Interface Oracle pour les attaques d'extraction de modèles
Simule différents niveaux d'accès à un modèle déployé via API
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod


class BaseOracle(ABC):
    """Classe abstraite pour tous les oracles"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.query_count = 0

    @abstractmethod
    def query(self, x: torch.Tensor) -> Any:
        """Requête au modèle"""
        pass

    def get_query_count(self) -> int:
        """Retourne le nombre de requêtes effectuées"""
        return self.query_count

    def reset_query_count(self):
        """Remet le compteur à zéro"""
        self.query_count = 0


class WhiteBoxOracle(BaseOracle):
    """
    Accès complet au modèle (poids, architecture, gradients)
    Utilisé principalement pour la validation
    """

    def query(self, x: torch.Tensor) -> Dict[str, Any]:
        self.query_count += 1

        with torch.no_grad():
            logits = self.model(x)

        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1),
            'prediction': logits.argmax(dim=-1),
        }

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Accès aux paramètres du modèle"""
        return dict(self.model.named_parameters())

    def get_gradients(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calcul des gradients"""
        self.model.zero_grad()
        logits = self.model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()

        return {name: param.grad.clone() for name, param in self.model.named_parameters()}


class RawOutputOracle(BaseOracle):
    """
    Accès aux logits ou probabilités (sortie brute)
    Mode d'attaque utilisé dans CRYPTO 2020 et EUROCRYPT 2024
    """

    def __init__(self, model: nn.Module, return_type: str = 'logits'):
        """
        Args:
            model: Le modèle cible
            return_type: 'logits' ou 'probabilities'
        """
        super().__init__(model)
        self.return_type = return_type

    def query(self, x: torch.Tensor) -> torch.Tensor:
        self.query_count += 1

        with torch.no_grad():
            logits = self.model(x)

            if self.return_type == 'probabilities':
                return F.softmax(logits, dim=-1)
            return logits


class HardLabelOracle(BaseOracle):
    """
    Accès uniquement à la classe prédite (hard label)
    Mode d'attaque le plus restrictif - utilisé dans EUROCRYPT 2025
    """

    def query(self, x: torch.Tensor) -> torch.Tensor:
        self.query_count += 1

        with torch.no_grad():
            logits = self.model(x)
            return logits.argmax(dim=-1)


class NoisyOracle(BaseOracle):
    """
    Oracle avec bruit ajouté aux réponses (défense simple)
    """

    def __init__(self, model: nn.Module, noise_std: float = 0.1,
                 access_level: str = 'raw_output'):
        super().__init__(model)
        self.noise_std = noise_std
        self.access_level = access_level

    def query(self, x: torch.Tensor) -> torch.Tensor:
        self.query_count += 1

        with torch.no_grad():
            logits = self.model(x)

            # Ajouter du bruit
            noisy_logits = logits + torch.randn_like(logits) * self.noise_std

            if self.access_level == 'hard_label':
                return noisy_logits.argmax(dim=-1)
            elif self.access_level == 'probabilities':
                return F.softmax(noisy_logits, dim=-1)
            return noisy_logits


class RateLimitedOracle(BaseOracle):
    """
    Oracle avec limitation du nombre de requêtes (défense)
    """

    def __init__(self, model: nn.Module, max_queries: int = 10000,
                 access_level: str = 'hard_label'):
        super().__init__(model)
        self.max_queries = max_queries
        self.access_level = access_level

    def query(self, x: torch.Tensor) -> torch.Tensor:
        if self.query_count >= self.max_queries:
            raise RuntimeError(f"Limite de requêtes atteinte ({self.max_queries})")

        self.query_count += 1

        with torch.no_grad():
            logits = self.model(x)

            if self.access_level == 'hard_label':
                return logits.argmax(dim=-1)
            elif self.access_level == 'probabilities':
                return F.softmax(logits, dim=-1)
            return logits


# =============================================================================
# Utilitaires pour les attaques
# =============================================================================

def numerical_gradient(oracle: RawOutputOracle, x: np.ndarray,
                       epsilon: float = 1e-5) -> np.ndarray:
    """
    Calcul du gradient par différences finies
    Utilisé dans les attaques d'extraction

    Args:
        oracle: Oracle avec accès raw_output
        x: Point où calculer le gradient (numpy array)
        epsilon: Pas pour les différences finies

    Returns:
        Gradient estimé
    """
    x = np.array(x, dtype=np.float32)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        f_plus = oracle.query(torch.tensor(x_plus).unsqueeze(0)).numpy()
        f_minus = oracle.query(torch.tensor(x_minus).unsqueeze(0)).numpy()

        grad[i] = (f_plus - f_minus) / (2 * epsilon)

    return grad.squeeze()


def find_decision_boundary(oracle: HardLabelOracle, x1: np.ndarray, x2: np.ndarray,
                           tolerance: float = 1e-6, max_iter: int = 100) -> Optional[np.ndarray]:
    """
    Trouve un point sur la frontière de décision entre x1 et x2
    par recherche binaire

    Args:
        oracle: Oracle hard-label
        x1, x2: Deux points de classes différentes
        tolerance: Précision de la recherche
        max_iter: Nombre maximum d'itérations

    Returns:
        Point sur la frontière ou None si même classe
    """
    x1 = np.array(x1, dtype=np.float32)
    x2 = np.array(x2, dtype=np.float32)

    # Vérifier que les points sont de classes différentes
    label1 = oracle.query(torch.tensor(x1).unsqueeze(0)).item()
    label2 = oracle.query(torch.tensor(x2).unsqueeze(0)).item()

    if label1 == label2:
        return None

    for _ in range(max_iter):
        mid = (x1 + x2) / 2

        if np.linalg.norm(x2 - x1) < tolerance:
            return mid

        mid_label = oracle.query(torch.tensor(mid).unsqueeze(0)).item()

        if mid_label == label1:
            x1 = mid
        else:
            x2 = mid

    return (x1 + x2) / 2


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    # Créer un modèle simple pour les tests
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 5)
    )

    # Test des différents oracles
    x = torch.randn(1, 10)

    print("=== Tests des Oracles ===\n")

    # White-box
    wb_oracle = WhiteBoxOracle(model)
    wb_result = wb_oracle.query(x)
    print(f"White-box - Logits: {wb_result['logits'].shape}")
    print(f"White-box - Prediction: {wb_result['prediction'].item()}")

    # Raw-output
    raw_oracle = RawOutputOracle(model, 'logits')
    raw_result = raw_oracle.query(x)
    print(f"\nRaw-output - Logits: {raw_result.shape}")

    # Hard-label
    hl_oracle = HardLabelOracle(model)
    hl_result = hl_oracle.query(x)
    print(f"\nHard-label - Prediction: {hl_result.item()}")

    # Noisy
    noisy_oracle = NoisyOracle(model, noise_std=0.5, access_level='hard_label')
    noisy_result = noisy_oracle.query(x)
    print(f"\nNoisy oracle - Prediction: {noisy_result.item()}")

    # Rate-limited
    rl_oracle = RateLimitedOracle(model, max_queries=5)
    for i in range(5):
        rl_oracle.query(x)
    print(f"\nRate-limited - Queries used: {rl_oracle.get_query_count()}")

    try:
        rl_oracle.query(x)
    except RuntimeError as e:
        print(f"Rate limit error: {e}")

    # Test gradient numérique
    print("\n=== Test Gradient Numérique ===")
    raw_oracle.reset_query_count()
    x_np = np.random.randn(10).astype(np.float32)
    grad = numerical_gradient(raw_oracle, x_np)
    print(f"Gradient shape: {grad.shape}")
    print(f"Queries used for gradient: {raw_oracle.get_query_count()}")

    # Test recherche de frontière
    print("\n=== Test Frontière de Décision ===")
    hl_oracle.reset_query_count()

    # Trouver deux points de classes différentes
    x1 = np.random.randn(10).astype(np.float32) * 2
    x2 = np.random.randn(10).astype(np.float32) * 2

    boundary = find_decision_boundary(hl_oracle, x1, x2)
    if boundary is not None:
        print(f"Point frontière trouvé")
        print(f"Queries used: {hl_oracle.get_query_count()}")
    else:
        print("Points de même classe - pas de frontière")
