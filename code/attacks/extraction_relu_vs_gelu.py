"""
Reproduction de l'attaque d'extraction CRYPTO 2020 sur ReLU
et démonstration de son échec sur GELU/SiLU.

Objectif pédagogique:
- Comprendre POURQUOI ReLU est vulnérable (points de coude)
- Comprendre POURQUOI GELU/SiLU résistent (fonctions lisses)
- Voir concrètement la différence de comportement

Usage:
    python extraction_relu_vs_gelu.py
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.special import erf

# ─────────────────────────────────────────────────────────────
# PARTIE 1 : RÉSEAUX CIBLES
# ─────────────────────────────────────────────────────────────

def make_relu_network(input_dim=2, hidden_dim=4, seed=42):
    """Réseau ReLU - architecture typique CRYPTO 2020."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    return model


def make_gelu_network(input_dim=2, hidden_dim=4, seed=42):
    """Même architecture mais avec GELU - cible de la thèse."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 1),
    )
    return model


def make_silu_network(input_dim=2, hidden_dim=4, seed=42):
    """Même architecture mais avec SiLU/Swish."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )
    return model


def oracle(model, x_np):
    """Interface boîte noire : entrée numpy → sortie scalaire."""
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return model(x).item()


# ─────────────────────────────────────────────────────────────
# PARTIE 2 : OUTILS MATHÉMATIQUES
# ─────────────────────────────────────────────────────────────

def numerical_gradient(model_fn, x, eps=1e-6):
    """
    Gradient numérique par différences finies centrées.
    ∂f/∂xᵢ ≈ [f(x + ε·eᵢ) - f(x - ε·eᵢ)] / (2ε)
    """
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (model_fn(xp) - model_fn(xm)) / (2 * eps)
    return grad


def gradient_change_magnitude(model_fn, x, direction, delta=1e-4):
    """
    Mesure le changement de gradient de part et d'autre d'un point.
    Si grand → point de coude (kink) → neurone ReLU qui commute.
    Si petit → pas de discontinuité → activation lisse (GELU/SiLU).
    """
    direction = direction / np.linalg.norm(direction)
    x_before = x - delta * direction
    x_after  = x + delta * direction
    g_before = numerical_gradient(model_fn, x_before)
    g_after  = numerical_gradient(model_fn, x_after)
    return np.linalg.norm(g_after - g_before)


# ─────────────────────────────────────────────────────────────
# PARTIE 3 : TROUVER LES POINTS CRITIQUES (KINKS)
# ─────────────────────────────────────────────────────────────

def binary_search_kink(model_fn, x_start, direction, t_range=(-3.0, 3.0),
                       tol=1e-7, n_probes=500):
    """
    Recherche binaire d'un point de coude (kink) ReLU le long d'une direction.

    Principe CRYPTO 2020 :
    On trace la valeur de f le long d'une ligne : f(x + t·d) pour t ∈ [a,b]
    Là où la dérivée change brusquement = un neurone ReLU a commué.

    Retourne (t_kink, x_kink) ou None si aucun kink trouvé.
    """
    direction = direction / np.linalg.norm(direction)
    t_vals = np.linspace(t_range[0], t_range[1], n_probes)
    f_vals = np.array([model_fn(x_start + t * direction) for t in t_vals])

    # Dérivée numérique le long de la ligne
    derivs = np.diff(f_vals) / np.diff(t_vals)
    # Changements brusques de dérivée = kinks
    kink_strength = np.abs(np.diff(derivs))

    threshold = np.mean(kink_strength) + 2.5 * np.std(kink_strength)
    candidates = np.where(kink_strength > threshold)[0]

    if len(candidates) == 0:
        return None

    # Affiner le meilleur candidat par recherche binaire
    best_idx = candidates[np.argmax(kink_strength[candidates])]
    t_lo, t_hi = t_vals[best_idx], t_vals[best_idx + 2]

    for _ in range(50):
        t_mid = (t_lo + t_hi) / 2
        change = gradient_change_magnitude(
            model_fn, x_start + t_mid * direction, direction
        )
        t_lo_change = gradient_change_magnitude(
            model_fn, x_start + t_lo * direction, direction
        )
        if t_lo_change > change:
            t_hi = t_mid
        else:
            t_lo = t_mid
        if abs(t_hi - t_lo) < tol:
            break

    t_kink = (t_lo + t_hi) / 2
    return t_kink, x_start + t_kink * direction


def find_all_kinks(model_fn, input_dim, n_searches=80, seed=0):
    """
    Cherche des points critiques sur plusieurs directions aléatoires.
    Retourne la liste des kinks trouvés et le nombre de requêtes.
    """
    rng = np.random.default_rng(seed)
    kinks = []
    queries = [0]

    original_fn = model_fn

    def counting_fn(x):
        queries[0] += 1
        return original_fn(x)

    for _ in range(n_searches):
        x0 = rng.standard_normal(input_dim) * 1.5
        d  = rng.standard_normal(input_dim)
        result = binary_search_kink(counting_fn, x0, d)
        if result is not None:
            kinks.append(result[1])

    return kinks, queries[0]


# ─────────────────────────────────────────────────────────────
# PARTIE 4 : EXTRACTION SIMPLIFIÉE (ReLU)
# ─────────────────────────────────────────────────────────────

def extract_weights_relu(model, model_fn, input_dim, hidden_dim):
    """
    Extraction simplifiée des poids W₁ de la couche cachée.

    Principe :
    Au point critique xᵢ, le neurone i passe de 0 à actif.
    Le gradient change brusquement de part et d'autre.
    La différence Δg = g_après - g_avant est proportionnelle à w₂ᵢ · w₁ᵢᵀ.
    En collectant ces Δg depuis plusieurs points critiques, on peut
    identifier les directions des vecteurs de poids w₁ᵢ.
    """
    rng = np.random.default_rng(42)
    gradient_jumps = []

    for _ in range(hidden_dim * 15):
        x0 = rng.standard_normal(input_dim) * 1.5
        d  = rng.standard_normal(input_dim)
        result = binary_search_kink(model_fn, x0, d)
        if result is None:
            continue
        _, x_kink = result
        eps = 1e-4
        d_unit = d / np.linalg.norm(d)
        g_before = numerical_gradient(model_fn, x_kink - eps * d_unit)
        g_after  = numerical_gradient(model_fn, x_kink + eps * d_unit)
        delta_g = g_after - g_before
        if np.linalg.norm(delta_g) > 1e-6:
            gradient_jumps.append(delta_g / np.linalg.norm(delta_g))

    # Les vrais poids W₁ (normalisés) pour comparaison
    true_W1 = model[0].weight.detach().numpy()  # (hidden_dim, input_dim)
    true_W1_norm = true_W1 / np.linalg.norm(true_W1, axis=1, keepdims=True)

    return gradient_jumps, true_W1_norm


def cosine_similarity_best_match(extracted_dirs, true_W1_norm):
    """
    Pour chaque vecteur de poids réel, trouve la meilleure correspondance
    parmi les directions extraites. Score : cosine similarity max.
    """
    if not extracted_dirs:
        return [], 0.0

    scores = []
    for w_true in true_W1_norm:
        best = max(
            abs(np.dot(w_true, d)) for d in extracted_dirs
        )
        scores.append(best)
    return scores, np.mean(scores)


# ─────────────────────────────────────────────────────────────
# PARTIE 5 : MESURER LE NIVEAU DE "KINKINESS"
# ─────────────────────────────────────────────────────────────

def measure_kinkiness(model_fn, input_dim, n_samples=200, seed=1):
    """
    Mesure quantitative de la discontinuité du gradient.

    Pour ReLU   : valeurs élevées (gradients discontinus)
    Pour GELU   : valeurs très faibles (gradients continus)
    Pour SiLU   : valeurs très faibles (gradients continus)

    C'est LE critère qui explique pourquoi l'attaque fonctionne
    sur ReLU mais échoue sur GELU/SiLU.
    """
    rng = np.random.default_rng(seed)
    changes = []
    for _ in range(n_samples):
        x = rng.standard_normal(input_dim) * 1.5
        d = rng.standard_normal(input_dim)
        changes.append(gradient_change_magnitude(model_fn, x, d))
    return np.mean(changes), np.max(changes)


# ─────────────────────────────────────────────────────────────
# PARTIE 6 : VISUALISATION 1D
# ─────────────────────────────────────────────────────────────

def plot_1d_comparison():
    """
    Visualise le comportement des 3 activations sur un réseau 1D→4→1.
    Montre clairement les kinks ReLU vs la continuité GELU/SiLU.
    """
    torch.manual_seed(0)
    relu_net = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))
    gelu_net = nn.Sequential(nn.Linear(1, 4), nn.GELU(), nn.Linear(4, 1))
    silu_net = nn.Sequential(nn.Linear(1, 4), nn.SiLU(), nn.Linear(4, 1))

    # Copier les mêmes poids pour comparaison équitable
    with torch.no_grad():
        gelu_net[0].weight.copy_(relu_net[0].weight)
        gelu_net[0].bias.copy_(relu_net[0].bias)
        gelu_net[2].weight.copy_(relu_net[2].weight)
        gelu_net[2].bias.copy_(relu_net[2].bias)
        silu_net[0].weight.copy_(relu_net[0].weight)
        silu_net[0].bias.copy_(relu_net[0].bias)
        silu_net[2].weight.copy_(relu_net[2].weight)
        silu_net[2].bias.copy_(relu_net[2].bias)

    x_vals = np.linspace(-3, 3, 2000)

    def eval_net(net, x_arr):
        with torch.no_grad():
            t = torch.tensor(x_arr.reshape(-1, 1), dtype=torch.float32)
            return net(t).numpy().flatten()

    f_relu = eval_net(relu_net, x_vals)
    f_gelu = eval_net(gelu_net, x_vals)
    f_silu = eval_net(silu_net, x_vals)

    # Dérivées numériques (= là où les kinks sont visibles)
    dx = x_vals[1] - x_vals[0]
    d_relu = np.gradient(f_relu, dx)
    d_gelu = np.gradient(f_gelu, dx)
    d_silu = np.gradient(f_silu, dx)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Réseaux ReLU vs GELU vs SiLU : fonctions et dérivées\n"
        "(Même architecture, mêmes poids - seule l'activation change)",
        fontsize=13
    )

    colors = ["#e74c3c", "#2980b9", "#27ae60"]
    titles = ["ReLU", "GELU", "SiLU/Swish"]
    funcs  = [(f_relu, d_relu), (f_gelu, d_gelu), (f_silu, d_silu)]
    notes  = [
        "Points de coude = kinks\n→ Attaque CRYPTO 2020 possible",
        "Lisse, sans kink\n→ Attaque directe IMPOSSIBLE",
        "Lisse, non-monotone\n→ Attaque directe IMPOSSIBLE",
    ]

    for col, (color, title, (f, df), note) in enumerate(
        zip(colors, titles, funcs, notes)
    ):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        ax_top.plot(x_vals, f, color=color, linewidth=2)
        ax_top.set_title(f"f(x) — {title}", fontsize=11)
        ax_top.set_xlabel("x")
        ax_top.set_ylabel("f(x)")
        ax_top.grid(True, alpha=0.3)

        # Annoter les kinks sur ReLU
        if title == "ReLU":
            d2 = np.abs(np.diff(df))
            kink_idxs = np.where(d2 > np.mean(d2) + 2 * np.std(d2))[0]
            for ki in kink_idxs:
                ax_top.axvline(x_vals[ki], color="red", alpha=0.4,
                               linestyle="--", linewidth=1)
            ax_top.text(
                0.05, 0.95, f"{len(kink_idxs)} kinks détectés\n(points exploitables)",
                transform=ax_top.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow"),
            )

        ax_bot.plot(x_vals, df, color=color, linewidth=2)
        ax_bot.set_title(f"f'(x) - {title}", fontsize=11)
        ax_bot.set_xlabel("x")
        ax_bot.set_ylabel("f'(x)")
        ax_bot.grid(True, alpha=0.3)
        ax_bot.text(
            0.05, 0.05, note,
            transform=ax_bot.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow"),
        )

    plt.tight_layout()
    plt.savefig("relu_vs_gelu_silu_comparison.png", dpi=150, bbox_inches="tight")
    print("  Figure sauvegardée : relu_vs_gelu_silu_comparison.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
# PARTIE 7 : EXPÉRIENCE PRINCIPALE
# ─────────────────────────────────────────────────────────────

def run_experiment():
    """
    Expérience complète :
    1. Mesurer la "kinkiness" (discontinuité gradient) pour ReLU / GELU / SiLU
    2. Chercher des points critiques avec l'algorithme CRYPTO 2020
    3. Tenter d'extraire les poids de chaque réseau
    4. Comparer les résultats
    """
    print("=" * 65)
    print("REPRODUCTION ATTAQUE CRYPTO 2020")
    print("ReLU vs GELU vs SiLU - Thèse Tidiane DIALLO")
    print("=" * 65)

    INPUT_DIM  = 2
    HIDDEN_DIM = 4

    relu_net = make_relu_network(INPUT_DIM, HIDDEN_DIM)
    gelu_net = make_gelu_network(INPUT_DIM, HIDDEN_DIM)
    silu_net = make_silu_network(INPUT_DIM, HIDDEN_DIM)

    # Mêmes poids pour les 3 réseaux
    with torch.no_grad():
        gelu_net[0].weight.copy_(relu_net[0].weight)
        gelu_net[0].bias.copy_(relu_net[0].bias)
        gelu_net[2].weight.copy_(relu_net[2].weight)
        gelu_net[2].bias.copy_(relu_net[2].bias)
        silu_net[0].weight.copy_(relu_net[0].weight)
        silu_net[0].bias.copy_(relu_net[0].bias)
        silu_net[2].weight.copy_(relu_net[2].weight)
        silu_net[2].bias.copy_(relu_net[2].bias)

    relu_fn = lambda x: oracle(relu_net, x)
    gelu_fn = lambda x: oracle(gelu_net, x)
    silu_fn = lambda x: oracle(silu_net, x)

    # ── ÉTAPE 1 : Mesure de la discontinuité du gradient ──────
    print("\n── ÉTAPE 1 : Discontinuité du gradient (kinkiness) ──────")
    print("Mesure : magnitude moyenne du saut de gradient en un point aléatoire")
    print("ReLU → valeurs élevées | GELU/SiLU → valeurs quasi nulles\n")

    for name, fn in [("ReLU ", relu_fn), ("GELU ", gelu_fn), ("SiLU ", silu_fn)]:
        mean_change, max_change = measure_kinkiness(fn, INPUT_DIM, n_samples=300)
        exploitable = "✓ EXPLOITABLE" if mean_change > 0.05 else "✗ NON exploitable"
        print(f"  {name} | saut moyen = {mean_change:.5f} | max = {max_change:.5f} | {exploitable}")

    # ── ÉTAPE 2 : Recherche de points critiques ───────────────
    print("\n── ÉTAPE 2 : Recherche de kinks (CRYPTO 2020) ───────────")
    print(f"Paramètre : {HIDDEN_DIM} neurones cachés → {HIDDEN_DIM} kinks théoriques\n")

    results = {}
    for name, fn in [("ReLU", relu_fn), ("GELU", gelu_fn), ("SiLU", silu_fn)]:
        kinks, n_queries = find_all_kinks(fn, INPUT_DIM, n_searches=80)
        results[name] = {"kinks": kinks, "queries": n_queries}
        print(
            f"  {name:4s} | kinks trouvés = {len(kinks):3d} | requêtes = {n_queries:5d}"
        )

    # ── ÉTAPE 3 : Tentative d'extraction des poids ────────────
    print("\n── ÉTAPE 3 : Tentative d'extraction des poids W₁ ────────")
    print("Score = cosine similarity avec les vrais poids (1.0 = extraction parfaite)\n")

    for name, fn, net in [
        ("ReLU", relu_fn, relu_net),
        ("GELU", gelu_fn, gelu_net),
        ("SiLU", silu_fn, silu_net),
    ]:
        grad_jumps, true_W1_norm = extract_weights_relu(net, fn, INPUT_DIM, HIDDEN_DIM)
        scores, mean_score = cosine_similarity_best_match(grad_jumps, true_W1_norm)
        print(f"  {name:4s} | directions extraites = {len(grad_jumps):3d} | "
              f"score moyen = {mean_score:.3f} | scores = {[f'{s:.3f}' for s in scores]}")

    # ── SYNTHÈSE ──────────────────────────────────────────────
    print("\n── SYNTHÈSE ──────────────────────────────────────────────")
    print("""
  ReLU  → Kinks clairs, extraction possible, score proche de 1.0
           Méthode CRYPTO 2020 fonctionne directement.

  GELU  → Pas de kinks (gradient continu partout), extraction échoue.
           Les "changements de gradient" trouvés sont du bruit numérique.
           Raison mathématique : GELU''(x) ≠ 0 partout → pas de discontinuité.

  SiLU  → Même constat que GELU.
           SiLU(x) = x·σ(x) est C∞, pas de non-différentiabilité.

  → C'est exactement le verrou de la thèse :
    Generaliser l'attaque à ces fonctions lisses est une question ouverte.
""")

    return results


# ─────────────────────────────────────────────────────────────
# PARTIE 8 : ANALYSE MATHÉMATIQUE DES ACTIVATIONS
# ─────────────────────────────────────────────────────────────

def plot_activation_analysis():
    """
    Analyse mathématique : fonctions, dérivées premières et secondes.
    La dérivée seconde est clé : pour ReLU elle est nulle partout
    sauf aux kinks (impulsion de Dirac) → exploitable.
    Pour GELU/SiLU elle est continue et non nulle → rien à exploiter directement.
    """
    x = np.linspace(-3, 3, 2000)

    # Fonctions et dérivées
    relu   = np.maximum(0, x)
    d_relu = (x > 0).astype(float)

    gelu   = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    phi    = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)   # PDF normale
    Phi    = 0.5 * (1 + erf(x / np.sqrt(2)))           # CDF normale
    d_gelu = Phi + x * phi                             # GELU'(x) = Φ(x) + x·φ(x)
    d2_gelu = 2 * phi + x * (-x * phi)                # GELU''(x)

    silu   = x / (1 + np.exp(-x))
    sig    = 1 / (1 + np.exp(-x))
    d_silu = sig + x * sig * (1 - sig)                # SiLU'(x)
    d2_silu = np.gradient(d_silu, x[1] - x[0])       # SiLU''(x) numérique

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        "Analyse mathématique : ReLU vs GELU vs SiLU\n"
        "Fonctions | Dérivées premières | Dérivées secondes",
        fontsize=13,
    )

    data = [
        ("ReLU",        x, relu,    d_relu,  np.gradient(d_relu, x[1]-x[0]),
         "#e74c3c",
         "f''(x) = 0 partout\nsauf kinks (Dirac)\n→ EXPLOITABLE"),
        ("GELU",        x, gelu,    d_gelu,  d2_gelu,
         "#2980b9",
         "f''(x) continue\net non nulle\n→ Pas de kinks"),
        ("SiLU/Swish",  x, silu,    d_silu,  d2_silu,
         "#27ae60",
         "f''(x) continue\net non nulle\n→ Pas de kinks"),
    ]

    row_labels = ["f(x)", "f'(x)", "f''(x)"]
    for col, (name, xv, f, df, d2f, color, note) in enumerate(data):
        for row, (vals, label) in enumerate([(f, "f(x)"), (df, "f'(x)"), (d2f, "f''(x)")]):
            ax = axes[row, col]
            ax.plot(xv, np.clip(vals, -5, 5), color=color, linewidth=2)
            ax.set_title(f"{name} — {label}", fontsize=10)
            ax.axhline(0, color="k", linewidth=0.5)
            ax.axvline(0, color="k", linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-3, 3)
            if row == 2:
                ax.text(
                    0.05, 0.95, note,
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
                )

    plt.tight_layout()
    plt.savefig("activation_math_analysis.png", dpi=150, bbox_inches="tight")
    print("  Figure sauvegardée : activation_math_analysis.png")
    plt.show()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[1/3] Visualisation des réseaux 1D...")
    plot_1d_comparison()

    print("\n[2/3] Analyse mathématique des activations...")
    plot_activation_analysis()

    print("\n[3/3] Expérience d'extraction...")
    run_experiment()

    print("\nTerminé. Vérifiez les figures générées dans ce dossier.")
