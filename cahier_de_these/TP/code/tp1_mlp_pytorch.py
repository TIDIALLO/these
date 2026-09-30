"""
TP1 — Construire, ENTRAINER et interroger un MLP (PyTorch + variante TensorFlow).

But : te mettre en selle en deep learning. On entraine un petit classifieur, puis
on l'interroge des DEUX manieres : raw-output (logits) et hard-label (argmax).
Ce sont exactement les deux oracles que tu attaqueras ensuite.

Necessite PyTorch :  pip install torch
(La variante TensorFlow est donnee en commentaire en bas, pour le code de Carlini.)

Lance :  python3 tp1_mlp_pytorch.py
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def make_data(n=1000, dim=4, n_classes=3, seed=0):
    """Petit jeu de classification synthetique (melange de gaussiennes)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, dim)) * 3
    y = rng.integers(0, n_classes, size=n)
    X = centers[y] + rng.standard_normal((n, dim))
    return X.astype(np.float32), y.astype(np.int64)


def run_pytorch():
    torch.manual_seed(0)
    dim, n_classes = 4, 3
    X, y = make_data(dim=dim, n_classes=n_classes)
    Xt, yt = torch.tensor(X), torch.tensor(y)

    # --- definir le MLP. Changer nn.ReLU() pour ta these : nn.PReLU(), nn.GELU()... ---
    model = nn.Sequential(
        nn.Linear(dim, 16), nn.ReLU(),
        nn.Linear(16, 8),  nn.ReLU(),
        nn.Linear(8, n_classes),          # couche de sortie : pas d'activation
    )
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # --- entrainement ---
    for epoch in range(200):
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    acc = (model(Xt).argmax(1) == yt).float().mean().item()
    print(f"PyTorch — perte finale = {loss.item():.4f}, precision = {acc:.3f}")

    # --- les deux oracles ---
    x = torch.tensor(X[0:1])
    with torch.no_grad():
        logits = model(x)                 # RAW-OUTPUT (ce que voient f.03-05, f.10-11)
        label = logits.argmax(1).item()   # HARD-LABEL (ce que voient f.06-09)
    print(f"  raw-output (logits) : {logits.numpy().round(3)}")
    print(f"  hard-label (classe) : {label}")
    print()
    print("=> Tu sais entrainer une cible et l'interroger des 2 facons.")
    print("   Etapes suivantes : tp2 (points critiques) -> tp3 (signature) -> ...")
    print()
    print("EXERCICE THESE : refais l'entrainement avec nn.PReLU() puis nn.GELU().")
    print("La precision change-t-elle ? (souvent peu). Toute la difficulte du sujet")
    print("est cote ATTAQUE, pas cote entrainement.")


if __name__ == "__main__":
    if HAS_TORCH:
        run_pytorch()
    else:
        print("PyTorch non installe. Installe-le :  pip install torch")
        print("En attendant, les TP2-TP7 fonctionnent en NumPy pur (cible MLP maison).")
        print("Le module common.py fournit deja un MLP entrainable a la main si besoin.")

# ---------------------------------------------------------------------------
# VARIANTE TENSORFLOW / KERAS (pour relire/reutiliser le code de Carlini 2020)
# ---------------------------------------------------------------------------
# import tensorflow as tf
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
#     tf.keras.layers.Dense(8,  activation='relu'),
#     tf.keras.layers.Dense(3),                      # sortie sans activation
# ])
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# model.fit(X, y, epochs=200, verbose=0)
# logits = model.predict(X[0:1])        # raw-output
# label  = int(logits.argmax(1)[0])     # hard-label
