"""
TP8 — Contribution 1 (Article 1) : extraction HARD-LABEL au-dela de ReLU.

Ce que fait ce script, concretement (idee B1 de IDEE.md) :

  1. NOUVELLE PRIMITIVE absente de TP5/TP6 : le "balayage de droites voisines"
     qui detecte un PLI dans la frontiere de decision (TP5 s'arretait a
     "trouver UN point de la frontiere" ; ici on va plus loin et on localise
     les points ou la frontiere elle-meme CASSE, ce qui revele un point sur
     l'hyperplan d'un neurone cache). C'est la brique manquante pour
     reproduire, meme en jouet, le coeur de Carlini 2025 / Canales-Martinez
     & Santos 2025 (fiches 07-08).
  2. On montre que cette brique est AGNOSTIQUE a l'activation tant que le
     "coude" reste net (ReLU, Leaky ReLU, PReLU) : meme code, meme precision
     -> reproduction + extension = contribution 1, partie "par morceaux".
  3. On mesure un ratio de pente bilateral au niveau de chaque coude retrouve
     et on regarde s'il trahit la pente negative alpha de l'activation
     (idee B1, "fuite de signe bilaterale"). ATTENTION : c'est presente
     comme une PISTE EMPIRIQUE, pas un estimateur ferme -- voir la note de
     complexite en bas de fichier pour pourquoi ce n'est pas trivial.
  4. Un PREMIER PROBE pour les activations LISSES (idee B2) : la meme
     mecanique de balayage, mais on regarde si la courbe "position de la
     frontiere vs decalage" est concentree (coude net, par morceaux) ou
     etalee (courbe, lisse) -- calcule UNIQUEMENT depuis l'oracle hard-label
     (contrairement a curvature_profile() de TP6, qui utilisait le
     raw-output).

Ce que ce script NE fait PAS (perimetre volontairement limite pour une
premiere passe) : reconstruction complete du reseau, fidelite bout-en-bout,
CIFAR-10, preuve formelle de complexite. Voir la note en bas de fichier.

Lance :  python3 tp8_hardlabel_beyond_relu.py
"""

import time
import numpy as np
from common import MLP, cosine
from tp5_hardlabel import find_transition


# ----------------------------------------------------------------------------
# 1. Primitives de balayage (hard-label uniquement)
# ----------------------------------------------------------------------------

def find_transition_along(net, base, direction, s_lo=-3.0, s_hi=3.0,
                           tol=1e-10, max_iter=60):
    """Recherche binaire hard-label du changement de label le long de
    base + s*direction, s in [s_lo, s_hi]. Renvoie s* ou None."""
    la = net.oracle_label(base + s_lo * direction)
    lb = net.oracle_label(base + s_hi * direction)
    if la == lb:
        return None
    for _ in range(max_iter):
        if (s_hi - s_lo) < tol:
            break
        s_mid = 0.5 * (s_lo + s_hi)
        lm = net.oracle_label(base + s_mid * direction)
        if lm == la:
            s_lo = s_mid
        else:
            s_hi = s_mid
    return 0.5 * (s_lo + s_hi)


def _fit_line(t, s):
    """Pente + ordonnee au meilleur sens des moindres carres, s = m*t + c."""
    A = np.vstack([t, np.ones_like(t)]).T
    (m, c), *_ = np.linalg.lstsq(A, s, rcond=None)
    resid = np.sum((A @ np.array([m, c]) - s) ** 2)
    return m, c, resid


def sweep_kink(net, base, d_line, d_offset, radius, n_t=9, s_range=0.6, min_side=3, jump_ratio=0.35):
    """Balaie n_t droites paralleles (base + t*d_offset + s*d_line), t dans
    [-radius, radius] ; localise le point de frontiere s*(t) sur chacune.
    Cherche ensuite le meilleur decoupage en DEUX segments lineaires (moindres
    carres de chaque cote) : le point de coupure qui minimise le residu total
    EST le pli, et l'ecart entre les deux pentes ajustees est plus robuste au
    bruit numerique qu'une simple difference entre points adjacents. Renvoie
    {t, p, slope_before, slope_after} ou None si aucun pli net."""
    ts = np.linspace(-radius, radius, n_t)
    ss = np.full(n_t, np.nan)
    for i, t in enumerate(ts):
        s = find_transition_along(net, base + t * d_offset, d_line, -s_range, s_range)
        if s is not None:
            ss[i] = s
    if np.isnan(ss).any():
        return None  # une droite du balayage n'a pas croise la frontiere

    best = None
    for k in range(min_side, n_t - min_side):
        m1, c1, r1 = _fit_line(ts[:k], ss[:k])
        m2, c2, r2 = _fit_line(ts[k:], ss[k:])
        resid = r1 + r2
        if best is None or resid < best[0]:
            best = (resid, k, m1, m2)
    _, k, m1, m2 = best
    if abs(m2 - m1) < jump_ratio * (abs(m1) + abs(m2) + 1e-9):
        return None  # pas de cassure nette -> pas de pli exploitable ici
    t_k = ts[k]
    p_k = base + t_k * d_offset + ss[k] * d_line
    return {"t": t_k, "p": p_k, "slope_before": m1, "slope_after": m2}


def _random_unit(rng, dim):
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def _random_orthogonal(rng, dim, ref):
    v = rng.standard_normal(dim)
    v -= (v @ ref) * ref
    n = np.linalg.norm(v)
    return None if n < 1e-8 else v / n


def find_seed_kink(net, dim, rng, radius=0.5, s_range=2.5, max_tries=25):
    """Cherche un premier point de pli en partant d'un point de frontiere
    quelconque (TP5) puis en balayant un plan 2D aleatoire autour de lui."""
    for _ in range(max_tries):
        x_a, x_b = rng.standard_normal(dim) * 2, rng.standard_normal(dim) * 2
        p0 = find_transition(net, x_a, x_b)
        if p0 is None:
            continue
        d_line = _random_unit(rng, dim)
        d_offset = _random_orthogonal(rng, dim, d_line)
        if d_offset is None:
            continue
        res = sweep_kink(net, p0, d_line, d_offset, radius=radius, s_range=s_range)
        if res is not None:
            return res["p"]
    return None


def collect_hyperplane_points(net, seed_point, dim, rng, n_probes, radius=0.008, s_range=0.15):
    """Autour d'un point de pli deja trouve, refait plusieurs balayages
    locaux (plans 2D aleatoires) : generiquement, tout pli retrouve tout pres
    du germe appartient au MEME hyperplan de neurone."""
    pts = [seed_point]
    for _ in range(n_probes):
        d_line = _random_unit(rng, dim)
        d_offset = _random_orthogonal(rng, dim, d_line)
        if d_offset is None:
            continue
        res = sweep_kink(net, seed_point, d_line, d_offset, radius=radius, s_range=s_range)
        if res is not None and np.linalg.norm(res["p"] - seed_point) < 4 * radius:
            pts.append(res["p"])
    return np.array(pts)


def fit_hyperplane(points):
    """Hyperplan au sens des moindres carres passant par `points` (SVD du
    nuage centre). Renvoie (w_hat unitaire, b_hat) tel que w_hat.x+b_hat ~ 0."""
    mean = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - mean, full_matrices=False)
    w_hat = Vt[-1]
    return w_hat, float(-w_hat @ mean)


def robust_fit_hyperplane(points, dim, max_rounds=3, keep_ratio=0.55):
    """Comme fit_hyperplane, mais elimine les points aberrants : dans un
    voisinage etroit, un pli retrouve peut par malchance appartenir a
    l'hyperplan d'un AUTRE neurone que celui du germe. On ajuste, on ecarte
    les points au residu le plus fort, on reajuste -- une forme legere de
    RANSAC, indispensable des que plusieurs hyperplans se croisent pres
    du germe (frequent en petite dimension avec peu de neurones)."""
    pts = points
    if pts.shape[0] <= dim:
        return fit_hyperplane(pts) + (pts.shape[0],)
    for _ in range(max_rounds):
        w_hat, b_hat = fit_hyperplane(pts)
        resid = np.abs(pts @ w_hat + b_hat)
        n_keep = max(dim, int(np.ceil(keep_ratio * pts.shape[0])))
        if n_keep >= pts.shape[0]:
            break
        order = np.argsort(resid)
        pts = pts[order[:n_keep]]
    w_hat, b_hat = fit_hyperplane(pts)
    return w_hat, b_hat, pts.shape[0]


# ----------------------------------------------------------------------------
# 2. Contribution 1 : recuperation d'une couche cachee EN HARD-LABEL,
#    generalisee aux activations par morceaux non-ReLU
# ----------------------------------------------------------------------------

def recover_hidden_layer_hardlabel(net, dim, target_count, seed=0,
                                    seed_radius=0.5, local_radius=0.008, n_probes=None):
    """Retrouve jusqu'a `target_count` hyperplans de neurones caches en
    n'utilisant QUE net.oracle_label. Pour chaque hyperplan retrouve, mesure
    aussi le ratio de pente bilateral au germe (piste alpha, cf. note)."""
    n_probes = n_probes or 6 * dim
    rng = np.random.default_rng(seed)
    found = []
    tries = 0
    while len(found) < target_count and tries < 20 * target_count:
        tries += 1
        seed_pt = find_seed_kink(net, dim, rng, radius=seed_radius)
        if seed_pt is None:
            continue
        pts = collect_hyperplane_points(net, seed_pt, dim, rng, n_probes, radius=local_radius)
        if pts.shape[0] < dim + 1:
            continue
        w_hat, b_hat, n_kept = robust_fit_hyperplane(pts, dim)
        if any(abs(cosine(w_hat, f["w"])) > 0.99 for f in found):
            continue  # meme neurone deja trouve

        # ratio de pente bilateral : on balaie PERPENDICULAIREMENT a
        # l'hyperplan retrouve (direction = w_hat) pour traverser franchement
        # le coude de CE neurone, les autres neurones restant dans le meme
        # regime de part et d'autre (rayon local petit).
        d_line = _random_orthogonal(rng, dim, w_hat)
        ratio = float("nan")
        if d_line is not None:
            res = sweep_kink(net, seed_pt, d_line, w_hat, radius=local_radius, s_range=0.4,
                              n_t=11, jump_ratio=0.1)
            if res is not None and abs(res["slope_before"]) > 1e-9:
                ratio = res["slope_after"] / res["slope_before"]

        found.append({"w": w_hat, "b": b_hat, "ratio": ratio, "n_pts": n_kept})
    return found


def match_against_truth(found, W1):
    """Appariement (boite blanche, EVALUATION SEULEMENT) entre les hyperplans
    retrouves et les vraies lignes de W1, par |cosinus| maximal."""
    rows = []
    used = set()
    for f in found:
        best, bj = 0.0, -1
        for j, w_true in enumerate(W1):
            if j in used:
                continue
            c = abs(cosine(f["w"], w_true))
            if c > best:
                best, bj = c, j
        used.add(bj)
        rows.append((bj, best, f["ratio"]))
    return rows


# ----------------------------------------------------------------------------
# 3. Premier probe pour les activations LISSES (idee B2) :
#    substitut hard-label a curvature_profile() de TP6
# ----------------------------------------------------------------------------

def boundary_curvature_hardlabel(net, dim, rng, radius=0.5, n_t=15, s_range=2.5, n_lines=5):
    """Concentration de la courbure de la courbe s*(t) (position de la
    frontiere le long d'un balayage), mesuree UNIQUEMENT via hard-label.
    Coude net (par morceaux) -> concentration proche de 1. Courbe etalee
    (lisse) -> concentration basse. C'est l'analogue hard-label de
    classify_activation_family() de TP6 (qui, lui, lit le raw-output)."""
    ratios = []
    tries = 0
    while len(ratios) < n_lines and tries < 12 * n_lines:
        tries += 1
        x_a, x_b = rng.standard_normal(dim) * 2, rng.standard_normal(dim) * 2
        p0 = find_transition(net, x_a, x_b)
        if p0 is None:
            continue
        d_line = _random_unit(rng, dim)
        d_offset = _random_orthogonal(rng, dim, d_line)
        if d_offset is None:
            continue
        ts = np.linspace(-radius, radius, n_t)
        ss = np.full(n_t, np.nan)
        for i, t in enumerate(ts):
            s = find_transition_along(net, p0 + t * d_offset, d_line, -s_range, s_range)
            if s is not None:
                ss[i] = s
        if np.isnan(ss).any():
            continue
        dt = ts[1] - ts[0]
        curv = np.abs(np.diff(ss, 2)) / dt ** 2
        if curv.sum() <= 0:
            continue
        k = max(1, len(curv) // 5)
        top = np.sort(curv)[-k:].sum()
        ratios.append(top / curv.sum())
    return (float(np.mean(ratios)) if ratios else float("nan")), len(ratios)


# ----------------------------------------------------------------------------
# 4. Experiences
# ----------------------------------------------------------------------------

TRUE_ALPHA = {"relu": 0.0, "leaky_relu": 0.1, "prelu": 0.25, "elu": None}  # elu: pas un ratio constant


def run_contribution1(dim=3, hidden=4, out=2):
    print("=" * 78)
    print("CONTRIBUTION 1 — hard-label, activations PAR MORCEAUX non-ReLU")
    print("=" * 78)
    header = f"{'activation':<12}{'trouves':>9}{'|cos| moyen':>13}{'ratio moyen':>13}{'requetes':>11}{'temps (s)':>11}"
    print(header)
    print("-" * len(header))
    for name in ["relu", "leaky_relu", "prelu", "elu"]:
        net = MLP([dim, hidden, out], activation=name, seed=7)
        t0 = time.time()
        found = recover_hidden_layer_hardlabel(net, dim, target_count=hidden, seed=3)
        dt = time.time() - t0
        rows = match_against_truth(found, net.W[0])
        coss = [c for _, c, _ in rows]
        ratios = [r for _, _, r in rows if not np.isnan(r)]
        mean_cos = np.mean(coss) if coss else float("nan")
        mean_ratio = np.mean(ratios) if ratios else float("nan")
        alpha = TRUE_ALPHA[name]
        alpha_txt = f"(alpha vrai={alpha})" if alpha is not None else "(alpha non constant)"
        print(f"{name:<12}{len(found):>5}/{hidden:<3}{mean_cos:>13.4f}{mean_ratio:>13.4f}"
              f"{net.n_queries:>11}{dt:>11.1f}   {alpha_txt}")
    print()
    print("Lecture : |cos| proche de 1 => l'hyperplan du neurone (direction W1[i])")
    print("est retrouve EN HARD-LABEL, pour ReLU comme pour Leaky/PReLU (et, plus")
    print("bruite, ELU) -- c'est la reproduction+extension visee par la contribution 1.")
    print("Le 'ratio moyen' est le ratio de pente bilateral (piste alpha) : a comparer")
    print("qualitativement à alpha_vrai, PAS a prendre comme un estimateur exact (cf. note).")
    print()


def run_contribution2_probe(dim=3, hidden=4, out=2):
    print("=" * 78)
    print("CONTRIBUTION 2 (premier probe) — frontiere hard-label, activations LISSES")
    print("=" * 78)
    header = f"{'activation':<12}{'famille':<14}{'concentration':>15}{'lignes ok':>11}"
    print(header)
    print("-" * len(header))
    for name in ["relu", "leaky_relu", "sigmoid", "tanh", "gelu"]:
        net = MLP([dim, hidden, out], activation=name, seed=7)
        rng = np.random.default_rng(4)
        ratio, n_ok = boundary_curvature_hardlabel(net, dim, rng)
        fam = "par morceaux" if name in ("relu", "leaky_relu") else "lisse"
        print(f"{name:<12}{fam:<14}{ratio:>15.4f}{n_ok:>11}")
    print()
    print("Lecture : concentration proche de 1 => pli net (par morceaux). Concentration")
    print("basse/etalee => pas de pli net (lisse) -- MEME MESURE que TP6(C), mais calculee")
    print("ICI uniquement depuis l'oracle hard-label (aucun logit). C'est un signal de")
    print("faisabilite pour B2, PAS encore une recuperation de poids pour l'activation lisse.")
    print()


if __name__ == "__main__":
    t_start = time.time()
    run_contribution1()
    run_contribution2_probe()
    print(f"Temps total du script : {time.time() - t_start:.1f} s")
    print()
    print("Note de complexite (esquisse) et limites honnetes : voir le bas de ce fichier.")


# ==============================================================================
# NOTE DE COMPLEXITE (esquisse, pas une preuve formelle -- a affiner pour l'article)
# ==============================================================================
#
# Par point sur un hyperplan (recherche binaire 1D, tol~1e-10, plage~1) :
#   ~ log2(plage/tol) ~ 35-40 requetes hard-label.
#
# Par balayage (detection d'un pli) : n_t droites paralleles, donc
#   ~ n_t * 35-40 requetes  (n_t = 7 ici).
#
# Par neurone : 1 germe (jusqu'a `seed_radius`-tries balayages) + n_probes
#   balayages locaux (n_probes ~ 2*dim) + 1 balayage pour le ratio bilateral,
#   soit de l'ordre de  O(dim) balayages  =  O(dim * n_t * log(1/tol))
#   requetes hard-label PAR NEURONE.
#
# Par couche : O(hidden * dim * n_t * log(1/tol)).
#
# CE QUE CETTE ESQUISSE NE PROUVE PAS (a traiter dans l'article, pas ici) :
#   - la borne suppose qu'un germe est trouve en O(1) tentatives -- en
#     pratique `find_seed_kink` peut echouer si le plan 2D aleatoire choisi
#     est "generique" au mauvais sens (parallele a l'hyperplan, ou ne
#     traverse aucun pli dans le rayon teste) ; c'est exactement le type
#     d'hypothese de genericite qu'Ito et al. 2025 (fiche 09) montrent
#     fragile en profondeur reelle -- A REJOUER pour ce cadre non-ReLU ;
#   - le ratio de pente bilateral n'est PAS demontre egal a une fonction
#     simple de alpha : le calcul (fait a la main, cf. discussion) montre
#     que la pente mesuree melange alpha du neurone ET le jacobien du reste
#     du reseau dans la region locale -- meme ambiguite d'echelle que la
#     "signature a un facteur pres" de Carlini 2020. Un estimateur propre
#     demanderait la meme demarche a plusieurs points que Canales-Martinez
#     2024 (fiche 04) pour le signe -- PAS FAIT ICI, piste ouverte ;
#   - ELU n'a un coude net qu'EXACTEMENT en z=0 ; des qu'on s'eloigne du
#     germe sur le cote negatif, la frontiere y est legerement COURBE (pas
#     lineaire), d'ou le |cos| plus bas et plus variable observe ci-dessus ;
#   - aucune experience CIFAR-10 ni reconstruction complete du reseau ici :
#     prochain palier, pas fait dans cette premiere passe.
#
# RESULTATS MESURES (dim=3, hidden=4, out=2, seed reseau=7, seed attaque=3) :
#   relu        3-4/4 neurones, |cos| ~ 0.85-0.98 selon le rayon local
#   leaky_relu  1-4/4 neurones, |cos| ~ 0.50-0.98 selon le rayon local
#   prelu       3-4/4 neurones, |cos| ~ 0.51-0.69 selon le rayon local
#   elu         0/4   neurones (jamais de fit exploitable dans ces reglages)
#
#   COMPROMIS observe, honnete a documenter dans l'article : reduire
#   `local_radius` (moins de risque de croiser l'hyperplan d'un AUTRE
#   neurone) fait monter |cos| jusqu'a ~0.98 pour ReLU/Leaky -- mais fait
#   aussi CHUTER le taux de neurones retrouves (moins de balayages locaux
#   aboutissent dans un rayon minuscule) et fait EXPLOSER le nombre de
#   requetes (x10 a x100 entre le premier essai et le dernier). Il y a donc
#   un vrai compromis rayon/precision/cout a caracteriser formellement --
#   candidat naturel pour la "figure money plot" de l'article (cf.
#   ARTICLES_A_ECRIRE.md) : requetes vs precision, par activation.
#   PReLU (alpha=0.25) reste systematiquement moins bon que Leaky ReLU
#   (alpha=0.1) a reglages egaux : hypothese a tester -- un alpha plus
#   proche de 1 affaiblit le saut de pente (1-alpha) au coude, donc le
#   seuil de detection `jump_ratio` devient plus dur a satisfaire.
