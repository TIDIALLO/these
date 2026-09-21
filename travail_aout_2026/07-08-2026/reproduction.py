import numpy as np

def simuler_extraction_cryptanalytique():
    # -------------------------------------------------------------------------
    # 1. PARAMÈTRES ET CONFIGURATION DU NEURONE CIBLE (SECRET)
    # -------------------------------------------------------------------------
    # Nous modélisons un neurone cible dans un espace de dimension n = 4.
    n = 4
    np.random.seed(42)  # Fixation de la graine pour reproductibilité
    
    # Génération des poids et du biais secrets de la victime
    w_secret = np.random.uniform(-2, 2, size=n)
    b_secret = np.random.uniform(-1, 1)
    
    print("=================================================================")
    print("           ÉTAPE 1 : PARAMÈTRES SECRETS DU NEURONE CIBLE")
    print("=================================================================")
    print(f"Poids réels (w_secret) : {w_secret}")
    print(f"Biais réel (b_secret)  : {b_secret}\n")
    
    # -------------------------------------------------------------------------
    # 2. DEFINITION DE L'ORACLE DE CLASSIFICATION BOÎTE-NOIRE (HARD-LABEL)
    # -------------------------------------------------------------------------
    # L'attaquant n'a accès qu'à la fonction 'oracle' qui renvoie 0 ou 1.
    def oracle(x):
        pre_activation = np.dot(w_secret, x) + b_secret
        return 1 if pre_activation > 0 else 0

    # -------------------------------------------------------------------------
    # 3. ALGORITHME DE RECHERCHE DICHOTOMIQUE DE POINT CRITIQUE (TRANSITION)
    # -------------------------------------------------------------------------
    # Recherche un point où la pré-activation s'annule (frontière de décision).
    def trouver_point_transition(oracle, dimension, delta=1e-9):
        # A. Échantillonnage de points de départ dans des états différents
        while True:
            x1 = np.random.uniform(-5, 5, size=dimension)
            x2 = np.random.uniform(-5, 5, size=dimension)
            if oracle(x1) != oracle(x2):
                break
        
        # Identification du point actif (1) et inactif (0)
        x_on = x1 if oracle(x1) == 1 else x2
        x_off = x2 if oracle(x1) == 1 else x1
            
        # B. Dichotomie le long du segment jusqu'à la tolérance delta
        while np.linalg.norm(x_on - x_off) > delta:
            mid = (x_on + x_off) / 2.0
            if oracle(mid) == 1:
                x_on = mid
            else:
                x_off = mid
                
        # Le point milieu approxime la frontière w_secret * x + b_secret = 0
        return (x_on + x_off) / 2.0

    print("=================================================================")
    print("           ÉTAPE 2 : RECHERCHE DES POINTS DE TRANSITION")
    print("=================================================================")
    points_transition = []
    tentatives = 0
    max_tentatives = 200
    
    # Nous devons collecter n points de transition linéairement indépendants
    while len(points_transition) < n and tentatives < max_tentatives:
        candidat = trouver_point_transition(oracle, n)
        
        if len(points_transition) == 0:
            points_transition.append(candidat)
            print(f"[+] Point critique {len(points_transition)} identifié.")
        else:
            # On vérifie que le candidat est linéairement indépendant des précédents
            matrice_test = np.array(points_transition + [candidat])
            if np.linalg.matrix_rank(matrice_test) == len(matrice_test):
                points_transition.append(candidat)
                print(f"[+] Point critique {len(points_transition)} identifié (indépendant).")
        tentatives += 1

    if len(points_transition) < n:
        raise RuntimeError("Échec : Impossible de trouver n points linéairement indépendants.")
        
    X = np.array(points_transition)

    # -------------------------------------------------------------------------
    # 4. RÉSOLUTION DU SYSTÈME LINÉAIRE POUR EXTRAIRE LA SIGNATURE
    # -------------------------------------------------------------------------
    # En posant w_hat * x^(k) + b_hat = 0 avec b_hat arbitrairement fixé à -1.0 :
    # Nous devons résoudre le système X * w_hat = [3]^T
    vecteur_un = np.ones(n)
    w_hat = np.linalg.solve(X, vecteur_un)
    b_hat = -1.0
    
    print("\n=================================================================")
    print("           ÉTAPE 3 : RÉSULTATS DE L'EXTRACTION BRUTE")
    print("=================================================================")
    print(f"Poids extraits bruts (w_hat) : {w_hat}")
    print(f"Biais extrait brut (b_hat)   : {b_hat}\n")
    
    # -------------------------------------------------------------------------
    # 5. ALIGNEMENT DU SIGNE DU FACTEUR D'ÉCHELLE
    # -------------------------------------------------------------------------
    # On teste sur une entrée quelconque si les prédictions sont inversées.
    point_test = np.random.uniform(-5, 5, size=n)
    label_vrai = oracle(point_test)
    
    pre_act_extraite = np.dot(w_hat, point_test) + b_hat
    label_extrait = 1 if pre_act_extraite > 0 else 0
    
    if label_vrai != label_extrait:
        print("[!] Inversion de signe détectée. Ajustement géométrique...")
        w_hat = -w_hat
        b_hat = -b_hat
        print(f"Poids extraits ajustés (w_hat) : {w_hat}")
        print(f"Biais extrait ajusté (b_hat)   : {b_hat}\n")
    else:
        print("[+] Le signe extrait est géométriquement cohérent d'emblée !\n")

    # -------------------------------------------------------------------------
    # 6. ÉVALUATION DE LA FIDÉLITÉ ET DE LA COLINÉARITÉ
    # -------------------------------------------------------------------------
    # A. Test de fidélité empirique sur 10 000 points aléatoires
    echantillons_test = np.random.uniform(-5, 5, size=(10000, n))
    succes = 0
    for x in echantillons_test:
        l_vrai = oracle(x)
        l_pred = 1 if (np.dot(w_hat, x) + b_hat) > 0 else 0
        if l_vrai == l_pred:
            succes += 1
            
    fidelite = (succes / 10000) * 100
    
    # B. Calcul du facteur de colinéarité alpha constant
    ratios = w_secret / w_hat
    variance_ratio = np.var(ratios)
    
    print("=================================================================")
    print("           ÉTAPE 4 : ÉVALUATION DE LA FIDÉLITÉ")
    print("=================================================================")
    print(f"Ratios de proportionnalité w_secret/w_hat par dimension :")
    print(f" -> {ratios}")
    print(f"Variance du ratio (doit être proche de 0) : {variance_ratio:.2e}")
    print(f"Fidélité de prédiction obtenue sur l'ensemble de test : {fidelite:.2f} %")
    print("=================================================================")

if __name__ == "__main__":
    simuler_extraction_cryptanalytique()
