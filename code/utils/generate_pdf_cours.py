"""
Générateur de PDF pour les cours de thèse
Tidiane DIALLO — EPT 2026
Utiliser: py -3.14 generate_pdf_cours.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, PageBreak, HRFlowable,
                                 KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus.flowables import Flowable
import os

# ── Couleurs de la charte ────────────────────────────────────────────────────
BLEU_FONCE   = HexColor('#1a3a6b')
BLEU_MOYEN   = HexColor('#2563eb')
BLEU_CLAIR   = HexColor('#dbeafe')
VERT         = HexColor('#16a34a')
VERT_CLAIR   = HexColor('#dcfce7')
ROUGE        = HexColor('#dc2626')
ORANGE       = HexColor('#ea580c')
ORANGE_CLAIR = HexColor('#ffedd5')
GRIS_CLAIR   = HexColor('#f8fafc')
GRIS_MOYEN   = HexColor('#e2e8f0')
GRIS_TEXTE   = HexColor('#374151')

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'cours')

# ── Styles ────────────────────────────────────────────────────────────────────
def build_styles():
    styles = getSampleStyleSheet()

    custom = {
        'Titre': ParagraphStyle('Titre',
            fontSize=28, textColor=white, fontName='Helvetica-Bold',
            alignment=TA_CENTER, spaceAfter=6, leading=34),
        'SousTitre': ParagraphStyle('SousTitre',
            fontSize=14, textColor=BLEU_CLAIR, fontName='Helvetica',
            alignment=TA_CENTER, spaceAfter=4),
        'H1': ParagraphStyle('H1',
            fontSize=18, textColor=white, fontName='Helvetica-Bold',
            alignment=TA_LEFT, spaceBefore=10, spaceAfter=6, leading=22),
        'H2': ParagraphStyle('H2',
            fontSize=14, textColor=BLEU_FONCE, fontName='Helvetica-Bold',
            spaceBefore=12, spaceAfter=4, leading=18),
        'H3': ParagraphStyle('H3',
            fontSize=12, textColor=BLEU_MOYEN, fontName='Helvetica-Bold',
            spaceBefore=8, spaceAfter=3, leading=16),
        'Corps': ParagraphStyle('Corps',
            fontSize=10.5, textColor=GRIS_TEXTE, fontName='Helvetica',
            alignment=TA_JUSTIFY, spaceAfter=6, leading=16),
        'Code': ParagraphStyle('Code',
            fontSize=8.5, textColor=HexColor('#1e293b'), fontName='Courier',
            spaceAfter=4, leading=12, leftIndent=10),
        'Bullet': ParagraphStyle('Bullet',
            fontSize=10.5, textColor=GRIS_TEXTE, fontName='Helvetica',
            spaceAfter=4, leading=15, leftIndent=15, bulletIndent=5),
        'Info': ParagraphStyle('Info',
            fontSize=10, textColor=BLEU_FONCE, fontName='Helvetica-Oblique',
            alignment=TA_CENTER, spaceAfter=4, leading=14),
        'Formule': ParagraphStyle('Formule',
            fontSize=11, textColor=BLEU_FONCE, fontName='Courier-Bold',
            alignment=TA_CENTER, spaceAfter=6, spaceBefore=6, leading=16),
    }
    return custom


# ── Blocs utilitaires ─────────────────────────────────────────────────────────
def entete_module(titre, sous_titre, styles):
    """Bloc d'en-tête coloré pour chaque module"""
    table = Table([[Paragraph(titre, styles['Titre']),
                    Paragraph(sous_titre, styles['SousTitre'])]],
                  colWidths=[13*cm, 5.5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BLEU_FONCE),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [BLEU_FONCE]),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 14),
        ('RIGHTPADDING',  (0,0), (-1,-1), 10),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ROUNDEDCORNERS', (0,0), (-1,-1), 8),
    ]))
    return table


def section_coloree(titre, styles, couleur=BLEU_MOYEN):
    """Barre de section colorée"""
    table = Table([[Paragraph(titre, styles['H1'])]],
                  colWidths=[18.5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), couleur),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('ROUNDEDCORNERS', (0,0), (-1,-1), 5),
    ]))
    return table


def bloc_code(lignes, styles):
    """Bloc de code avec fond gris"""
    contenu = '\n'.join(lignes)
    table = Table([[Paragraph(contenu.replace('\n','<br/>'), styles['Code'])]],
                  colWidths=[18.5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 1, HexColor('#cbd5e1')),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    return table


def encadre(titre, texte, styles, bg=ORANGE_CLAIR, border=ORANGE):
    """Encadré info/important"""
    contenu = [Paragraph(f'<b>{titre}</b>', styles['H3']),
               Paragraph(texte, styles['Corps'])]
    table = Table([[contenu]], colWidths=[18.5*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('BOX', (0,0), (-1,-1), 2, border),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 12),
    ]))
    return table


def tableau_simple(headers, rows, styles, col_widths=None):
    data = [[Paragraph(f'<b>{h}</b>', styles['Corps']) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), styles['Corps']) for c in row])
    if col_widths is None:
        col_widths = [18.5*cm / len(headers)] * len(headers)
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BLEU_FONCE),
        ('TEXTCOLOR',  (0,0), (-1,0), white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, BLEU_CLAIR]),
        ('GRID', (0,0), (-1,-1), 0.5, GRIS_MOYEN),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# PDF 1 — MODULE 1: Maths essentielles pour le Deep Learning
# ═══════════════════════════════════════════════════════════════════════════════
def creer_pdf_maths(output_path, styles):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []
    S = styles

    # ── Couverture ──────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(entete_module('MODULE 1', 'Maths pour le Deep Learning', S))
    story.append(Spacer(1, 0.5*cm))

    infos = [
        ['Doctorant', 'Tidiane DIALLO'],
        ['Directeur', 'Pr. Abdoul Aziz Ciss — EPT'],
        ['Période', 'Semaines 1-2 | Avril 2026'],
        ['Niveau', 'Débutant → Bases solides'],
        ['Durée estimée', '6-8 heures'],
    ]
    t_info = Table(infos, colWidths=[5*cm, 13*cm])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), BLEU_CLAIR),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, GRIS_MOYEN),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 0.5*cm))

    # ── Objectifs ───────────────────────────────────────────────────────────────
    story.append(section_coloree('Objectifs du module', S))
    story.append(Spacer(1, 0.3*cm))
    objectifs = [
        'Comprendre les vecteurs et matrices (briques du DNN)',
        'Maîtriser la dérivée et le gradient (base de l\'apprentissage)',
        'Comprendre la descente de gradient (comment le réseau apprend)',
        'Calculer une fonction de perte (comment mesurer l\'erreur)',
        'Savoir faire ces calculs en Python/NumPy',
    ]
    for o in objectifs:
        story.append(Paragraph(f'• {o}', S['Corps']))
    story.append(Spacer(1, 0.4*cm))

    # ── Chapitre 1: Vecteurs ─────────────────────────────────────────────────
    story.append(section_coloree('Chapitre 1 — Vecteurs & Matrices', S))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('1.1 Vecteur', S['H2']))
    story.append(Paragraph(
        'Un vecteur est une liste ordonnée de nombres. Dans un DNN, un vecteur '
        'représente les valeurs d\'entrée (ex: les pixels d\'une image aplatie).',
        S['Corps']))
    story.append(Paragraph('Notation: x = [x₁, x₂, ..., xₙ]', S['Formule']))
    story.append(bloc_code([
        'import numpy as np',
        '',
        '# Vecteur = 1 image de 3 pixels',
        'x = np.array([0.5, 0.8, 0.2])',
        'print("Taille:", x.shape)    # (3,)',
        'print("2ème valeur:", x[1])  # 0.8',
    ], S))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('1.2 Matrice et Multiplication matricielle', S['H2']))
    story.append(Paragraph(
        'Une matrice est un tableau 2D de nombres. Dans un DNN, la matrice W '
        'contient les poids qui transforment les entrées en activations. '
        'C\'est l\'opération la plus répétée dans un réseau de neurones.',
        S['Corps']))
    story.append(Paragraph('z = W · x + b', S['Formule']))
    story.append(Paragraph(
        'Où W = matrice des poids, x = vecteur d\'entrée, b = vecteur de biais, '
        'z = vecteur de sortie (avant activation).', S['Corps']))
    story.append(bloc_code([
        'W = np.array([[1, 0, 2],   # matrice 2×3',
        '              [3, 1, 0]])',
        'x = np.array([1, 2, 3])    # vecteur 3×1',
        '',
        'z = np.dot(W, x)  # produit matriciel',
        'print(z)           # [7, 5]',
        '',
        '# Vérification à la main:',
        '# z[0] = 1×1 + 0×2 + 2×3 = 7',
        '# z[1] = 3×1 + 1×2 + 0×3 = 5',
    ], S))
    story.append(Spacer(1, 0.3*cm))

    story.append(encadre(
        'Lien avec ta thèse',
        'Chaque couche d\'un DNN effectue: z = W·x + b. '
        'Voler un réseau de neurones = retrouver les matrices W et les biais b '
        'de chaque couche. C\'est exactement l\'objectif d\'une attaque d\'extraction!',
        S, ORANGE_CLAIR, ORANGE))
    story.append(Spacer(1, 0.4*cm))

    # ── Chapitre 2: Dérivées ─────────────────────────────────────────────────
    story.append(section_coloree('Chapitre 2 — Dérivées & Gradient', S, VERT))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('2.1 Dérivée d\'une fonction', S['H2']))
    story.append(Paragraph(
        'La dérivée d\'une fonction en un point mesure son taux de variation: '
        'est-ce que la fonction monte ou descend, et à quelle vitesse?',
        S['Corps']))

    dervs = [
        ['Fonction f(x)', 'Dérivée f\'(x)', 'Utilisée pour'],
        ['xⁿ', 'n · xⁿ⁻¹', 'Polynômes'],
        ['eˣ', 'eˣ', 'Activation exponentielle'],
        ['ln(x)', '1/x', 'Log-vraisemblance'],
        ['sigmoid(x)', 'sigmoid(x)·(1-sigmoid(x))', 'Activation sigmoïde'],
        ['ReLU: max(0,x)', '0 si x<0, 1 si x>0', 'Activation ReLU (thèse!)'],
    ]
    story.append(tableau_simple(
        dervs[0], dervs[1:], S,
        col_widths=[5*cm, 7*cm, 6*cm]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph('2.2 Gradient (dérivée multivariable)', S['H2']))
    story.append(Paragraph(
        'Le gradient ∇f est un vecteur qui contient les dérivées partielles par rapport '
        'à chaque variable. Il pointe dans la direction de la montée maximale. '
        'En DNN, on calcule le gradient de la fonction de perte par rapport à chaque poids.',
        S['Corps']))
    story.append(Paragraph('∇f(x,y) = [∂f/∂x, ∂f/∂y]', S['Formule']))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('2.3 Descente de gradient', S['H2']))
    story.append(Paragraph(
        'La descente de gradient est l\'algorithme central d\'entraînement d\'un DNN. '
        'On met à jour les poids dans la direction OPPOSÉE au gradient pour réduire la perte.',
        S['Corps']))
    story.append(Paragraph('w ← w − α · ∇L(w)', S['Formule']))
    story.append(Paragraph(
        'α (alpha) = learning rate = taille du pas (valeur typique: 0.001 à 0.01)',
        S['Info']))
    story.append(bloc_code([
        '# Descente de gradient simple',
        'def f(x): return x**2 + 2*x + 1  # (x+1)²',
        'def df(x): return 2*x + 2         # dérivée',
        '',
        'x = 5.0   # point de départ',
        'lr = 0.1  # learning rate',
        '',
        'for i in range(20):',
        '    x = x - lr * df(x)  # ← formule de mise à jour',
        '',
        'print(f"Minimum trouvé: x={x:.4f}")  # → x ≈ -1.0',
    ], S))
    story.append(Spacer(1, 0.3*cm))

    # ── Chapitre 3: Fonctions de perte ────────────────────────────────────────
    story.append(section_coloree('Chapitre 3 — Fonctions de Perte', S, HexColor('#7c3aed')))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        'La fonction de perte (loss) mesure à quel point le réseau se trompe. '
        'L\'entraînement consiste à minimiser cette perte.',
        S['Corps']))

    pertes = [
        ['Nom', 'Formule', 'Utilisation'],
        ['MSE (régression)', 'L = (1/n)·Σ(y-ŷ)²', 'Prédire des valeurs réelles'],
        ['Cross-Entropy', 'L = -Σ y·log(ŷ)', 'Classification multi-classes'],
        ['Binary Cross-Entropy', 'L = -(y·log(ŷ)+(1-y)·log(1-ŷ))', 'Classification binaire'],
    ]
    story.append(tableau_simple(pertes[0], pertes[1:], S,
                                col_widths=[5.5*cm, 7*cm, 6*cm]))
    story.append(Spacer(1, 0.4*cm))

    # ── Exercices ──────────────────────────────────────────────────────────────
    story.append(section_coloree('Exercices Pratiques', S, HexColor('#dc2626')))
    story.append(Spacer(1, 0.3*cm))

    exercices = [
        ('Exercice 1.1 — Matrices [30min]',
         'Créer une matrice W (3×4) et un vecteur x (4×1) en NumPy. Calculer z = W·x. '
         'Vérifier le résultat à la main pour la première ligne.'),
        ('Exercice 1.2 — Gradient visuel [45min]',
         'Implémenter la descente de gradient sur f(x) = x⁴ - 4x². '
         'Visualiser avec matplotlib. Trouver les deux minima locaux.'),
        ('Exercice 1.3 — Backprop à la main [1h]',
         'Calculer à la main les gradients d\'un réseau 2→3→1 avec activation sigmoid. '
         'Comparer avec le calcul automatique de PyTorch (autograd).'),
    ]
    for titre_ex, desc_ex in exercices:
        story.append(Paragraph(titre_ex, S['H3']))
        story.append(Paragraph(desc_ex, S['Corps']))
        story.append(Spacer(1, 0.2*cm))

    # ── Ressources ─────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(section_coloree('Ressources Recommandées', S))
    story.append(Spacer(1, 0.3*cm))

    ressources = [
        ['Type', 'Ressource', 'Durée', 'Lien'],
        ['Vidéo', '3Blue1Brown: Essence of Linear Algebra', '3h',
         'youtube.com/watch?v=fNk_zzaMoSs'],
        ['Vidéo', '3Blue1Brown: Neural Networks', '1h',
         'youtube.com/playlist PLZHQOb...'],
        ['Livre', 'Deep Learning - Goodfellow (gratuit)', 'Réf.',
         'deeplearningbook.org'],
        ['Code', 'NumPy Tutorial officiel', '2h',
         'numpy.org/learn'],
        ['Code', 'Karpathy: micrograd (backprop from scratch)', '2h30',
         'youtube.com/watch?v=VMj-3S1tku0'],
    ]
    story.append(tableau_simple(ressources[0], ressources[1:], S,
                                col_widths=[2.5*cm, 6*cm, 2*cm, 8*cm]))

    doc.build(story)
    print(f"PDF créé: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PDF 2 — MODULE 2: Deep Learning & Fonctions d'Activation
# ═══════════════════════════════════════════════════════════════════════════════
def creer_pdf_deeplearning(output_path, styles):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []
    S = styles

    story.append(Spacer(1, 1*cm))
    story.append(entete_module('MODULE 2', 'Deep Learning & Activations (Thèse)', S))
    story.append(Spacer(1, 0.5*cm))

    infos = [
        ['Doctorant', 'Tidiane DIALLO'],
        ['Directeur', 'Pr. Abdoul Aziz Ciss — EPT'],
        ['Période', 'Semaines 3-5 | Avril-Mai 2026'],
        ['Niveau', 'Bases → Intermédiaire'],
        ['Durée estimée', '8-10 heures'],
    ]
    t_info = Table(infos, colWidths=[5*cm, 13*cm])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), BLEU_CLAIR),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, GRIS_MOYEN),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 0.5*cm))

    # ── Chap 1: Perceptron ───────────────────────────────────────────────────
    story.append(section_coloree('Chapitre 1 — Le Perceptron', S))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        'Le perceptron est l\'unité de base d\'un réseau de neurones. Il imite '
        'le neurone biologique: reçoit des signaux, les combine, et décide de '
        's\'activer ou non.', S['Corps']))
    story.append(Paragraph('ŷ = activation(W·x + b)', S['Formule']))

    perc_data = [
        ['Composant', 'Rôle', 'Valeur typique'],
        ['x (entrées)', 'Signal reçu', 'Vecteur réel'],
        ['W (poids)', 'Importance de chaque entrée', 'Initialisé aléatoirement'],
        ['b (biais)', 'Décalage du seuil d\'activation', 'Initialisé à 0'],
        ['activation()', 'Introduit la non-linéarité', 'ReLU, GELU, Sigmoid...'],
        ['ŷ (sortie)', 'Signal émis vers la couche suivante', 'Scalaire ou vecteur'],
    ]
    story.append(tableau_simple(perc_data[0], perc_data[1:], S,
                                col_widths=[5*cm, 8*cm, 5.5*cm]))
    story.append(Spacer(1, 0.4*cm))

    # ── Chap 2: MLP ──────────────────────────────────────────────────────────
    story.append(section_coloree('Chapitre 2 — Réseau Multicouche (MLP)', S, VERT))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        'Un MLP (Multi-Layer Perceptron) empile plusieurs couches de neurones. '
        'Les couches intermédiaires s\'appellent "couches cachées". '
        'La profondeur (nombre de couches) donne son nom au "Deep Learning".',
        S['Corps']))
    story.append(bloc_code([
        'import torch.nn as nn',
        '',
        '# Architecture MLP: 784 → 256 → 128 → 10',
        'model = nn.Sequential(',
        '    nn.Linear(784, 256),  # couche 1',
        '    nn.ReLU(),            # activation',
        '    nn.Linear(256, 128),  # couche 2',
        '    nn.GELU(),            # activation GELU (thèse!)',
        '    nn.Linear(128, 10),   # couche de sortie',
        ')',
        '',
        '# Forward pass: calculer la sortie',
        'import torch',
        'x = torch.randn(1, 784)  # 1 image de 784 pixels',
        'output = model(x)',
        'print(output.shape)      # torch.Size([1, 10])',
    ], S))
    story.append(Spacer(1, 0.3*cm))

    # ── Chap 3: Activations (CŒUR THÈSE) ─────────────────────────────────────
    story.append(section_coloree('Chapitre 3 — Fonctions d\'Activation (CŒUR DE LA THÈSE)', S, ORANGE))
    story.append(Spacer(1, 0.3*cm))
    story.append(encadre(
        'Pourquoi ce chapitre est central pour ta thèse',
        'Ta thèse porte sur les ATTAQUES D\'EXTRACTION pour les réseaux au-delà de ReLU. '
        'Les méthodes existantes exploitent la structure particulière de ReLU (point de coude). '
        'Les activations modernes (GELU, SiLU) n\'ont pas ce point de coude, rendant '
        'les attaques connues inopérantes. C\'est le verrou scientifique que tu dois résoudre.',
        S, ORANGE_CLAIR, ORANGE))
    story.append(Spacer(1, 0.3*cm))

    activations = [
        ['Activation', 'Formule', 'Propriété clé', 'Utilisée dans'],
        ['ReLU', 'max(0, x)', 'Point de coude en 0 → EXPLOITABLE par attaque', 'Réseaux classiques'],
        ['GELU', 'x · Φ(x)', 'Lisse, sans discontinuité → RÉSISTANTE?', 'GPT, BERT, LLMs'],
        ['SiLU/Swish', 'x · σ(x)', 'Non-monotone, lisse → RÉSISTANTE?', 'EfficientNet, vision'],
        ['Sigmoid', '1/(1+e⁻ˣ)', 'Sortie entre [0,1]', 'Couche de sortie'],
        ['Tanh', '(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)', 'Sortie entre [-1,1]', 'RNN, LSTM'],
    ]
    story.append(tableau_simple(activations[0], activations[1:], S,
                                col_widths=[3.5*cm, 5*cm, 6.5*cm, 3.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('Propriété critique de ReLU pour les attaques', S['H2']))
    story.append(Paragraph(
        'ReLU est défini par deux régimes linéaires: f(x)=0 pour x≤0 et f(x)=x pour x>0. '
        'La frontière entre ces deux régimes (le "kink" ou point de coude) est exploitable '
        'par les attaques cryptanalytiques de Carlini et al. (2020). '
        'En envoyant des requêtes ciblées, l\'attaquant peut localiser exactement '
        'ce point de coude et ainsi déduire les paramètres du réseau.',
        S['Corps']))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('Challenge pour GELU et SiLU', S['H2']))
    story.append(Paragraph(
        'GELU = x · Φ(x) où Φ est la fonction de répartition de la loi normale. '
        'Cette fonction est infiniment dérivable (C∞), sans aucun point de coude. '
        'Il n\'y a pas de frontière binaire exploitable. '
        'Ta thèse doit trouver une nouvelle propriété mathématique exploitable, '
        'ou prouver que l\'extraction est impossible/exponentiellement coûteuse.',
        S['Corps']))

    # ── Chap 4: Entraînement ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(section_coloree('Chapitre 4 — Entraînement d\'un DNN', S, HexColor('#7c3aed')))
    story.append(Spacer(1, 0.3*cm))

    etapes = [
        ['Étape', 'Opération', 'Code PyTorch'],
        ['1. Forward pass', 'Calculer la sortie ŷ = f(x)', 'output = model(x)'],
        ['2. Calcul de la perte', 'L = loss(ŷ, y)', 'loss = criterion(output, y)'],
        ['3. Backpropagation', 'Calculer ∇L par rapport à tous les poids', 'loss.backward()'],
        ['4. Mise à jour', 'w ← w - α·∇L', 'optimizer.step()'],
        ['5. Reset gradient', 'Remettre les gradients à 0', 'optimizer.zero_grad()'],
    ]
    story.append(tableau_simple(etapes[0], etapes[1:], S,
                                col_widths=[4*cm, 7*cm, 7.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    story.append(bloc_code([
        '# Boucle d\'entraînement complète (template standard)',
        'model = MonReseau()',
        'optimizer = torch.optim.Adam(model.parameters(), lr=0.001)',
        'criterion = nn.CrossEntropyLoss()',
        '',
        'for epoch in range(num_epochs):',
        '    for x_batch, y_batch in dataloader:',
        '        optimizer.zero_grad()              # étape 5 (avant)',
        '        output = model(x_batch)            # étape 1',
        '        loss = criterion(output, y_batch)  # étape 2',
        '        loss.backward()                    # étape 3',
        '        optimizer.step()                   # étape 4',
    ], S))

    doc.build(story)
    print(f"PDF créé: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PDF 3 — MODULE 3: Introduction aux Attaques d'Extraction (Thèse)
# ═══════════════════════════════════════════════════════════════════════════════
def creer_pdf_attaques(output_path, styles):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []
    S = styles

    story.append(Spacer(1, 1*cm))
    story.append(entete_module('MODULE 3', 'Attaques d\'Extraction de DNN', S))
    story.append(Spacer(1, 0.5*cm))

    infos = [
        ['Doctorant', 'Tidiane DIALLO'],
        ['Directeur', 'Pr. Abdoul Aziz Ciss — EPT'],
        ['Période', 'Semaines 5-8 | Mai-Juin 2026'],
        ['Niveau', 'Intermédiaire → Avancé'],
        ['Durée estimée', '10-12 heures'],
    ]
    t_info = Table(infos, colWidths=[5*cm, 13*cm])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), BLEU_CLAIR),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, GRIS_MOYEN),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ]))
    story.append(t_info)
    story.append(Spacer(1, 0.5*cm))

    # ── Chap 1: Contexte ─────────────────────────────────────────────────────
    story.append(section_coloree('Chapitre 1 — Contexte et Menace', S))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        'Les DNN modernes sont déployés via des APIs (Machine Learning as a Service — MLaaS). '
        'Ces modèles représentent un investissement considérable. Une attaque d\'extraction '
        'permet de reconstruire un modèle en l\'interrogeant comme une boîte noire.',
        S['Corps']))

    menace_data = [
        ['Acteur', 'Description', 'Exemple'],
        ['Victime', 'Entreprise ayant entraîné un DNN coûteux', 'OpenAI, Google, Meta'],
        ['Attaquant', 'Veut voler le modèle sans payer', 'Concurrent, hackeur'],
        ['Moyen', 'API publique du modèle', 'ChatGPT API, Google AI'],
        ['Objectif', 'Reconstruire le modèle avec peu de requêtes', 'f̂ ≈ f avec coût minimal'],
    ]
    story.append(tableau_simple(menace_data[0], menace_data[1:], S,
                                col_widths=[4*cm, 8*cm, 6.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    # ── Chap 2: Types d'attaques ──────────────────────────────────────────────
    story.append(section_coloree('Chapitre 2 — Taxonomie des Attaques', S, VERT))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('2.1 Selon le type de sortie observable', S['H2']))
    types_output = [
        ['Type', 'Description', 'Exemple de réponse API', 'Difficulté'],
        ['Soft-label', 'API retourne les probabilités', '[0.85, 0.10, 0.05]', 'Facile'],
        ['Hard-label', 'API retourne seulement la classe', '"chat"', 'Difficile'],
        ['Score', 'API retourne un score continu', '0.923', 'Moyen'],
    ]
    story.append(tableau_simple(types_output[0], types_output[1:], S,
                                col_widths=[3*cm, 6*cm, 5*cm, 4.5*cm]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('2.2 Selon l\'objectif de l\'attaquant', S['H2']))
    types_obj = [
        ['Objectif', 'Description', 'Résultat attendu'],
        ['Extraction exacte', 'Copier les poids exacts W, b', 'f̂(x) = f(x) pour tout x'],
        ['Extraction fonctionnelle', 'Imiter le comportement', 'f̂(x) ≈ f(x) sur distribution'],
        ['Extraction de propriétés', 'Voler architecture/hyperparams', 'Connaître nb couches, activations'],
    ]
    story.append(tableau_simple(types_obj[0], types_obj[1:], S,
                                col_widths=[5*cm, 7*cm, 6.5*cm]))
    story.append(Spacer(1, 0.4*cm))

    # ── Chap 3: Carlini 2020 ─────────────────────────────────────────────────
    story.append(section_coloree('Chapitre 3 — Carlini et al. CRYPTO 2020 (Article fondateur)', S, HexColor('#dc2626')))
    story.append(Spacer(1, 0.3*cm))
    story.append(encadre(
        'Référence complète',
        'Carlini, N., Jagielski, M., & Mironov, I. (2020). '
        '"Cryptanalytic Extraction of Neural Network Models." '
        'Advances in Cryptology — CRYPTO 2020. '
        'ArXiv: 1910.00866',
        S, BLEU_CLAIR, BLEU_MOYEN))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph('Résultat principal', S['H2']))
    story.append(Paragraph(
        'Pour un réseau ReLU à L couches et n neurones par couche, '
        'il est possible de reconstruire tous les paramètres (W, b) de manière exacte '
        'en temps polynomial O(poly(n, L)) en utilisant les différentielles du réseau.',
        S['Corps']))

    story.append(Paragraph('Intuition de la méthode (version simplifiée)', S['H2']))
    methode = [
        ['Étape', 'Opération', 'Résultat'],
        ['1. Choisir direction', 'Choisir vecteur d\'entrée x et direction d', '—'],
        ['2. Interpoler', 'Calculer f(x + t·d) pour t ∈ [-ε, ε]', 'Courbe linéaire par morceaux'],
        ['3. Trouver kink', 'Détecter le point de non-différentiabilité', 'Position d\'un neurone ReLU'],
        ['4. Extraire poids', 'Calculer W et b à partir du kink', 'Paramètres d\'un neurone'],
        ['5. Répéter', 'Pour chaque neurone, chaque couche', 'Réseau complet extrait'],
    ]
    story.append(tableau_simple(methode[0], methode[1:], S,
                                col_widths=[3.5*cm, 8*cm, 7*cm]))
    story.append(Spacer(1, 0.3*cm))

    story.append(encadre(
        'Pourquoi cela marche pour ReLU mais pas pour GELU',
        'ReLU crée une courbe linéaire par morceaux: f(x+td) est linéaire sur chaque segment, '
        'avec des kinks aux frontières des neurones. Ces kinks sont détectables avec précision. '
        'GELU est une courbe LISSE (infiniment dérivable): pas de kinks, pas de morceaux linéaires. '
        'La méthode de Carlini ne peut pas s\'appliquer directement. '
        'C\'est le verrou central de ta thèse.',
        S, ORANGE_CLAIR, ORANGE))

    # ── Chap 4: Articles à lire ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(section_coloree('Chapitre 4 — Articles Essentiels à Lire', S))
    story.append(Spacer(1, 0.3*cm))

    articles = [
        ['Priorité', 'Article', 'Venue', 'Pourquoi le lire'],
        ['★★★', 'Carlini et al. (2020): Cryptanalytic Extraction', 'CRYPTO 2020',
         'FONDATEUR — Base de ta thèse'],
        ['★★★', 'Canales-Martínez et al. (2024): Polynomial Time Extraction',
         'EUROCRYPT 2024', 'SUITE DIRECTE — Amélioration majeure'],
        ['★★★', 'Carlini et al. (2025): Hard-Label Extraction',
         'EUROCRYPT 2025', 'ÉTAT DE L\'ART — Hard-label setting'],
        ['★★', 'Jagielski et al. (2020): High Accuracy Model Stealing',
         'IEEE S&P 2020', 'Vue d\'ensemble model stealing'],
        ['★★', 'Tramèr et al. (2016): Stealing Machine Learning Models',
         'USENIX 2016', 'PREMIER article sur model stealing'],
        ['★', 'Papernot et al. (2017): Practical Black-Box Attacks',
         'AsiaCCS 2017', 'Attaques adversariales + extraction'],
    ]
    story.append(tableau_simple(articles[0], articles[1:], S,
                                col_widths=[2*cm, 7*cm, 3.5*cm, 6*cm]))

    doc.build(story)
    print(f"PDF créé: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    styles = build_styles()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Génération des PDFs de cours...\n")

    creer_pdf_maths(
        os.path.join(OUTPUT_DIR, 'MODULE_01_Maths_DL.pdf'), styles)

    creer_pdf_deeplearning(
        os.path.join(OUTPUT_DIR, 'MODULE_02_DeepLearning_Activations.pdf'), styles)

    creer_pdf_attaques(
        os.path.join(OUTPUT_DIR, 'MODULE_03_Attaques_Extraction.pdf'), styles)

    print("\nTous les PDFs ont été générés dans: cours/")
    print("  - MODULE_01_Maths_DL.pdf")
    print("  - MODULE_02_DeepLearning_Activations.pdf")
    print("  - MODULE_03_Attaques_Extraction.pdf")
