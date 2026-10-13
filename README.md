# Maintenance prédictive et Machine Learning explicable

**Français** · [English](README.en.md)

Prédire les pannes d'équipements industriels à partir de données de fonctionnement, et expliquer *pourquoi* le modèle les prédit — pas seulement *qu'il* les prédit.

**Stack** — Python · Pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn

**Rapport complet** — [PDF, 18 pages](rapport_maintenance_predictive.pdf) ([English version](predictive_maintenance_report_en.pdf))

---

## Problème

Les entreprises industrielles subissent des pannes d'équipement non planifiées qui immobilisent leurs lignes de production. Deux postes de coûts en découlent : la perte de production pendant l'immobilisation, et le surcoût d'une intervention en urgence par rapport à une maintenance planifiée.

Anticiper ces pannes permettrait de planifier les interventions. La difficulté tient à l'arbitrage : intervenir trop tôt conduit à remplacer des pièces encore fonctionnelles ; intervenir trop tard laisse survenir la panne que l'on cherchait à éviter. Cet arbitrage structure tout le projet, du choix des métriques au réglage du seuil de décision.

**Question de recherche**

> Dans quelle mesure des modèles de Machine Learning permettent-ils de détecter efficacement les risques de panne, et quels facteurs influencent le plus leurs prédictions ?

**Hypothèse testée**

> Les modèles non linéaires devraient mieux détecter les pannes que la régression logistique, mais pourraient produire davantage de faux positifs.

---

## Données

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository.

10 000 observations, 14 variables, **339 pannes (3,39 %)**. Aucune valeur manquante, aucun doublon.

Le jeu de données n'est pas versionné. Pour reproduire l'analyse, télécharger `ai4i2020.csv` depuis le lien ci-dessus et le placer dans `data/raw/`.

### Sélection des variables

Sur les 13 variables disponibles hors cible, **6 sont retenues** :

| Variable | Rôle | Motif |
|---|---|---|
| `Type` | Explicative | Qualité du produit (L / M / H) |
| `Air temperature [K]` | Explicative | Température ambiante |
| `Process temperature [K]` | Explicative | Température du process |
| `Rotational speed [rpm]` | Explicative | Vitesse de rotation |
| `Torque [Nm]` | Explicative | Couple appliqué |
| `Tool wear [min]` | Explicative | Usure cumulée de l'outil |
| `Machine failure` | **Cible** | Survenue d'une panne |
| `UDI`, `Product ID` | Exclues | Identifiants — 10 000 valeurs distinctes pour 10 000 lignes |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Exclues | **Fuite de données** |

Les cinq indicatrices `TWF`, `HDF`, `PWF`, `OSF` et `RNF` désignent le *mode* de défaillance constaté. Elles ne sont renseignées qu'après la survenue de la panne et ne seraient pas disponibles au moment de la prédiction. Les conserver produirait un modèle aux performances quasi parfaites en test et sans valeur en production.

---

## Méthode

1. **Analyse exploratoire** — distributions, corrélations, déséquilibre des classes, valeurs atypiques
2. **Préparation** — encodage de `Type` par variables indicatrices, standardisation pour la seule régression logistique
3. **Séparation train/test stratifiée** (80 / 20), taux de panne préservé dans les deux échantillons
4. **Compensation du déséquilibre** — pondération des classes, appliquée identiquement aux trois modèles
5. **Validation croisée stratifiée** en 5 blocs sur l'échantillon d'entraînement
6. **Évaluation** — précision, rappel, F1, PR-AUC, matrice de confusion
7. **Ajustement du seuil** sur prédictions hors échantillon, sans consulter le jeu de test
8. **Interprétation** par valeurs de Shapley (SHAP)
9. **Analyse des erreurs** sur faux positifs et faux négatifs

### Pourquoi la PR-AUC plutôt que l'accuracy

Avec 3,39 % de pannes, prédire systématiquement « pas de panne » suffit à atteindre **96,6 % d'accuracy** sans détecter la moindre défaillance. La PR-AUC est retenue comme métrique principale : contrairement à la ROC-AUC, elle ne bénéficie pas mécaniquement de l'abondance de vrais négatifs, et elle est indépendante du seuil de décision.

---

## Résultats

### Validation croisée (5 blocs, échantillon d'entraînement)

| Modèle | PR-AUC | Écart-type |
|---|---|---|
| Logistic Regression | 0,441 | ± 0,030 |
| Random Forest | 0,750 | ± 0,037 |
| XGBoost | 0,751 | ± 0,036 |

### Jeu de test (2 000 observations, 68 pannes) — seuil 0,50

| Modèle | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0,142 | 0,824 | 0,242 | 0,382 | 0,907 |
| Random Forest | 0,808 | 0,618 | 0,700 | 0,774 | 0,971 |
| **XGBoost** | 0,609 | 0,824 | 0,700 | **0,813** | 0,972 |
| XGBoost (seuil 0,815) | **0,828** | 0,706 | **0,762** | 0,813 | 0,972 |

![Comparaison des modèles](figures/07_comparaison_modeles.png)

### Traduction opérationnelle

| Modèle | Pannes détectées | Pannes manquées | Fausses alertes |
|---|---|---|---|
| Logistic Regression | 56 / 68 | 12 | **339** |
| Random Forest | 42 / 68 | **26** | 10 |
| XGBoost | 56 / 68 | 12 | 36 |
| XGBoost (seuil 0,815) | 48 / 68 | 20 | 10 |

![Matrices de confusion](figures/08_matrices_confusion.png)

![Courbes précision-rappel](figures/09_courbes_precision_rappel.png)

---

## Interprétation

### L'hypothèse était à moitié fausse

La première partie se vérifie : les modèles non linéaires détectent bien mieux les pannes, la PR-AUC passant de 0,382 à 0,813.

**La seconde est infirmée.** Loin de produire davantage de faux positifs, XGBoost en génère **neuf fois moins** que la régression logistique — 36 contre 339.

L'explication tient à la géométrie du problème. Les pannes surviennent dans **deux régimes opposés**, surcharge et sous-charge, situés aux extrémités de la même variable. Aucun hyperplan ne peut isoler simultanément deux régions opposées sans englober la zone centrale, où se concentrent les observations saines. La régression logistique atteint donc un bon rappel (0,824) uniquement en élargissant massivement sa zone d'alerte : elle déclenche 395 interventions pour 56 pannes réelles. Les modèles à base d'arbres partitionnent l'espace et isolent chaque régime séparément.

Ce n'est pas la « puissance » du modèle qui fait la différence, mais **l'adéquation entre la forme de sa frontière de décision et la structure du problème.**

### ROC-AUC contre PR-AUC

La régression logistique affiche une ROC-AUC de 0,907 tout en n'émettant qu'une alerte justifiée sur sept. La ROC-AUC intègre les vrais négatifs, très abondants ici, et donne une image flatteuse d'un modèle inexploitable. Sa PR-AUC, elle, tombe à 0,382. **Démonstration empirique du choix de métrique.**

### Ce que le modèle a appris

![SHAP summary plot](figures/11_shap_summary.png)

| Variable | Part de l'importance |
|---|---|
| Torque (Nm) | 30,8 % |
| Tool wear (min) | 25,1 % |
| Rotational speed (rpm) | 18,8 % |
| Air temperature (K) | 14,9 % |
| Process temperature (K) | 7,7 % |
| Type (L + M) | 2,7 % |

Les trois variables mécaniques concentrent **75 % de l'importance**.

La qualité du produit ne pèse que 2,7 %, alors que le taux de panne brut y est presque deux fois plus élevé pour les produits de qualité inférieure (3,92 % contre 2,09 %). L'effet est **médié par les conditions de fonctionnement** : les produits de faible qualité sont usinés dans des régimes plus contraignants, déjà décrits par le couple et la vitesse. Illustration de la différence entre corrélation marginale et contribution conditionnelle.

![Dépendances SHAP](figures/13_shap_dependence.png)

Les graphiques de dépendance montrent des relations franchement non linéaires :

- **Le couple dessine une courbe en U** — risque minimal autour de 40–45 Nm, croissant aux deux extrémités. Les deux régimes de défaillance apparaissent explicitement.
- **L'usure de l'outil produit un effet de seuil** — contribution plate jusqu'à 200 minutes, bascule brutale au-delà. Le modèle a appris un seuil de remplacement implicite.
- **La vitesse de rotation reproduit le U du couple, inversé**, les deux grandeurs étant mécaniquement liées.
- **La température de l'air bascule autour de 301,5 K.**

Ces formes expliquent en une figure pourquoi un modèle linéaire ne pouvait pas convenir.

### Analyse des erreurs

![Analyse des erreurs](figures/14_analyse_erreurs.png)

| | Vrais négatifs | Faux positifs | Faux négatifs | Vrais positifs |
|---|---|---|---|---|
| Couple (Nm) | 39,5 | **51,9** | 44,7 | 52,4 |
| Usure (min) | 107,5 | 94,4 | **170,3** | 146,3 |
| Vitesse (rpm) | 1 539 | 1 430 | 1 476 | 1 481 |

**Les faux négatifs sont des pannes d'usure.** Leur usure moyenne (170 min) dépasse celle des pannes détectées (146 min), mais leur couple reste ordinaire (44,7 Nm). Ce sont des défaillances par usure survenant en régime mécanique normal. Le modèle les manque parce qu'il s'appuie prioritairement sur la contrainte instantanée, alors que ces pannes relèvent d'un processus cumulatif.

**Les faux positifs sont des états à risque qui n'ont pas dégénéré.** Couple le plus élevé des quatre catégories (51,9 Nm, supérieur même aux vrais positifs), vitesse la plus basse. Ils sont **physiquement indiscernables des pannes** sur la base des variables disponibles. Du point de vue métier, une alerte sur ces points relève de la détection précoce plus que de l'erreur.

### Choix du modèle

**XGBoost** est retenu : meilleure PR-AUC (0,813 contre 0,774) et meilleur rappel à précision comparable une fois le seuil ajusté.

Réserve méthodologique : en validation croisée, Random Forest et XGBoost sont **statistiquement indiscernables** (0,750 contre 0,751, pour des écarts-types de l'ordre de 0,036). L'écart mesuré sur le jeu de test tient en partie à cet échantillon particulier et ne constitue pas une supériorité établie.

### Sur le seuil de décision

![Seuil de décision](figures/10_seuil_decision.png)

Le seuil optimal identifié (0,815) est très supérieur à la convention de 0,50, conséquence directe de la pondération des classes qui déplace les probabilités vers le haut.

L'ajustement fait passer le F1 de 0,700 à 0,762 et divise les fausses alertes par plus de trois — mais **huit pannes supplémentaires échappent à la détection**.

Le F1 pondère identiquement précision et rappel, ce qui n'a aucune justification métier. Le seuil devrait dériver du rapport entre le coût d'un arrêt subi et celui d'une intervention inutile. Si une panne coûte dix fois plus cher qu'une fausse alerte, le seuil de 0,50 reste préférable. **C'est un paramètre métier, pas une constante statistique.**

---

## Limites

**Données**

- Jeu de données **synthétique** : relations générées par règles déterministes, plus régulières que des relevés de capteurs réels soumis au bruit et à la dérive.
- **9 pannes sur 339 sans mode de défaillance identifié**, et 18 observations portant un mode `RNF` (panne aléatoire, non prédictible par construction) sans étiquette de panne. Ces cas abaissent le plafond de performance atteignable.
- **68 pannes seulement dans le jeu de test** : chaque erreur déplace le rappel de 1,5 point. Les écarts faibles entre modèles sont peu significatifs.
- **Indépendance des observations supposée.** S'il s'agissait de relevés successifs d'une même machine, la séparation aléatoire train/test produirait une fuite temporelle.

**Protocole**

- **Hyperparamètres non optimisés** — aucune recherche systématique conduite.
- **Aucune ingénierie de variables**, alors que les modes de défaillance du dataset sont définis à partir de grandeurs dérivées (puissance mécanique, écart entre températures). Les construire améliorerait vraisemblablement la détection des pannes d'usure actuellement manquées.
- **Seuil optimisé sur le F1**, faute de fonction de coût métier explicite.
- **Colinéarité forte** entre les deux températures (r ≈ 0,88) et entre couple et vitesse (r ≈ −0,88) : l'importance SHAP se répartit arbitrairement entre variables redondantes et ne mesure pas une causalité.

---

## Prolongements

1. Construire les variables dérivées (puissance mécanique, écart de températures, interaction usure × couple) et mesurer leur apport par ablation.
2. Optimiser les hyperparamètres par recherche aléatoire avec validation croisée sur la PR-AUC.
3. Remplacer le F1 par une fonction de coût métier explicite pour fixer le seuil.
4. Reproduire le protocole sur des données industrielles réelles.

---

## Structure du dépôt

```
.
├── notebooks/
│   ├── 01_eda.ipynb          # Exploration, sélection des variables, export du jeu préparé
│   └── 02_modeling.ipynb     # Modélisation, évaluation, SHAP, analyse d'erreurs
├── data/
│   ├── raw/                  # ai4i2020.csv (non versionné — voir section Données)
│   └── processed/            # Jeu préparé, généré par 01_eda.ipynb
├── figures/                  # 16 figures + tableau comparatif
├── requirements.txt
└── README.md
```

## Reproduire l'analyse

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Sur macOS, XGBoost requiert la bibliothèque OpenMP :

```bash
brew install libomp
```

Télécharger `ai4i2020.csv` depuis [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), le placer dans `data/raw/`, puis exécuter les notebooks dans l'ordre :

```bash
jupyter lab
```

`01_eda.ipynb` génère `data/processed/ai4i2020_model.csv`, consommé par `02_modeling.ipynb`.
