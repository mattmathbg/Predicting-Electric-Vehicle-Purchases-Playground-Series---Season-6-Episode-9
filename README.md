# 🚗 Kaggle Playground Series - Season 6, Episode 9
## 🔋 Predicting Electric Vehicle Purchases (ROC-AUC ~0.946+)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/competitions/playground-series-s6e9)
[![Frameworks](https://img.shields.io/badge/Frameworks-LightGBM%20%7C%20XGBoost%20%7C%20CatBoost-green.svg)]()
[![Metric](https://img.shields.io/badge/Metric-ROC--AUC-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattmathbg/Predicting-Electric-Vehicle-Purchases-Playground-Series---Season-6-Episode-9/blob/main/ensemble_solution.ipynb)

> **Objectif :** Prédire avec précision la probabilité d'achat d'un véhicule électrique (`Will_Buy_EV`) à partir des caractéristiques socio-démographiques, des habitudes de transport et de l'infrastructure de recharge.

---

### 🚀 Exécution Directe sur Google Colab (GPU Gratuit)

Cliquez sur le badge ci-dessous pour ouvrir et exécuter le notebook directement sur **Google Colab** avec accélération GPU (T4 / A100) :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mattmathbg/Predicting-Electric-Vehicle-Purchases-Playground-Series---Season-6-Episode-9/blob/main/ensemble_solution.ipynb)

1. **Activer le GPU** : `Exécution` > `Modifier le type d'exécution` > Sélectionner **GPU T4**.
2. **Déposer les fichiers** : Glissez `train.csv` et `test.csv` dans le volet des fichiers Colab à gauche (ou montez Google Drive).
3. **Lancer l'entraînement** : `Exécution` > `Tout exécuter`. L'ensemble 10-Fold (LGBM + XGBoost GPU + CatBoost GPU + Pseudo-labeling) s'exécute automatiquement en ~5-10 minutes.


---

## 📊 Résumé des Performances

| Pipeline / Modèle | CV Strategy | ROC-AUC (OOF) | Remarques |
|---|---|---|---|
| **Baseline LightGBM** | 5-Fold Stratified | `0.9405` | Modèle initial avec features de base |
| **LightGBM Feature Eng.** | 5-Fold Stratified | `0.9419` | + Interaction bornes & recharge domicile |
| **XGBoost (GPU Hist)** | 5-Fold Stratified | `0.9417` | Regularisation $L_1/L_2$ |
| **CatBoost (GPU CUDA)** | 5-Fold Stratified | `0.9418` | Encodage natif des catégories |
| **Ensemble (Rank Averaging)** | 5-Fold Stratified | `0.9421` | Fusion pondérée des rangs |
| **Ensemble + Pseudo-Labeling** | 5-Fold Stratified | `0.9442` | Auto-apprentissage semi-supervisé |
| **Grandmaster Pipeline (10-Fold + Full Stack)** | **10-Fold Stratified** | **`0.9465+`** | **Architecture Finale Optimisée** |

---

## 🧠 Architecture de la Solution

```mermaid
flowchart TD
    A[Train & Test Data] --> B[Feature Engineering Avancé]
    B --> C[Simpson's Paradox & Elasticity Features]
    C --> D[Group Aggregations & Out-Of-Fold Target Encoding]
    
    D --> E1[LightGBM 10-Fold]
    D --> E2[XGBoost GPU 10-Fold]
    D --> E3[CatBoost GPU 10-Fold]
    
    E1 --> F[Nelder-Mead Rank Averaging Optimizer]
    E2 --> F
    E3 --> F
    
    F --> G[Round 1 Predictions]
    G --> H[High-Confidence Test Pseudo-Labeling]
    H --> I[Round 2 Retraining on Augmented Dataset]
    I --> J[Final Submission CSV]
```

---

## ⚡ Les 6 Piliers Techniques

1. **Paradoxe de Simpson & Recharge Domicile** :
   * La densité des bornes publiques n'est critique que pour les personnes **sans solution de recharge à domicile**.
   * Features : `Need_Public_Charging = (1 - Home_Charging) * Total_Stations`, `No_Home_Charge_Anxiety`, `Home_Charge_High_Income`.
2. **Élasticité Économique des Subventions** :
   * L'impact d'une aide financière est maximal sur les tranches de revenus intermédiaires : `Subsidy_Elasticity = Subsidy / (Income / 10000 + 1)`.
3. **Agrégations Statistiques de Groupe** :
   * Déviation et ratios par rapport aux moyennes de groupe (`City_and_Car`, `HomeCharge_and_City`).
4. **Smooth Out-Of-Fold Target Encoding (Sans Leakage)** :
   * Encodage bayésien lissé calculé fold par fold pour les combinaisons catégorielles clés.
5. **Optimisation Mathématique des Poids de Blending (Nelder-Mead)** :
   * Recherche numérique des poids optimaux maximisant directement la métrique ROC-AUC sur les rangs.
6. **Pseudo-Labeling Itératif Semi-Supervisé** :
   * Extraction des échantillons test avec prédictions ultra-confiantes ($p \ge 0.98$ ou $p \le 0.02$) pour ré-entraîner l'ensemble et affiner la frontière de décision.

---

## 📁 Structure du Répertoire

```text
├── data/                               # Données du concours (non versionnées)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── submissions/
│   ├── final/                          # Soumissions finales prêtes pour Kaggle
│   └── temp/                           # Prédictions intermédiaires et modèles individuels
├── ensemble_solution.ipynb             # 📓 Notebook interactif optimisé Kaggle & Google Colab (GPU)
├── ensemble_solution.py                # 🚀 Pipeline Grandmaster complet (LGBM + XGB + CatBoost + Pseudo-Labeling)
├── solution.py                         # Modèle LightGBM rapide et autonome
├── requirements.txt                    # Dépendances du projet
├── .gitignore                          # Règles d'exclusion Git
└── README.md                           # Documentation du projet
```

---

## 🛠️ Utilisation : Notebook (Kaggle & Colab) ou Scripts Python

### Option A : Exécuter sur Kaggle ou Google Colab (Recommandé)
Le notebook [`ensemble_solution.ipynb`](file:///d:/Kaggle/Predicting%20Electric%20Vehicle%20Purchases/ensemble_solution.ipynb) détecte automatiquement l'environnement et le GPU :
1. **Sur Kaggle Notebooks** :
   * Créer un nouveau Notebook sur la page du concours.
   * `File` -> `Import Notebook` -> Uploader `ensemble_solution.ipynb`.
   * Dans les paramètres à droite : Activer **GPU T4 x2** ou **P100**.
   * Cliquer sur *Run All* / *Save Version (Commit)*.
2. **Sur Google Colab** :
   * Ouvrir Google Colab et uploader `ensemble_solution.ipynb`.
   * Menu `Exécution` -> `Modifier le type d'exécution` -> Choisir **GPU T4**.
   * Exécuter les cellules.

### Option B : Exécution Locale en Ligne de Commande

#### 1. Cloner le dépôt et installer les dépendances
```bash
git clone https://github.com/mattmathbg/Predicting-Electric-Vehicle-Purchases-Playground-Series---Season-6-Episode-9.git
cd Predicting-Electric-Vehicle-Purchases-Playground-Series---Season-6-Episode-9
pip install -r requirements.txt
```

#### 2. Télécharger les données
Placez les fichiers `train.csv`, `test.csv` et `sample_submission.csv` dans le dossier `data/` (ou directement à la racine).

#### 3. Exécuter la solution

* **Solution Rapide (LightGBM 5-Fold)** :
  ```bash
  python solution.py
  ```

* **Solution Grandmaster Complète (10-Fold Ensemble + GPU + Pseudo-Labeling)** :
  ```bash
  python ensemble_solution.py
  ```

Le fichier de soumission prêt pour Kaggle sera automatiquement enregistré dans :
`submissions/final/submission_ensemble_grandmaster_auc_0.94XXX.csv` *(avec une copie miroir `submission.csv` à la racine)*.

---

## 💻 Accélération Matérielle
* **GPU NVIDIA (CUDA)** : Détection et accélération automatiques pour CatBoost (`task_type='GPU'`) et XGBoost (`device='cuda'`).
* **Multi-threading CPU** : Support parallèle automatique sur tous les cœurs disponibles pour LightGBM.
