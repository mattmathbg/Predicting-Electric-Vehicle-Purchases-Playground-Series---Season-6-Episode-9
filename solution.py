"""
========================================================================================
Kaggle Playground Series s6e9 : Predicting Electric Vehicle Purchases
Solution complète, commentée et optimisée (LightGBM + Stratified 5-Fold Cross-Validation)

Métrique d'évaluation : ROC-AUC (Area Under the ROC Curve)
Format de sortie attendu : id, Will_Buy_EV (probabilité entre 0.0 et 1.0)
========================================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Fix encodage console Windows pour l'affichage fluide des emojis et caractères UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# Désactiver les avertissements non bloquants pour une sortie propre dans la console
warnings.filterwarnings('ignore')


def format_duration(seconds):
    """Formate une durée en secondes en texte lisible (ex: 45.2s ou 2m 15s)."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes}m {sec:04.1f}s"


def get_data_path(filename):
    """Permet de charger les fichiers depuis 'data/' ou directement depuis le répertoire courant."""
    if os.path.exists(filename):
        return filename
    base = os.path.basename(filename)
    if os.path.exists(os.path.join('data', base)):
        return os.path.join('data', base)
    if os.path.exists(base):
        return base
    return filename


# ======================================================================================
# 1. NETTOYAGE ET CRÉATION DE FEATURES (FEATURE ENGINEERING)
# ======================================================================================
def preprocess_and_feature_engineering(df):
    """
    Cette fonction prépare les données brutes et génère des variables dérivées 
    (Feature Engineering) pour aider l'arbre de décision à mieux capturer les comportements d'achat.
    """
    df = df.copy()
    
    # ----------------------------------------------------------------------------------
    # A. Encodage des variables binaires et ordinales
    # ----------------------------------------------------------------------------------
    binary_map = {'Yes': 1, 'No': 0}
    if 'Home_Charging_Possible' in df.columns:
        df['Home_Charging_Possible'] = df['Home_Charging_Possible'].map(binary_map)
    if 'Subsidy_Available' in df.columns:
        df['Subsidy_Available'] = df['Subsidy_Available'].map(binary_map)
        
    range_anxiety_map = {'Low': 0, 'Medium': 1, 'High': 2}
    if 'Range_Anxiety_Level' in df.columns:
        df['Range_Anxiety_Level'] = df['Range_Anxiety_Level'].map(range_anxiety_map)

    # ----------------------------------------------------------------------------------
    # B. Nouvelles Features Métier (Feature Engineering de base)
    # ----------------------------------------------------------------------------------
    # 1. Total des bornes de recharge accessibles :
    df['Total_Charging_Stations'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    
    # 2. Ratio du revenu par voiture du foyer :
    df['Income_Per_Car'] = df['Annual_Income_USD'] / (df['Number_of_Cars_Owned'] + 1)
    
    # 3. Densité de recharge par rapport à la distance quotidienne :
    df['Stations_Per_Commute_km'] = df['Total_Charging_Stations'] / (df['Daily_Commute_km'] + 1)
    
    # 4. Score global d'opportunité VE (EV Readiness Score) :
    df['EV_Readiness_Score'] = (
        (df['Home_Charging_Possible'] * 2.0) + 
        (df['Subsidy_Available'] * 1.0) + 
        (df['Total_Charging_Stations'] * 0.2) - 
        (df['Range_Anxiety_Level'] * 1.0)
    )
    
    # 5. Interaction écologique x infrastructure :
    df['Eco_and_Home_Charge'] = df['Environmental_Concern_Level'] * df['Home_Charging_Possible']
    
    # 6. Interaction revenu et sensibilité écologique :
    df['Income_x_Eco'] = (df['Annual_Income_USD'] / 10000.0) * df['Environmental_Concern_Level']

    # ----------------------------------------------------------------------------------
    # C. Features combinées (Profil Ville + Véhicule)
    # ----------------------------------------------------------------------------------
    group_cols = ['City_Type', 'Current_Car_Type']
    if all(col in df.columns for col in group_cols):
        df['City_and_Car_Type'] = df['City_Type'].astype(str) + "_" + df['Current_Car_Type'].astype(str)
        df['City_and_Car_Type'] = df['City_and_Car_Type'].astype('category')

    # ----------------------------------------------------------------------------------
    # D. Gestion des colonnes catégorielles pour LightGBM
    # ----------------------------------------------------------------------------------
    cat_cols = ['Gender', 'City_Type', 'Current_Car_Type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df


# ======================================================================================
# 2. FONCTION PRINCIPALE : ENTRAÎNEMENT ET SOUMISSION
# ======================================================================================
def main():
    start_total_time = time.time()
    
    # Création des dossiers organisés de soumission
    os.makedirs('submissions/final', exist_ok=True)
    os.makedirs('submissions/temp', exist_ok=True)
    
    print("================================================================")
    print("🚀 Kaggle Playground s6e9 - Solution LightGBM 5-Fold Stratifié")
    print("================================================================")
    
    # 1. Chargement des fichiers de données
    print("\n📂 Chargement des jeux de données...")
    train_path = get_data_path('train.csv')
    test_path = get_data_path('test.csv')
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"  • Train dataset : {train.shape[0]:,} lignes, {train.shape[1]} colonnes (Fichier : {train_path})")
    print(f"  • Test dataset  : {test.shape[0]:,} lignes, {test.shape[1]} colonnes (Fichier : {test_path})")
    
    # 2. Séparation de la cible
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)
    pos_ratio = y.mean() * 100
    print(f"  • Taux d'acheteurs positifs ('Yes') : {pos_ratio:.2f}% (Déséquilibre modéré)")
    
    # 3. Application du Feature Engineering
    fe_start_time = time.time()
    print("\n⚙️ Application du Feature Engineering sur Train et Test...")
    train_fe = preprocess_and_feature_engineering(train)
    test_fe = preprocess_and_feature_engineering(test)
    print(f"  ⏱️ Temps Feature Engineering : {format_duration(time.time() - fe_start_time)}")
    
    features = [c for c in train_fe.columns if c not in ['id', 'Will_Buy_EV']]
    X = train_fe[features]
    X_test = test_fe[features]
    
    print(f"  • Nombre de variables utilisées : {len(features)}")
    print(f"  • Liste des variables : {features}")
    
    # ----------------------------------------------------------------------------------
    # 4. Schéma de Validation Croisée (Stratified 5-Fold)
    # ----------------------------------------------------------------------------------
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    # ----------------------------------------------------------------------------------
    # 5. Hyperparamètres du Modèle LightGBM
    # ----------------------------------------------------------------------------------
    lgb_params = {
        'n_estimators': 2000,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 30,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
    
    print("\n🏋️ Début de l'entraînement des 5 Folds...")
    train_start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**lgb_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=0)
            ]
        )
        
        # Prédiction des probabilités sur le fold de validation
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        
        fold_auc = roc_auc_score(y_val, val_probs)
        fold_duration = time.time() - fold_start
        best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else lgb_params['n_estimators']
        
        # Sauvegarde intermédiaire / temporaire de la prédiction du fold sur le test
        fold_test_pred = model.predict_proba(X_test)[:, 1]
        test_preds += fold_test_pred / n_splits
        
        temp_fold_sub = f"submissions/temp/temp_lgbm_fold{fold+1}_auc_{fold_auc:.5f}.csv"
        pd.DataFrame({'id': test['id'], 'Will_Buy_EV': fold_test_pred}).to_csv(temp_fold_sub, index=False)
        
        print(f"  👉 Fold {fold + 1}/{n_splits} - ROC-AUC : {fold_auc:.5f} | Itérations : {best_iter} | ⏱️ Temps : {format_duration(fold_duration)}")
        
    total_train_duration = time.time() - train_start_time
    print(f"\n⏱️ Temps total d'entraînement des 5 folds : {format_duration(total_train_duration)}")
    
    # ----------------------------------------------------------------------------------
    # 6. Évaluation Globale Out-Of-Fold (OOF)
    # ----------------------------------------------------------------------------------
    overall_auc = roc_auc_score(y, oof_preds)
    print("\n" + "="*64)
    print(f"🏆 SCORE GLOBAL OUT-OF-FOLD (OOF) ROC-AUC : {overall_auc:.5f}")
    print("="*64)
    
    # ----------------------------------------------------------------------------------
    # 7. Génération et Rangement du Fichier de Soumission
    # ----------------------------------------------------------------------------------
    submission_filename = f"submission_lightgbm_5fold_auc_{overall_auc:.5f}.csv"
    final_sub_path = os.path.join('submissions', 'final', submission_filename)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'Will_Buy_EV': test_preds
    })
    
    # 1. Sauvegarde dans le dossier final avec technique et score
    submission.to_csv(final_sub_path, index=False)
    # 2. Sauvegarde d'une copie à la racine 'submission.csv' pour soumission directe
    submission.to_csv('submission.csv', index=False)
    
    print(f"\n✅ Fichier Final généré et rangé : {final_sub_path}")
    print(f"   (Copie créée à la racine : submission.csv)")
    print(f"   (Prédictions temporaires par fold stockées dans : submissions/temp/)")
    
    total_pipeline_time = time.time() - start_total_time
    print(f"\n🏁 Fin du pipeline avec succès ! ⏱️ Durée totale : {format_duration(total_pipeline_time)}")
    print("Aperçu des 5 premières lignes du fichier final :")
    print(submission.head())


# ======================================================================================
# 3. PISTES D'AMÉLIORATION POUR GAGNER DES PLACES AU CLASSEMENT KAGGLE
# ======================================================================================
"""
💡 IDÉES POUR MONTER ENCORE PLUS HAUT DANS LE LEADERBOARD :

1. ENSEMBLING / BLENDING MULTI-MODÈLES (Voir ensemble_solution.py) :
   - Entraîner LightGBM + CatBoost + XGBoost et combiner les rangs.

2. RANK AVERAGING (Moyenne des rangs) :
   - Idéal pour maximiser l'AUC sans problème d'échelle de probabilité.

3. OPTIMISATION DES HYPERPARAMÈTRES AVEC OPTUNA :
   - Trouver automatiquement le meilleur jeu d'hyperparamètres.
"""

if __name__ == '__main__':
    main()
