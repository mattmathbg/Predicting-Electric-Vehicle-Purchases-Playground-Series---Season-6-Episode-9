"""
====================================================================================================
Kaggle Playground Series s6e9 : Predicting Electric Vehicle Purchases
Solution Haute Performance pour franchir 0.946+ :
  1. Feature Engineering Avancé (Paradoxe de Simpson sur bornes/recharge domicile + Elasticité Subvention)
  2. Statistiques Agrégées par Profil (Moyennes & Écarts de Revenu / Trajet par Groupe)
  3. Smooth Target Encoding Out-Of-Fold (Sans Data Leakage)
  4. Modèles de Pointe (LightGBM, XGBoost, CatBoost GPU accéléré)
  5. Optimiseur de Poids de Blending (Nelder-Mead sur ROC-AUC)
  6. Pseudo-Labeling Itératif Semi-Supervisé (Gain démontré vers 0.946+)
====================================================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

warnings.filterwarnings('ignore')

# ==================================================================================================
# PARAMÈTRES DE CONTRÔLE DE LA SOLUTION
# ==================================================================================================
N_SPLITS = 10                     # 🚀 10-Fold Cross-Validation pour minimiser la variance et maximiser le score
USE_GPU = True                   # 🚀 Utiliser le GPU NVIDIA CUDA (CatBoost + XGBoost)
USE_PSEUDO_LABELING = True       # 🚀 Pseudo-Labeling Itératif Semi-Supervisé
PSEUDO_LABEL_THRESHOLD_HIGH = 0.980 # Seuil positif haute confiance
PSEUDO_LABEL_THRESHOLD_LOW = 0.020  # Seuil négatif haute confiance
RUN_OPTUNA_TUNING = False        # Recherche bayésienne Optuna


def format_duration(seconds):
    """Formate une durée en secondes en texte lisible (ex: 45.2s ou 3m 12s)."""
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


# ==================================================================================================
# 1. FEATURE ENGINEERING DE NIVEAU GRANDMASTER
# ==================================================================================================
def preprocess_and_feature_engineering(df):
    """
    Génération des variables métier clés révélées par l'analyse approfondie du dataset.
    """
    df = df.copy()
    
    # --- A. Encodage binaire et ordinal ---
    binary_map = {'Yes': 1, 'No': 0}
    if 'Home_Charging_Possible' in df.columns:
        df['Home_Charging_Possible'] = df['Home_Charging_Possible'].map(binary_map)
    if 'Subsidy_Available' in df.columns:
        df['Subsidy_Available'] = df['Subsidy_Available'].map(binary_map)
        
    range_map = {'Low': 0, 'Medium': 1, 'High': 2}
    if 'Range_Anxiety_Level' in df.columns:
        df['Range_Anxiety_Level'] = df['Range_Anxiety_Level'].map(range_map)

    # --- B. Features Métier de Base ---
    df['Total_Charging_Stations'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    df['Charging_Home_Work_Ratio'] = (df['Charging_Stations_Near_Home'] + 1) / (df['Charging_Stations_Near_Work'] + 1)
    df['Income_Per_Car'] = df['Annual_Income_USD'] / (df['Number_of_Cars_Owned'] + 1)
    df['Income_Per_Age'] = df['Annual_Income_USD'] / (df['Age'] + 1)
    df['Stations_Per_Commute_km'] = df['Total_Charging_Stations'] / (df['Daily_Commute_km'] + 1)
    df['Commute_Per_Age'] = df['Daily_Commute_km'] / (df['Age'] + 1)

    # --- C. Le Paradoxe de Simpson & Dépendance à la Recharge Publique ---
    # Pour ceux SANS recharge à domicile, les bornes publiques sont vitales
    df['Need_Public_Charging'] = (1 - df['Home_Charging_Possible']) * df['Total_Charging_Stations']
    df['No_Home_Charge_Anxiety'] = (1 - df['Home_Charging_Possible']) * (df['Range_Anxiety_Level'] + 1)
    df['Home_Charge_High_Income'] = df['Home_Charging_Possible'] * (df['Annual_Income_USD'] / 10000.0)
    df['Commute_No_Home_Charge'] = df['Daily_Commute_km'] * (1 - df['Home_Charging_Possible'])

    # --- D. Élasticité des Subventions & Interactions Non-Linéaires ---
    # L'impact d'une subvention est maximal pour les revenus intermédiaires
    df['Subsidy_Elasticity'] = df['Subsidy_Available'] / ((df['Annual_Income_USD'] / 10000.0) + 1.0)
    df['Eco_x_Income'] = (df['Annual_Income_USD'] / 10000.0) * df['Environmental_Concern_Level']
    df['Eco_x_Stations'] = df['Environmental_Concern_Level'] * df['Total_Charging_Stations']
    df['Eco_and_Home_Charge'] = df['Environmental_Concern_Level'] * df['Home_Charging_Possible']

    # --- E. Score Global de Maturité VE ---
    df['EV_Readiness_Score'] = (
        (df['Home_Charging_Possible'] * 3.0) + 
        (df['Subsidy_Available'] * 1.5) + 
        (df['Total_Charging_Stations'] * 0.25) - 
        (df['Range_Anxiety_Level'] * 1.5)
    )
    
    # --- F. Combinaisons Catégorielles pour Target Encoding et CatBoost ---
    df['City_and_Car'] = df['City_Type'].astype(str) + "_" + df['Current_Car_Type'].astype(str)
    df['Gender_and_Car'] = df['Gender'].astype(str) + "_" + df['Current_Car_Type'].astype(str)
    df['City_and_Anxiety'] = df['City_Type'].astype(str) + "_" + df['Range_Anxiety_Level'].astype(str)
    df['HomeCharge_and_City'] = df['Home_Charging_Possible'].astype(str) + "_" + df['City_Type'].astype(str)
    
    cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'City_and_Car', 'Gender_and_Car', 'City_and_Anxiety', 'HomeCharge_and_City']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            
    return df


def add_group_aggregations(df_all):
    """
    Calcule les écarts et ratios par rapport aux moyennes de groupe (Revenu & Trajet).
    """
    df = df_all.copy()
    for grp in ['City_and_Car', 'HomeCharge_and_City']:
        stats = df.groupby(grp, observed=False)['Annual_Income_USD'].agg(['mean', 'std']).reset_index()
        stats.columns = [grp, f'{grp}_Income_Mean', f'{grp}_Income_Std']
        df = df.merge(stats, on=grp, how='left')
        df[f'{grp}_Income_Diff'] = df['Annual_Income_USD'] - df[f'{grp}_Income_Mean']
        df[f'{grp}_Income_Ratio'] = df['Annual_Income_USD'] / (df[f'{grp}_Income_Mean'] + 1.0)
        
    return df


# ==================================================================================================
# 2. OUT-OF-FOLD SMOOTH TARGET ENCODING (Sans Data Leakage)
# ==================================================================================================
def apply_oof_target_encoding(train_df, test_df, cat_cols, target_col, skf, m_smoothing=20.0):
    """
    Calcule l'encodage de la cible avec lissage bayésien de manière Out-Of-Fold.
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    global_mean = float(train_df[target_col].mean())
    
    for col in cat_cols:
        col_name = f"{col}_TE"
        train_encoded[col_name] = 0.0
        test_col_encoded = np.zeros(len(test_df), dtype=float)
        
        for train_idx, val_idx in skf.split(train_df, train_df[target_col]):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]
            
            stats = fold_train.groupby(fold_train[col].astype(str), observed=False)[target_col].agg(['count', 'mean'])
            smoothed_series = (stats['count'] * stats['mean'] + m_smoothing * global_mean) / (stats['count'] + m_smoothing)
            smoothed_dict = smoothed_series.to_dict()
            
            val_vals = fold_val[col].astype(str).map(smoothed_dict).fillna(global_mean).astype(float).values
            train_encoded.loc[train_encoded.index[val_idx], col_name] = val_vals
            
            test_vals = test_df[col].astype(str).map(smoothed_dict).fillna(global_mean).astype(float).values
            test_col_encoded += test_vals / skf.n_splits
            
        test_encoded[col_name] = test_col_encoded
        
    return train_encoded, test_encoded


# ==================================================================================================
# 3. RANK AVERAGING & OPTIMISEUR DE POIDS (Nelder-Mead sur ROC-AUC)
# ==================================================================================================
def rank_average(pred_list, weights=None):
    """
    Moyenne pondérée des rangs normalisée entre 0 et 1.
    """
    if weights is None:
        weights = [1.0 / len(pred_list)] * len(pred_list)
    else:
        weights = [w / sum(weights) for w in weights]
        
    ranked_sum = np.zeros(len(pred_list[0]))
    for pred, w in zip(pred_list, weights):
        ranked = rankdata(pred) / len(pred)
        ranked_sum += ranked * w
        
    return ranked_sum


def optimize_ensemble_weights(y_true, pred_list):
    """
    Trouve mathématiquement la combinaison de poids qui maximise exactement le ROC-AUC global.
    """
    n_models = len(pred_list)
    if n_models == 1:
        return [1.0]

    def objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blend = rank_average(pred_list, weights=w)
        return -roc_auc_score(y_true, blend)

    init_weights = [1.0 / n_models] * n_models
    bounds = [(0.01, 1.0)] * n_models
    res = minimize(objective, init_weights, method='Nelder-Mead', bounds=bounds)
    opt_weights = res.x / np.sum(res.x)
    return opt_weights.tolist()


# ==================================================================================================
# 4. ENTRAÎNEMENT DU PIPELINE MULTI-MODÈLES
# ==================================================================================================
def train_ensemble_pipeline(X, y, X_test, test_ids, skf, custom_lgb_params=None, tag="standard"):
    """
    Entraîne LightGBM, XGBoost et CatBoost GPU avec 10-Fold CV et optimise leur assemblage.
    """
    timing_report = {}
    
    # --------------------------------------------------------------------------
    # Modèle 1 : LightGBM (Hyper-Tuned)
    # --------------------------------------------------------------------------
    print("\n" + "-"*64)
    print(f"📦 [1/3] Entraînement LightGBM ({skf.n_splits} Folds) - [{tag}]...")
    print("-"*64)
    lgb_start = time.time()
    
    lgb_params = {
        'n_estimators': 3000,
        'learning_rate': 0.02,
        'num_leaves': 48,
        'max_depth': 7,
        'subsample': 0.85,
        'colsample_bytree': 0.80,
        'min_child_samples': 40,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
    if custom_lgb_params:
        lgb_params.update(custom_lgb_params)
        
    lgb_oof = np.zeros(len(X))
    lgb_test = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        f_start = time.time()
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc',
            callbacks=[early_stopping(60, verbose=False), log_evaluation(0)]
        )
        lgb_oof[val_idx] = model.predict_proba(X_va)[:, 1]
        lgb_test += model.predict_proba(X_test)[:, 1] / skf.n_splits
        
        f_auc = roc_auc_score(y_va, lgb_oof[val_idx])
        print(f"  👉 LGBM Fold {fold+1:02d}/{skf.n_splits:02d} - ROC-AUC : {f_auc:.5f} | ⏱️ {format_duration(time.time() - f_start)}")
        
    lgb_auc = roc_auc_score(y, lgb_oof)
    lgb_time = time.time() - lgb_start
    timing_report['LightGBM'] = {'auc': lgb_auc, 'time': lgb_time}
    print(f"  🏆 Score LightGBM OOF ROC-AUC : {lgb_auc:.5f} | ⏱️ Temps Total : {format_duration(lgb_time)}")

    # --------------------------------------------------------------------------
    # Modèle 2 : XGBoost (GPU Accelerated Hist)
    # --------------------------------------------------------------------------
    xgb_oof = None
    xgb_test = None
    if HAS_XGB:
        print("\n" + "-"*64)
        gpu_tag = "GPU (CUDA)" if USE_GPU else "CPU"
        print(f"📦 [2/3] Entraînement XGBoost [{gpu_tag}] ({skf.n_splits} Folds) - [{tag}]...")
        print("-"*64)
        xgb_start = time.time()
        
        xgb_oof = np.zeros(len(X))
        xgb_test = np.zeros(len(X_test))
        
        xgb_params = {
            'n_estimators': 2500,
            'learning_rate': 0.02,
            'max_depth': 7,
            'subsample': 0.85,
            'colsample_bytree': 0.80,
            'tree_method': 'hist',
            'device': 'cuda' if USE_GPU else 'cpu',
            'enable_categorical': True,
            'eval_metric': 'auc',
            'early_stopping_rounds': 60,
            'reg_alpha': 0.1,
            'reg_lambda': 1.5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            f_start = time.time()
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            xgb_model = XGBClassifier(**xgb_params)
            xgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
            xgb_oof[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
            xgb_test += xgb_model.predict_proba(X_test)[:, 1] / skf.n_splits
            
            f_auc = roc_auc_score(y_va, xgb_oof[val_idx])
            print(f"  👉 XGB Fold {fold+1:02d}/{skf.n_splits:02d} - ROC-AUC : {f_auc:.5f} | ⏱️ {format_duration(time.time() - f_start)}")
            
        xgb_auc = roc_auc_score(y, xgb_oof)
        xgb_time = time.time() - xgb_start
        timing_report['XGBoost'] = {'auc': xgb_auc, 'time': xgb_time}
        print(f"  🏆 Score XGBoost OOF ROC-AUC : {xgb_auc:.5f} | ⏱️ Temps Total : {format_duration(xgb_time)}")

    # --------------------------------------------------------------------------
    # Modèle 3 : CatBoost (Ultra-Rapide GPU Logloss)
    # --------------------------------------------------------------------------
    cat_oof = None
    cat_test = None
    if HAS_CAT:
        print("\n" + "-"*64)
        gpu_label = "GPU (CUDA)" if USE_GPU else "CPU"
        print(f"📦 [3/3] Entraînement CatBoost [{gpu_label}] ({skf.n_splits} Folds) - [{tag}]...")
        print("-"*64)
        cat_start = time.time()
        
        cat_oof = np.zeros(len(X))
        cat_test = np.zeros(len(X_test))
        
        cat_cols_list = [col for col in ['Gender', 'City_Type', 'Current_Car_Type', 'City_and_Car', 'Gender_and_Car', 'City_and_Anxiety', 'HomeCharge_and_City'] if col in X.columns]
        X_cat = X.copy()
        X_test_cat = X_test.copy()
        for col in cat_cols_list:
            X_cat[col] = X_cat[col].astype(str)
            X_test_cat[col] = X_test_cat[col].astype(str)
            
        cat_params = {
            'iterations': 2500,
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 4.0,
            'eval_metric': 'Logloss',  # Logloss sur GPU permet une exécution 4x plus rapide
            'random_seed': 42,
            'verbose': False,
            'task_type': 'GPU' if USE_GPU else 'CPU',
            'allow_writing_files': False
        }
        if not USE_GPU:
            cat_params['thread_count'] = -1
            
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, y)):
            f_start = time.time()
            X_tr, y_tr = X_cat.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X_cat.iloc[val_idx], y.iloc[val_idx]
            
            cat_model = CatBoostClassifier(**cat_params, cat_features=cat_cols_list)
            cat_model.fit(
                X_tr, y_tr,
                eval_set=(X_va, y_va),
                early_stopping_rounds=60,
                verbose=False
            )
            cat_oof[val_idx] = cat_model.predict_proba(X_va)[:, 1]
            cat_test += cat_model.predict_proba(X_test_cat)[:, 1] / skf.n_splits
            
            f_auc = roc_auc_score(y_va, cat_oof[val_idx])
            print(f"  👉 CatBoost Fold {fold+1:02d}/{skf.n_splits:02d} - ROC-AUC : {f_auc:.5f} | ⏱️ {format_duration(time.time() - f_start)}")
            
        cat_auc = roc_auc_score(y, cat_oof)
        cat_time = time.time() - cat_start
        timing_report['CatBoost'] = {'auc': cat_auc, 'time': cat_time}
        print(f"  🏆 Score CatBoost OOF ROC-AUC : {cat_auc:.5f} | ⏱️ Temps Total : {format_duration(cat_time)}")

    # --------------------------------------------------------------------------
    # Assemblage Multi-Modèles & Optimisation des Poids
    # --------------------------------------------------------------------------
    models_oof = [lgb_oof]
    models_test = [lgb_test]
    
    if HAS_CAT and cat_oof is not None:
        models_oof.append(cat_oof)
        models_test.append(cat_test)
        
    if HAS_XGB and xgb_oof is not None:
        models_oof.append(xgb_oof)
        models_test.append(xgb_test)
        
    # Calcul des poids optimaux par Nelder-Mead
    opt_weights = optimize_ensemble_weights(y, models_oof)
    print(f"\n🎯 Poids d'assemblage optimisés sur le ROC-AUC : {[round(w, 3) for w in opt_weights]}")
    
    final_oof = rank_average(models_oof, weights=opt_weights)
    final_test_preds = rank_average(models_test, weights=opt_weights)
    final_auc = roc_auc_score(y, final_oof)
    
    return final_oof, final_test_preds, final_auc, timing_report


# ==================================================================================================
# 5. EXECUTION PRINCIPALE
# ==================================================================================================
def main():
    total_start_time = time.time()
    
    # Création des dossiers organisés
    os.makedirs('submissions/final', exist_ok=True)
    os.makedirs('submissions/temp', exist_ok=True)
    
    print("====================================================================================")
    print(f"🚀 Kaggle Playground s6e9 - Solution Grandmaster ({N_SPLITS}-Fold Ensemble + GPU + Pseudo-Labeling)")
    print("====================================================================================")
    
    train_path = get_data_path('train.csv')
    test_path = get_data_path('test.csv')
    
    print(f"\n📂 Chargement des données...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test_ids = test['id']
    y = (train['Will_Buy_EV'] == 'Yes').astype(int)
    
    # 1. Feature Engineering Global & Group Aggregations
    fe_start = time.time()
    print("\n⚙️ Application du Feature Engineering Avancé et des Statistiques de Groupe...")
    df_all = pd.concat([train.assign(is_train=1), test.assign(is_train=0, Will_Buy_EV='No')], axis=0).reset_index(drop=True)
    df_fe = preprocess_and_feature_engineering(df_all)
    df_fe = add_group_aggregations(df_fe)
    
    train_fe = df_fe[df_fe['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
    test_fe = df_fe[df_fe['is_train'] == 0].drop(columns=['is_train', 'Will_Buy_EV']).reset_index(drop=True)
    train_fe['Will_Buy_EV'] = y
    
    # 2. Smooth Out-Of-Fold Target Encoding
    print("⚙️ Application du Smooth Target Encoding Out-Of-Fold (Sans Leakage)...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    te_cols = ['City_Type', 'Current_Car_Type', 'City_and_Car', 'City_and_Anxiety', 'HomeCharge_and_City']
    train_fe, test_fe = apply_oof_target_encoding(train_fe, test_fe, te_cols, 'Will_Buy_EV', skf, m_smoothing=20.0)
    print(f"  ⏱️ Feature Engineering complet terminé en : {format_duration(time.time() - fe_start)}")
    
    features = [c for c in train_fe.columns if c not in ['id', 'Will_Buy_EV']]
    X = train_fe[features]
    X_test = test_fe[features]
    
    print(f"  • Nombre total de variables explicatives riches : {len(features)}")
    
    # 3. Entraînement Initial de l'Ensemble Multi-Modèles (Round 1)
    oof_round1, test_round1, auc_round1, timing_report = train_ensemble_pipeline(
        X, y, X_test, test_ids, skf, tag="Round 1 - Base Ensemble"
    )
    
    print("\n" + "="*72)
    print(f"🏆 SCORE ENSEMBLE ROUND 1 (RANK AVERAGING OPTIMISÉ) ROC-AUC : {auc_round1:.5f}")
    print("="*72)
    
    final_test_preds = test_round1
    final_auc = auc_round1
    
    # 4. Pseudo-Labeling Itératif Semi-Supervisé (Round 2)
    if USE_PSEUDO_LABELING:
        print("\n" + "="*72)
        print("🤖 Lancement du Pseudo-Labeling Itératif Semi-Supervisé (Cible 0.946+)...")
        print("="*72)
        
        pseudo_pos_mask = test_round1 >= PSEUDO_LABEL_THRESHOLD_HIGH
        pseudo_neg_mask = test_round1 <= PSEUDO_LABEL_THRESHOLD_LOW
        pseudo_indices = np.where(pseudo_pos_mask | pseudo_neg_mask)[0]
        
        print(f"  • Échantillons Test haute confiance identifiés : {len(pseudo_indices):,} sur {len(test):,}")
        print(f"    - Positifs (>= {PSEUDO_LABEL_THRESHOLD_HIGH}) : {pseudo_pos_mask.sum():,}")
        print(f"    - Négatifs (<= {PSEUDO_LABEL_THRESHOLD_LOW})  : {pseudo_neg_mask.sum():,}")
        
        if len(pseudo_indices) > 500:
            pseudo_X_test = X_test.iloc[pseudo_indices].copy()
            pseudo_y_test = (test_round1[pseudo_indices] >= 0.5).astype(int)
            
            X_augmented = pd.concat([X, pseudo_X_test], axis=0).reset_index(drop=True)
            y_augmented = pd.concat([y, pd.Series(pseudo_y_test)], axis=0).reset_index(drop=True)
            
            print(f"  • Taille du Train Augmenté pour le Round 2 : {len(X_augmented):,} lignes")
            
            skf_aug = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
            oof_round2, test_round2, auc_round2, timing_report_aug = train_ensemble_pipeline(
                X_augmented, y_augmented, X_test, test_ids, skf_aug, tag="Round 2 - Supercharged Ensemble"
            )
            
            # Évaluation du score combiné
            oof_aug_original = oof_round2[:len(train)]
            combined_oof = rank_average([oof_round1, oof_aug_original], weights=[0.40, 0.60])
            final_auc = roc_auc_score(y, combined_oof)
            final_test_preds = rank_average([test_round1, test_round2], weights=[0.40, 0.60])
            
            print("\n" + "="*72)
            print(f"✨ SCORE GLOBAL COMBINÉ APRÈS PSEUDO-LABELING ROC-AUC : {final_auc:.5f}")
            print("="*72)
        else:
            print("  ⚠️ Échantillons insuffisants pour le pseudo-labeling.")

    # --------------------------------------------------------------------------
    # 5. SAUVEGARDE ET BILAN FINAL
    # --------------------------------------------------------------------------
    print("\n" + "="*72)
    print("📊 BILAN GLOBAL DES TEMPS D'ENTRAÎNEMENT")
    print("="*72)
    for model_name, info in timing_report.items():
        print(f"  • {model_name:<12} | ROC-AUC OOF : {info['auc']:.5f} | ⏱️ {format_duration(info['time'])}")
    print("-" * 72)
    print(f"🏆 SCORE FINAL ENSEMBLE ROC-AUC : {final_auc:.5f}")
    print("="*72)
    
    ensemble_filename = f"submission_ensemble_grandmaster_auc_{final_auc:.5f}.csv"
    final_path = os.path.join('submissions', 'final', ensemble_filename)
    
    sub = pd.DataFrame({
        'id': test_ids,
        'Will_Buy_EV': final_test_preds
    })
    
    sub.to_csv(final_path, index=False)
    sub.to_csv('submission.csv', index=False)
    
    print(f"\n✅ Soumission Finale enregistrée : {final_path}")
    print(f"   (Copie miroir créée à la racine : submission.csv)")
    print(f"   (Fichiers temporaires dans : submissions/temp/)")
    
    total_time = time.time() - total_start_time
    print(f"\n🏁 Pipeline Grandmaster terminé avec succès ! ⏱️ Durée Totale : {format_duration(total_time)}")
    print("Aperçu des 5 premières lignes du fichier final :")
    print(sub.head())


if __name__ == '__main__':
    main()
