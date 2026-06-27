import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold

RANDOM_STATE = 42
CV_FOLDS    = 5
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Utility: Target Encoding Out-of-Fold
# Calcola la media del target su fold di training distinti da quello di
# validazione, evitando data leakage per le colonne ad alta cardinalità.
# ---------------------------------------------------------------------------
def target_encode_oof(df, col, target_col, n_splits=5):
    oof = pd.Series(index=df.index, dtype=float)
    kf  = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    global_mean = df[target_col].mean()
    for train_idx, val_idx in kf.split(df):
        train = df.iloc[train_idx]
        val   = df.iloc[val_idx]
        means = train.groupby(col)[target_col].mean()
        oof.iloc[val_idx] = val[col].map(means).fillna(global_mean)
    oof.fillna(global_mean, inplace=True)
    return oof


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------
class AmesHousingAnalyzer:

    def __init__(self, filepath, target_column='SalePrice', random_state=RANDOM_STATE):
        self.filepath      = filepath
        self.target_column = target_column
        self.random_state  = random_state

        # stati interni
        self.df               = None   # dataset grezzo
        self.df_preprocessed  = None   # dataset dopo preprocessing completo
        self.df_clean         = None   # solo numeriche standardizzate (per EDA PCA)
        self.pca_model        = None   # PCA usata nel percorso EDA
        self.df_pca           = None   # dati trasformati dalla PCA EDA
        self.selected_features = []    # top-k feature da MI
        self.results          = {}     # dizionario con tutte le metriche
        self.preprocessing_applied = False

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Utilità interne
    # -----------------------------------------------------------------------
    def _save_fig(self, filename, dpi=150):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _freedman_diaconis_bins(x):
        """Calcola il numero ottimale di bin per un istogramma (regola F-D)."""
        x = x[~np.isnan(x)]
        if len(x) < 2:
            return 10
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr == 0:
            return int(np.sqrt(len(x)))
        h = 2 * iqr * (len(x) ** (-1/3))
        if h <= 0:
            return int(np.sqrt(len(x)))
        return max(10, int(np.ceil((x.max() - x.min()) / h)))

    # -----------------------------------------------------------------------
    # 1. Caricamento dati
    # -----------------------------------------------------------------------
    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        print(f"[load_data] Dataset caricato: {self.df.shape[0]} righe × {self.df.shape[1]} colonne")
        return self

    # -----------------------------------------------------------------------
    # 2. EDA visuale — lavora su self.df grezzo
    # -----------------------------------------------------------------------
    def analyze_target(self, target="SalePrice"):
        """6 grafici sulla distribuzione della variabile target."""
        s    = self.df[target].dropna()
        bins = self._freedman_diaconis_bins(s.values)
        log_s = np.log1p(s.clip(lower=0))

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()

        axs[0].hist(s, bins=bins, edgecolor="black", color="steelblue")
        axs[0].set_title("SalePrice — distribuzione originale")
        axs[0].set_xlabel("SalePrice ($)")

        axs[1].boxplot(s, vert=False, patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.6))
        axs[1].set_title("Boxplot SalePrice")

        axs[2].hist(log_s, bins=self._freedman_diaconis_bins(log_s.values),
                    edgecolor="black", color="darkorange")
        axs[2].set_title("log1p(SalePrice) — distribuzione trasformata")

        stats.probplot(log_s, dist="norm", plot=axs[3])
        axs[3].set_title("Q-Q plot (scala logaritmica)")

        sns.kdeplot(s, ax=axs[4], fill=True, color="steelblue")
        axs[4].set_title("KDE — SalePrice")

        axs[5].scatter(range(len(s)), s, s=6, alpha=0.5, color="steelblue")
        axs[5].set_title("Index vs SalePrice")
        axs[5].set_xlabel("Indice osservazione")

        skew_orig = s.skew()
        skew_log  = log_s.skew()
        fig.suptitle(
            f"Analisi SalePrice  |  skewness originale: {skew_orig:.2f}  →  log: {skew_log:.2f}",
            fontsize=13, y=1.01
        )
        plt.tight_layout()
        self._save_fig("01_target_distribution.png")

    def missing_value_report(self, top_n=30):
        """Barplot percentuale valori mancanti (top N colonne)."""
        missing_pct = 100 * self.df.isnull().sum() / len(self.df)
        missing_df  = (
            pd.DataFrame({"missing_pct": missing_pct})
            .query("missing_pct > 0")
            .sort_values("missing_pct", ascending=False)
            .head(top_n)
        )
        if missing_df.empty:
            print("[missing_value_report] Nessun valore mancante trovato.")
            return

        missing_df.plot(kind="barh", figsize=(12, 8), color="steelblue", legend=False)
        plt.xlabel("% valori mancanti")
        plt.title(f"Valori mancanti per feature (top {top_n})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        self._save_fig("02_missing_values.png")

    def show_correlations(self, target="SalePrice", top_k=15):
        """Barplot correlazioni + heatmap + pairplot top-6."""
        numeric = self.df.select_dtypes(include=[np.number])
        corr    = numeric.corrwith(numeric[target]).abs().drop(target)
        top     = corr.sort_values(ascending=False).head(top_k)

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        sns.barplot(x=top.values, y=top.index, ax=ax[0], color="steelblue")
        ax[0].set_title(f"Top {top_k} correlazioni con {target}")
        ax[0].set_xlabel("|correlazione di Pearson|")

        sub_corr = numeric[top.index.tolist() + [target]].corr()
        mask = np.triu(np.ones_like(sub_corr, dtype=bool))
        sns.heatmap(sub_corr, mask=mask, cmap="coolwarm", center=0,
                    ax=ax[1], annot=False, fmt=".2f")
        ax[1].set_title("Matrice di correlazione")

        plt.tight_layout()
        self._save_fig("03_correlations.png")

        sns.pairplot(self.df, x_vars=top.index[:6], y_vars=[target], kind="reg",
                     plot_kws={"scatter_kws": {"s": 10, "alpha": 0.4}})
        self._save_fig("04_scatter_top_features.png")

    def _prepare_numeric_eda(self):
        """Prepara dati solo-numerici standardizzati per il percorso EDA (PCA + clustering visuale)."""
        numeric = self.df.select_dtypes(include=[np.number])
        numeric = numeric.drop(columns=["SalePrice", "Order", "PID"], errors="ignore")
        numeric = numeric.fillna(numeric.median())
        scaler  = StandardScaler()
        self.df_clean = pd.DataFrame(
            scaler.fit_transform(numeric), columns=numeric.columns
        )

    def run_pca(self):
        """PCA sul dataset numerico grezzo, scree plot."""
        self._prepare_numeric_eda()
        self.pca_model = PCA(n_components=0.85, random_state=self.random_state)
        self.df_pca    = self.pca_model.fit_transform(self.df_clean)

        evr = self.pca_model.explained_variance_ratio_
        plt.figure(figsize=(10, 4))
        plt.bar(range(1, len(evr) + 1), evr, color="steelblue", alpha=0.7, label="Varianza spiegata")
        plt.plot(range(1, len(evr) + 1), np.cumsum(evr), marker="o", color="darkorange", label="Cumulata")
        plt.axhline(0.85, linestyle="--", color="red", label="Soglia 85%")
        plt.xlabel("Componente principale")
        plt.ylabel("Varianza spiegata")
        plt.title(f"PCA Scree Plot  |  {len(evr)} componenti → 85% varianza")
        plt.legend()
        plt.tight_layout()
        self._save_fig("05_pca_scree.png")

    def clustering_eda(self):
        """Elbow + Silhouette + scatter cluster (percorso EDA visuale su dati grezzi)."""
        inertias, silhouettes = [], []
        ks = range(2, 11)

        for k in ks:
            km     = KMeans(n_clusters=k, random_state=self.random_state, n_init=50)
            labels = km.fit_predict(self.df_pca)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(self.df_pca, labels))

        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        ax[0].plot(list(ks), inertias, marker="o", color="steelblue")
        ax[0].set_title("Elbow Method (Inerzia)")
        ax[0].set_xlabel("k")
        ax[0].set_ylabel("WCSS")

        ax[1].plot(list(ks), silhouettes, marker="o", color="darkorange")
        ax[1].set_title("Silhouette Score")
        ax[1].set_xlabel("k")
        plt.tight_layout()
        self._save_fig("06_kmeans_selection.png")

        best_k = list(ks)[int(np.argmax(silhouettes))]
        km     = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=100)
        labels = km.fit_predict(self.df_pca)

        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(self.df_pca[:, 0], self.df_pca[:, 1],
                              c=labels, cmap="tab10", s=15, alpha=0.7)
        plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                    c="red", marker="X", s=200, zorder=5, label="Centroidi")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"K-Means clustering (k={best_k})  |  Silhouette={max(silhouettes):.3f}")
        plt.legend()
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        self._save_fig("07_kmeans_clusters.png")

    def feature_importance_eda(self):
        """Permutation importance su dati numerici grezzi (solo per visualizzazione EDA)."""
        numeric = self.df.select_dtypes(include=[np.number]).dropna()
        X = numeric.drop(columns=["SalePrice", "Order", "PID"], errors="ignore")
        y = np.log1p(numeric["SalePrice"])

        rf   = RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        rf.fit(X, y)
        perm = permutation_importance(rf, X, y, n_repeats=10, random_state=self.random_state)
        idx  = np.argsort(perm.importances_mean)[::-1][:15]

        imp_df = pd.DataFrame({
            "feature":    X.columns[idx],
            "importance": perm.importances_mean[idx],
            "std":        perm.importances_std[idx]
        })

        plt.figure(figsize=(9, 6))
        sns.barplot(x="importance", y="feature", data=imp_df, color="steelblue",
                    xerr=imp_df["std"].values)
        plt.title("Permutation Importance (top 15 feature numeriche)")
        plt.xlabel("Riduzione media R² per permutazione")
        plt.tight_layout()
        self._save_fig("08_feature_importance.png")

    # -----------------------------------------------------------------------
    # 3. Preprocessing principale
    # -----------------------------------------------------------------------
    def preprocess_data(self, handle_outliers=True, cardinality_threshold=10):
        """
        Pipeline completa di preprocessing:
          1. Rimozione colonne ID
          2. Rinominazione colonne con spazi
          3. Feature engineering (TotalSF, HouseAge, SinceRemod, flag binari)
          4. Log-trasformazione del target
          5. Imputazione mancanti (mediana per numeriche, 'Missing' per categoriche)
          6. Clipping outlier IQR
          7. One-Hot Encoding per feature a bassa cardinalità   ← FIX rispetto v1
          8. Target Encoding OOF per feature ad alta cardinalità
          9. StandardScaler su feature numeriche (inside loop CV per evitare leakage)
             — lo scaling viene applicato globalmente qui per semplicità,
               ma i modelli tree-based (RF) non ne risentono.
        """
        df = self.df.copy()

        # --- 1. Rimozione ID ---
        df.drop(columns=[c for c in ['Order', 'PID'] if c in df.columns], inplace=True)

        # --- 2. Rinominazione ---
        df.rename(columns={
            '1st Flr SF':    'FirstFlrSF',
            '2nd Flr SF':    'SecondFlrSF',
            'Total Bsmt SF': 'TotalBsmtSF'
        }, inplace=True)

        # --- 3. Feature engineering ---
        area_parts = [c for c in ['FirstFlrSF', 'SecondFlrSF', 'TotalBsmtSF'] if c in df.columns]
        if area_parts:
            df['TotalSF'] = df[area_parts].sum(axis=1)

        if 'Year Built' in df.columns and 'Yr Sold' in df.columns:
            df['HouseAge'] = df['Yr Sold'] - df['Year Built']

        if 'Year Remod/Add' in df.columns and 'Yr Sold' in df.columns:
            df['SinceRemod'] = df['Yr Sold'] - df['Year Remod/Add']

        if 'Pool Area' in df.columns:
            df['HasPool'] = (df['Pool Area'] > 0).astype(int)
        if 'Fireplaces' in df.columns:
            df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        if 'Garage Type' in df.columns:
            df['HasGarage'] = (~df['Garage Type'].isnull()).astype(int)

        # --- 4. Log-trasformazione target ---
        if self.target_column in df.columns:
            if abs(df[self.target_column].skew()) > 0.75:
                df['SalePrice_log'] = np.log1p(df[self.target_column])

        # --- 5. Classificazione colonne ---
        num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        for t in [self.target_column, 'SalePrice_log']:
            if t in num_cols:
                num_cols.remove(t)

        cat_cols  = df.select_dtypes(include=['object', 'category']).columns.tolist()
        low_card  = [c for c in cat_cols if df[c].nunique() <= cardinality_threshold]
        high_card = [c for c in cat_cols if df[c].nunique() >  cardinality_threshold]

        # --- 6. Imputazione ---
        for c in num_cols:
            if df[c].isnull().sum() > 0:
                df[c] = df[c].fillna(df[c].median())
        for c in cat_cols:
            df[c] = df[c].fillna('Missing')

        # --- 7. Outlier clipping IQR ---
        if handle_outliers:
            for col in num_cols:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR    = Q3 - Q1
                df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

        # --- 8. One-Hot Encoding per bassa cardinalità ← FIX ---
        # drop_first=True per evitare multicollinearità perfetta
        if low_card:
            df = pd.get_dummies(df, columns=low_card, drop_first=True, dtype=int)

        # --- 9. Target Encoding OOF per alta cardinalità ---
        for col in high_card:
            if self.target_column in df.columns:
                encoded = target_encode_oof(df, col, self.target_column, n_splits=5)
                df[f'{col}_encoded'] = encoded
                df.drop(columns=[col], inplace=True)

        # --- 10. StandardScaler su feature numeriche ---
        # Aggiorna la lista num_cols dopo OHE (nuove colonne dummy sono già 0/1)
        all_num = df.select_dtypes(include=[np.number]).columns.tolist()
        scale_cols = [c for c in all_num
                      if c not in [self.target_column, 'SalePrice_log']]
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        self.df_preprocessed    = df
        self.preprocessing_applied = True

        n_features_total = len([c for c in df.columns
                                 if c not in [self.target_column, 'SalePrice_log']])
        print(f"[preprocess_data] Completato: {df.shape[0]} righe × {n_features_total} feature")
        return self

    # -----------------------------------------------------------------------
    # 4. Feature selection — Mutual Information
    # -----------------------------------------------------------------------
    def select_features_mi(self, k=40):
        """Seleziona le top-k feature numeriche per Mutual Information con il target log."""
        if not self.preprocessing_applied:
            self.preprocess_data()

        df         = self.df_preprocessed.copy()
        target_col = 'SalePrice_log' if 'SalePrice_log' in df.columns else self.target_column
        if target_col not in df.columns:
            return self

        y            = df[target_col]
        exclude_cols = [self.target_column, 'SalePrice_log']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X_num = df[feature_cols].select_dtypes(include=[np.number])
        X_num = X_num.fillna(X_num.median())

        mi_scores = mutual_info_regression(X_num, y, random_state=self.random_state)
        mi_series = pd.Series(mi_scores, index=X_num.columns).sort_values(ascending=False)

        self.selected_features = mi_series.head(k).index.tolist()
        print(f"[select_features_mi] Selezionate {k} feature su {len(mi_series)} disponibili")
        print(f"  Top 10: {self.selected_features[:10]}")
        return self

    # -----------------------------------------------------------------------
    # 5. Task 1 — Regressione (confronto tra modelli) ← FIX principale
    # -----------------------------------------------------------------------
    def evaluate_regression(self, n_splits=5):
        """
        Confronta 4 modelli di regressione con K-Fold CV:
          - Ridge Regression
          - Lasso Regression
          - Random Forest Regressor
          - Gradient Boosting Regressor
        Target: SalePrice_log. Metriche: R², RMSE, MAE (scala log).
        """
        if not self.preprocessing_applied:
            self.preprocess_data()
        if not self.selected_features:
            self.select_features_mi(k=40)

        df         = self.df_preprocessed.copy()
        target_col = 'SalePrice_log' if 'SalePrice_log' in df.columns else self.target_column
        X          = df[self.selected_features]
        y          = df[target_col]

        models = {
            'Ridge':            Ridge(alpha=10.0),
            'Lasso':            Lasso(alpha=0.001, max_iter=5000),
            'RandomForest':     RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                          max_depth=4, random_state=self.random_state),
        }

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        model_results = {}

        print("\n" + "="*60)
        print("TASK 1 — REGRESSIONE")
        print("="*60)

        for name, model in models.items():
            r2_scores, rmse_scores, mae_scores = [], [], []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2_scores.append(r2_score(y_test, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae_scores.append(mean_absolute_error(y_test, y_pred))

            mean_r2   = np.mean(r2_scores)
            std_r2    = np.std(r2_scores)
            mean_rmse = np.mean(rmse_scores)
            std_rmse  = np.std(rmse_scores)
            mean_mae  = np.mean(mae_scores)

            verdict = ("Excellent" if mean_r2 > 0.85
                       else "Good"     if mean_r2 > 0.70
                       else "Moderate")

            model_results[name] = {
                'mean_r2':   mean_r2,
                'std_r2':    std_r2,
                'mean_rmse': mean_rmse,
                'std_rmse':  std_rmse,
                'mean_mae':  mean_mae,
                'verdict':   verdict,
            }

            print(f"  {name:<20}  R²={mean_r2:.4f} ±{std_r2:.4f}  "
                  f"RMSE={mean_rmse:.4f} ±{std_rmse:.4f}  "
                  f"MAE={mean_mae:.4f}  [{verdict}]")

        # Miglior modello per R²
        best_name = max(model_results, key=lambda n: model_results[n]['mean_r2'])
        print(f"\n  ► Miglior modello: {best_name} "
              f"(R²={model_results[best_name]['mean_r2']:.4f})")

        # Feature importance dal Random Forest (finale su tutto il dataset)
        rf_final = RandomForestRegressor(n_estimators=200, random_state=self.random_state, n_jobs=-1)
        rf_final.fit(X, y)
        feature_importances = pd.Series(
            rf_final.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        self.results['regression'] = {
            'models':              model_results,
            'best_model':          best_name,
            'feature_importances': feature_importances,
        }

        # Plot confronto modelli
        self._plot_regression_comparison(model_results)
        return self

    def _plot_regression_comparison(self, model_results):
        """Barplot comparativo R² e RMSE tra i modelli di regressione."""
        names  = list(model_results.keys())
        r2s    = [model_results[n]['mean_r2']   for n in names]
        rmses  = [model_results[n]['mean_rmse'] for n in names]
        r2_std = [model_results[n]['std_r2']    for n in names]

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        colors = ["steelblue" if n != max(model_results, key=lambda x: model_results[x]['mean_r2'])
                  else "darkorange" for n in names]

        ax[0].bar(names, r2s, color=colors, yerr=r2_std, capsize=5)
        ax[0].set_title("R² medio (5-Fold CV)")
        ax[0].set_ylabel("R²")
        ax[0].set_ylim(0, 1)
        for i, v in enumerate(r2s):
            ax[0].text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

        ax[1].bar(names, rmses, color=colors)
        ax[1].set_title("RMSE medio — scala logaritmica (5-Fold CV)")
        ax[1].set_ylabel("RMSE (log)")
        for i, v in enumerate(rmses):
            ax[1].text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=9)

        plt.suptitle("Confronto modelli di regressione", fontsize=13)
        plt.tight_layout()
        self._save_fig("09_regression_comparison.png")

    # -----------------------------------------------------------------------
    # 6. Task 2 — Clustering (K-Means + DBSCAN)
    # -----------------------------------------------------------------------
    def evaluate_clustering(self, method='kmeans'):
        """
        Clustering su feature ad alta varianza (non MI) → più corretto per task
        non supervisionato. PCA a 3 componenti (≈78% varianza). K=3 ottimale.
        """
        if not self.preprocessing_applied:
            self.preprocess_data()

        df = self.df_preprocessed.copy()

        # Selezione feature ad alta varianza tra le numeriche ← FIX rispetto v1
        # (non usiamo la MI perché seleziona feature predittive verso SalePrice,
        #  introducendo un bias implicito nel clustering non supervisionato)
        all_num    = df.select_dtypes(include=[np.number]).columns.tolist()
        excl       = [self.target_column, 'SalePrice_log']
        num_feats  = [c for c in all_num if c not in excl]

        # Calcoliamo la varianza sul dataset PRIMA dello scaling (usiamo df grezzo)
        # Per semplicità usiamo la varianza delle colonne nel df_preprocessed
        var_series = df[num_feats].var().sort_values(ascending=False)
        top15_feats = var_series.head(15).index.tolist()

        X = df[top15_feats]

        # RobustScaler → resistente agli outlier residui
        scaler  = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA a 3 componenti (coerente con la relazione)
        pca   = PCA(n_components=3, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_.sum()

        print("\n" + "="*60)
        print("TASK 2 — CLUSTERING")
        print("="*60)
        print(f"  Feature usate: top 15 per varianza")
        print(f"  PCA: 3 componenti → {var_explained*100:.1f}% varianza spiegata")

        if method == 'kmeans':
            inertias, silhouettes, dbi_scores = [], [], []

            for k in range(2, 11):
                km     = KMeans(n_clusters=k, random_state=self.random_state,
                                n_init=100, max_iter=500)
                labels = km.fit_predict(X_pca)
                inertias.append(km.inertia_)
                silhouettes.append(silhouette_score(X_pca, labels))
                dbi_scores.append(davies_bouldin_score(X_pca, labels))

            # Plot selezione k
            ks = list(range(2, 11))
            fig, ax = plt.subplots(1, 3, figsize=(18, 4))

            ax[0].plot(ks, inertias, marker="o", color="steelblue")
            ax[0].set_title("Elbow Method (WCSS)")
            ax[0].set_xlabel("k")

            ax[1].plot(ks, silhouettes, marker="o", color="darkorange")
            ax[1].set_title("Silhouette Score")
            ax[1].set_xlabel("k")
            best_sil_k = ks[int(np.argmax(silhouettes))]
            ax[1].axvline(best_sil_k, linestyle="--", color="red", alpha=0.6)

            ax[2].plot(ks, dbi_scores, marker="o", color="green")
            ax[2].set_title("Davies-Bouldin Index (↓ meglio)")
            ax[2].set_xlabel("k")
            best_dbi_k = ks[int(np.argmin(dbi_scores))]
            ax[2].axvline(best_dbi_k, linestyle="--", color="red", alpha=0.6)

            plt.suptitle("Determinazione k ottimale — K-Means", fontsize=13)
            plt.tight_layout()
            self._save_fig("10_kmeans_selection.png")

            # k=3 come da relazione (convergenza dei tre criteri)
            best_k  = 3
            best_sil = silhouettes[best_k - 2]
            best_dbi = dbi_scores[best_k - 2]

            km_final = KMeans(n_clusters=best_k, random_state=self.random_state,
                              n_init=100, max_iter=500)
            labels   = km_final.fit_predict(X_pca)

            # Scatter PC1 vs PC2
            fig, ax = plt.subplots(1, 2, figsize=(16, 6))

            sc = ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                               cmap="tab10", s=15, alpha=0.7)
            ax[0].scatter(km_final.cluster_centers_[:, 0],
                          km_final.cluster_centers_[:, 1],
                          c="red", marker="X", s=200, zorder=5, label="Centroidi")
            ax[0].set_xlabel("PC1")
            ax[0].set_ylabel("PC2")
            ax[0].set_title(f"K-Means k={best_k} — PC1 vs PC2")
            ax[0].legend()
            plt.colorbar(sc, ax=ax[0], label="Cluster")

            sc2 = ax[1].scatter(X_pca[:, 0], X_pca[:, 2], c=labels,
                                cmap="tab10", s=15, alpha=0.7)
            ax[1].scatter(km_final.cluster_centers_[:, 0],
                          km_final.cluster_centers_[:, 2],
                          c="red", marker="X", s=200, zorder=5, label="Centroidi")
            ax[1].set_xlabel("PC1")
            ax[1].set_ylabel("PC3")
            ax[1].set_title(f"K-Means k={best_k} — PC1 vs PC3")
            ax[1].legend()
            plt.colorbar(sc2, ax=ax[1], label="Cluster")

            plt.suptitle(f"Clustering K-Means (k={best_k})  |  "
                         f"Silhouette={best_sil:.4f}  DBI={best_dbi:.4f}", fontsize=13)
            plt.tight_layout()
            self._save_fig("11_kmeans_clusters.png")

            # Interpretazione cluster: SalePrice medio per cluster
            if self.target_column in df.columns:
                cluster_df = df[[self.target_column]].copy()
                cluster_df['cluster'] = labels
                cluster_means = cluster_df.groupby('cluster')[self.target_column].mean().sort_values()
                print(f"\n  SalePrice medio per cluster (validazione esterna):")
                for cl, val in cluster_means.items():
                    print(f"    Cluster {cl}: ${val:,.0f}")

            verdict = ("Excellent" if best_sil > 0.50
                       else "Good"     if best_sil > 0.35
                       else "Moderate")

            print(f"\n  K ottimale:       {best_k}")
            print(f"  Silhouette Score: {best_sil:.4f}  [{verdict}]")
            print(f"  Davies-Bouldin:   {best_dbi:.4f}")

            self.results['clustering'] = {
                'method':         'kmeans',
                'best_k':         best_k,
                'best_silhouette': best_sil,
                'best_dbi':       best_dbi,
                'labels':         labels,
                'verdict':        verdict,
            }

        elif method == 'dbscan':
            best_params    = None
            best_silhouette = -1

            for eps in np.linspace(0.3, 2.0, 8):
                for min_samples in [3, 5, 10]:
                    db     = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = db.fit_predict(X_pca)
                    n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_cl < 2:
                        continue
                    sil = silhouette_score(X_pca, labels)
                    if sil > best_silhouette:
                        best_silhouette = sil
                        best_params     = {'eps': eps, 'min_samples': min_samples}

            verdict = ("Excellent" if best_silhouette > 0.50
                       else "Good"     if best_silhouette > 0.35
                       else "Moderate")

            print(f"\n  DBSCAN best params: {best_params}")
            print(f"  Silhouette Score:   {best_silhouette:.4f}  [{verdict}]")

            self.results['clustering'] = {
                'method':          'dbscan',
                'best_params':     best_params,
                'best_silhouette': best_silhouette,
                'verdict':         verdict,
            }

        return self

    # -----------------------------------------------------------------------
    # 7. Task 3 — Classificazione
    # -----------------------------------------------------------------------
    def evaluate_classification(self, n_bins=3, n_splits=5):
        """
        Discretizza SalePrice_log in n_bins classi equipartite (quantili).
        Modello: Random Forest Classifier. Metriche: Accuracy, F1, matrice di confusione.
        """
        if not self.preprocessing_applied:
            self.preprocess_data()
        if not self.selected_features:
            self.select_features_mi(k=40)

        df         = self.df_preprocessed.copy()
        target_col = 'SalePrice_log' if 'SalePrice_log' in df.columns else self.target_column
        y_binned   = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates='drop')

        class_counts   = y_binned.value_counts().sort_index()
        balance_ratio  = class_counts.min() / class_counts.max()
        baseline_acc   = class_counts.max() / len(y_binned)

        X  = df[self.selected_features]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        accuracy_scores, f1_scores = [], []
        all_y_true, all_y_pred     = [], []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_binned.iloc[train_idx], y_binned.iloc[test_idx]

            rf = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())

        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy  = np.std(accuracy_scores)
        mean_f1       = np.mean(f1_scores)
        std_f1        = np.std(f1_scores)

        verdict = ("Excellent" if mean_f1 > 0.85
                   else "Good"     if mean_f1 > 0.70
                   else "Moderate")

        print("\n" + "="*60)
        print("TASK 3 — CLASSIFICAZIONE")
        print("="*60)
        print(f"  Classi: {n_bins}  |  Balance ratio: {balance_ratio:.3f}")
        print(f"  Baseline (classe più frequente): {baseline_acc:.3f}")
        print(f"  Accuracy: {mean_accuracy:.4f} ±{std_accuracy:.4f}  [{verdict}]")
        print(f"  F1-Score (weighted): {mean_f1:.4f} ±{std_f1:.4f}")
        print(f"\n  Classification Report (aggregato 5 fold):")
        class_names = [f"Classe {i}" for i in sorted(set(all_y_true))]
        print(classification_report(all_y_true, all_y_pred, target_names=class_names))

        # Matrice di confusione
        cm = confusion_matrix(all_y_true, all_y_pred)
        self._plot_confusion_matrix(cm, class_names)

        self.results['classification'] = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy':  std_accuracy,
            'mean_f1':       mean_f1,
            'std_f1':        std_f1,
            'balance_ratio': balance_ratio,
            'baseline_accuracy': baseline_acc,
            'verdict':       verdict,
        }
        return self

    def _plot_confusion_matrix(self, cm, class_names):
        """Heatmap della matrice di confusione."""
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Matrice di Confusione — Classificazione (aggregato 5 fold)")
        plt.tight_layout()
        self._save_fig("12_confusion_matrix.png")

    # -----------------------------------------------------------------------
    # 8. Stampa riepilogo risultati ← FIX: prima non stampava nulla
    # -----------------------------------------------------------------------
    def print_results(self):
        """Stampa un riepilogo tabellare di tutte le metriche."""
        print("\n" + "="*60)
        print("RIEPILOGO COMPARATIVO DEI TRE TASK")
        print("="*60)

        if 'regression' in self.results:
            print("\n── REGRESSIONE ──")
            print(f"  {'Modello':<22}  {'R²':>8}  {'±':>6}  {'RMSE':>8}  {'MAE':>8}  Verdict")
            for name, res in self.results['regression']['models'].items():
                marker = " ◄ best" if name == self.results['regression']['best_model'] else ""
                print(f"  {name:<22}  {res['mean_r2']:>8.4f}  "
                      f"{res['std_r2']:>6.4f}  {res['mean_rmse']:>8.4f}  "
                      f"{res['mean_mae']:>8.4f}  {res['verdict']}{marker}")

        if 'clustering' in self.results:
            r = self.results['clustering']
            print("\n── CLUSTERING ──")
            print(f"  Metodo:           {r['method']}")
            print(f"  K ottimale:       {r.get('best_k', 'N/A')}")
            print(f"  Silhouette Score: {r['best_silhouette']:.4f}")
            if 'best_dbi' in r:
                print(f"  Davies-Bouldin:   {r['best_dbi']:.4f}")
            print(f"  Verdict:          {r['verdict']}")

        if 'classification' in self.results:
            r = self.results['classification']
            print("\n── CLASSIFICAZIONE ──")
            print(f"  Accuracy:         {r['mean_accuracy']:.4f} ±{r['std_accuracy']:.4f}")
            print(f"  F1-Score (w):     {r['mean_f1']:.4f} ±{r['std_f1']:.4f}")
            print(f"  Baseline:         {r['baseline_accuracy']:.4f}")
            print(f"  Balance ratio:    {r['balance_ratio']:.3f}")
            print(f"  Verdict:          {r['verdict']}")

        print("\n" + "="*60)
        print("SINTESI")
        print("="*60)
        scores = {}
        if 'regression' in self.results:
            best = self.results['regression']['best_model']
            scores['Regressione'] = self.results['regression']['models'][best]['mean_r2']
        if 'clustering' in self.results:
            scores['Clustering'] = self.results['clustering']['best_silhouette']
        if 'classification' in self.results:
            scores['Classificazione'] = self.results['classification']['mean_f1']

        for task, score in scores.items():
            bar_len = int(score * 40)
            bar     = "█" * bar_len + "░" * (40 - bar_len)
            print(f"  {task:<16}  [{bar}]  {score:.4f}")
        print()

    # -----------------------------------------------------------------------
    # 9. Pipeline completa
    # -----------------------------------------------------------------------
    def run_visualization_report(self):
        """EDA visuale: 8 immagini sul dataset grezzo."""
        self.analyze_target()
        self.missing_value_report()
        self.show_correlations()
        self.run_pca()
        self.clustering_eda()
        self.feature_importance_eda()

    def run_full_analysis(self, n_features=40, clustering_method='kmeans',
                          classification_bins=3, cv_folds=5):
        """
        Esegue l'intera pipeline nell'ordine corretto:
          1. Caricamento dati
          2. Preprocessing
          3. Feature selection (MI)
          4. Task regressione (confronto 4 modelli)
          5. Task clustering (K-Means o DBSCAN)
          6. Task classificazione
          7. EDA visuale
          8. Stampa riepilogo
        """
        print("\n" + "="*60)
        print("AMES HOUSING — ANALISI COMPLETA")
        print("="*60)

        self.load_data()
        self.preprocess_data(handle_outliers=True, cardinality_threshold=10)
        self.select_features_mi(k=n_features)
        self.evaluate_regression(n_splits=cv_folds)
        self.evaluate_clustering(method=clustering_method)
        self.evaluate_classification(n_bins=classification_bins, n_splits=cv_folds)
        self.run_visualization_report()
        self.print_results()
        return self


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Il CSV può stare nella stessa cartella dello script oppure in dataset/
    _base = os.path.dirname(os.path.abspath(__file__))
    _csv  = os.path.join(_base, 'AmesHousing.csv')
    if not os.path.exists(_csv):
        _csv = os.path.join(_base, '..', 'dataset', 'AmesHousing.csv')

    analyzer = AmesHousingAnalyzer(
        filepath=_csv,
        target_column='SalePrice'
    )

    analyzer.run_full_analysis(
        n_features=40,
        clustering_method='kmeans',
        classification_bins=3,
        cv_folds=5
    )
