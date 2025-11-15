# %%
# first of all we need to import the necessary libraries

import matplotlib.pyplot as plt  # for plotting
import numpy as np   # for numerical operations
import pandas as pd    # for data manipulation
import seaborn as sns   # for data visualization
from sklearn.inspection import permutation_importance
import tensorflow as tf    # for deep learning
from tensorflow import keras  # for building deep learning models
from tensorflow.keras import layers     # type: ignore # for layers in deep learning models
from sklearn.svm import SVC    # for Support Vector Classification
from sklearn.svm import SVR    # for Support Vector Regression
from sklearn.linear_model import LogisticRegression    # for Logistic Regression
from sklearn.preprocessing import PolynomialFeatures   # for Polynomial Features
from sklearn.linear_model import LinearRegression    # for Linear Regression
from sklearn.linear_model import BayesianRidge, ARDRegression   # for Bayesian Ridge and ARD Regression
from sklearn.model_selection import train_test_split   # for splitting the dataset
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score   # for evaluating the model
from sklearn.pipeline import make_pipeline # for creating a pipeline
from sklearn.impute import SimpleImputer # for handling missing values
from sklearn.preprocessing import StandardScaler  # for scaling the data
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# defining the DataModel class

class DataModel:
    # initializing the class with data and target variable
    def __init__(self, data, target = "RTML_mean"): 
        self.data = data
        self.target = target
        self.results = {}  # to store model results
        self.w = None # weights
        self.w_all = None # all weights for LinearRegression
        self.linear_model = None # Linear Regression model
        self.poly_model = None # Polynomial Regression model
        self.svm_model = None # Support Vector Machine model
        self.bayesian_model = None # Bayesian Regression model
        

# 1--------------   SPLIT THE DATA ON TRAIN AND TEST SETS   --------------
    def split_data(self, feature_cols, target_col="RTML_mean", test_size=0.25, random_state=42):
        """Splits dataset into train/test sets and stores them as class attributes"""
        # Store column names for later use
        self.feature_cols = feature_cols
        self.target = target_col

        # Prepare X and y
        X = self.data[feature_cols].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(self.data[target_col], errors='coerce')

        # Drop missing values
        Xy = pd.concat([X, y], axis=1).dropna()
        X, y = Xy[feature_cols], Xy[target_col]

        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
        )
        print(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test


#--------------LINEAL REGRESSION--------------#

    # train a Linear Regression model
    def train_linear_regression(self):
        """Trains a Linear Regression model on the training data"""
        if not hasattr(self, "X_train"): # check if data has been split
            print("You must call split_data() first!")
            return

        self.lin_pipe = make_pipeline( # create a pipeline  
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LinearRegression()
            )
        
        self.lin_pipe.fit(self.X_train, self.y_train) # define the model
        y_pred = self.lin_pipe.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        r2  = r2_score(self.y_test, y_pred) 
        self.results['Linear'] = {'MSE': mse, 'R²': r2}  # ADD RESULTS TO DICT
        print(f"MSE = {mse:.3f} | R² = {r2:.3f}")

        return mse, r2
    # plot Linear Regression results
    def plot_linear_regression(self, feature_col):
        if not hasattr(self, "lin_pipe"):
            print("Train the model first!")
            return
        if feature_col not in self.feature_cols: # check if feature exists
            print(f"Feature '{feature_col}' not used in model.")
            return
        y_pred = self.lin_pipe.predict(self.X_test) # predict on test set
        x = self.X_test[feature_col].to_numpy()
        order = np.argsort(x)

        fig, ax = plt.subplots(figsize=(7,5)) # create figure and axis
        ax.scatter(x, self.y_test, color='blue', alpha=0.5, label='Actual')
        ax.plot(x[order], y_pred[order], color='red', linewidth=2, label='Predicted')
        ax.set_xlabel(feature_col); ax.set_ylabel(self.target)
        ax.set_title(f'Linear Regression: {self.target} vs {feature_col}')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.show()
        return fig
    
    # feature importance from coefficients of Linear Regression

    def plot_feature_importance_linear(self):
        """Feature importance from |coefficients| of LinearRegression inside pipeline."""
        if not hasattr(self, "lin_pipe"): # check if data has been split
            print("Train the linear model first (train_linear_regression).")
            return

        # grab the LinearRegression step from the pipeline
        try:
            lr = self.lin_pipe.named_steps["linearregression"] # get the linear regression step
        except Exception:
            print("Could not find 'linearregression' step in the pipeline.")
            return
        coefs = lr.coef_.ravel() # get the coefficients
        names = list(self.feature_cols) # feature names

        imp = pd.Series(np.abs(coefs), index=names).sort_values(ascending=True) # importance as |coefficients|

        fig, ax = plt.subplots(figsize=(7,5))
        imp.plot(kind="barh", ax=ax) # horizontal bar plot
        ax.set_title("Feature importance (|coefficients|) — Linear Regression")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout(); # adjust layout to prevent clipping
        plt.show()
        return fig

    # permutation importance for the test set

    def plot_permutation_importance_linear(self, n_repeats=10, scoring="r2", title=None, save_fig=False):
        """
        Permutation importance for Linear Regression (on test set).
        Shows how much R² drops when each feature is shuffled.
        
        Args:
            n_repeats (int): Number of times to shuffle each feature
            scoring (str): 'r2', 'neg_mean_squared_error', etc.
            title (str): Custom title
            save_fig (bool): Save plot to PNG
        """
        if not hasattr(self, "lin_pipe"):
            print("Train the linear model first (train_linear_regression).")
            return

        print(f"Computing permutation importance for Linear Regression (scoring={scoring})...")
        r = permutation_importance(
            self.lin_pipe, self.X_test, self.y_test,
            n_repeats=n_repeats, random_state=42, scoring=scoring
        )

        # Create importance series
        importance = pd.Series(r.importances_mean, index=self.feature_cols)
        importance = importance.sort_values(ascending=True)

        # Plot
        plt.figure(figsize=(8, 6))
        bars = plt.barh(importance.index, importance.values, 
                       xerr=r.importances_std[importance.index.argsort()], 
                       color="#360eff", alpha=0.8, capsize=5)

        # Styling
        if title is None:
            title = f"Permutation Importance (Linear Regression)\nScoring: {scoring.upper()}"
        plt.title(title, fontsize=15, fontweight='bold', pad=20)
        plt.xlabel("Mean Decrease in Score", fontsize=12)
        plt.ylabel("Features", fontsize=12)

        # Add value labels on bars
        for i, (feat, val) in enumerate(importance.items()):
            plt.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

        plt.grid(True, axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()

        # Save
        if save_fig:
            fname = f"permutation_importance_linear_{scoring}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved: {fname}")

        plt.show()
        return plt.gcf()
    

    # ----------------SUPPORT VECTOR CLASSIFICATION--------------#

    # split data for classification
    def split_data_clf(self, feature_cols, target_col="RTML_mean_binary", test_size=0.25, random_state=42):
        self.feature_cols = feature_cols
        self.target = target_col
        # prepare X, y (numeric + drop NaN)
        X = self.data[feature_cols].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(self.data[target_col], errors='coerce')

        # drop missing values
        Xy = pd.concat([X, y], axis=1).dropna()
        X, y = Xy[feature_cols], Xy[target_col].astype(int)

        # split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"[CLF] Train: {self.X_train.shape}, Test: {self.X_test.shape}, Positives in test: {self.y_test.mean():.2f}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # ---------------TRAIN SUPPORT VECTOR MACHINE CLASSIFIER
    def train_svm(self, kernel="linear", C=1.0, gamma="scale"):
        if not hasattr(self, "X_train"): # check if data has been split
            print("Call split_data_clf() first.")
            return
            # create SVM pipeline
        self.svm_pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
            )
        self.svm_pipe.fit(self.X_train, self.y_train)
        y_pred = self.svm_pipe.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, zero_division=0)
        rec = recall_score(self.y_test, y_pred, zero_division=0)
        #store results with correct capitalization and formatting
        if not hasattr(self, 'results'):
            self.results = {}
        self.results['SVM'] = {'accuracy': acc, 'precision': prec, 'recall': rec}

        print(f"SVM ({kernel}) → accuracy={acc:.3f} | precision={prec:.3f} | recall={rec:.3f}")
        print(classification_report(self.y_test, y_pred, digits=3))
        self._last_svm_pred = y_pred # store for confusion matrix
        return acc, prec, rec
        
        # Confusion matrix of SVM model
    def plot_confusion(self):
        if not hasattr(self, "_last_svm_pred"): # check if SVM has been trained
            print("Train SVM first.")
            return
        cm = confusion_matrix(self.y_test, self._last_svm_pred) # confusion matrix
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout(); 
        plt.show()
        return fig

    
    # permutation importance for the test set
    def plot_permutation_importance_svm(self, scoring="accuracy", n_repeats=10):
        if not hasattr(self, "svm_pipe"): # check if SVM has been trained
            print("Train SVM first (train_svm).")
            return
        r = permutation_importance(
            self.svm_pipe, self.X_test, self.y_test,
            n_repeats=n_repeats, random_state=42, scoring=scoring
        )# compute permutation importance
        imp = pd.Series(r.importances_mean, index=self.feature_cols).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(7,5))
        imp.plot(kind="barh", ax=ax)
        ax.set_title(f"Permutation importance (SVM) — scoring={scoring}")
        ax.set_xlabel("Mean importance (score drop)")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout(); plt.show()
        return fig
    
# -- Beta–Binomial posterior for a binary target (e.g., RTML_mean_binary)
    def beta_binomial_posterior(self, target_col="RTML_mean_binary", a0=1, b0=1, n_samples=50000, plot=True):
        """
        Simple posterior for probability p using Beta-Binomial:
        prior: Beta(a0,b0); posterior: Beta(a0+successes, b0+failures).
        Uses numpy to sample posterior and estimate 95% credible interval.
        """
        # takes binary target (0/1) and drop NaN
        y = pd.to_numeric(self.data[target_col], errors='coerce').dropna().astype(int).values
        successes = int(y.sum())
        failures = int((1 - y).sum())

        # posterior parameters
        a_post = a0 + successes
        b_post = b0 + failures

        # sample from posterior with numpy (no SciPy needed)
        rng = np.random.default_rng(42)
        samples = rng.beta(a_post, b_post, size=n_samples)

        post_mean = samples.mean()
        # 95% credible interval
        ci_low, ci_high = np.percentile(samples, [2.5, 97.5])

        print(f"[Beta–Binomial] successes={successes}, failures={failures}")
        print(f"Posterior ~ Beta({a_post}, {b_post})")
        print(f"Posterior mean p ≈ {post_mean:.3f}")
        print(f"95% credible interval ≈ [{ci_low:.3f}, {ci_high:.3f}]")

        if plot:
            plt.figure(figsize=(6,4))
            plt.hist(samples, bins=40, density=True, alpha=0.7)
            plt.title(f"Posterior of p (Beta({a_post},{b_post}))")
            plt.xlabel("p"); plt.ylabel("density")
            plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

        # return simple dict with results
        return {"a_post": a_post, "b_post": b_post, "mean": post_mean, "ci": (ci_low, ci_high)}
    # -- Bayesian Regression (pipeline: impute + scale + BayesianRidge)
    def train_bayesian_regression(self, feature_cols, target_col="RTML_mean", test_size=0.25, random_state=42):
        """
        Trains BayesianRidge on selected features. Returns MSE and R^2 on test.
        """
        # keep names
        self.feature_cols = feature_cols
        self.target = target_col

        # prepare X, y (numeric + drop NaN)
        X = self.data[feature_cols].apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(self.data[target_col], errors='coerce')
        Xy = pd.concat([X, y], axis=1).dropna()
        X, y = Xy[feature_cols], Xy[target_col]

        # split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # pipeline
        self.br_pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            BayesianRidge()
        )

        # fit + evaluate
        self.br_pipe.fit(self.X_train, self.y_train)
        y_pred = self.br_pipe.predict(self.X_test)

        mse = mean_squared_error(self.y_test, y_pred)
        r2  = r2_score(self.y_test, y_pred)
        
        # store results
        self.results['BayesianRidge'] = {'MSE': mse, 'R²': r2}
        print(f"[BayesianRidge] MSE = {mse:.3f} | R² = {r2:.3f}")
        return mse, r2

    # -- (optional) quick bar plot of BayesianRidge coefficients
    def plot_bayesian_coeffs(self):
        if not hasattr(self, "br_pipe"): # check if model trained
            print("Train Bayesian Regression first (train_bayesian_regression).")
            return
        br = self.br_pipe.named_steps["bayesianridge"] # get the BayesianRidge step
        coefs = br.coef_.ravel() # get coefficients
        names = list(self.feature_cols)

        imp = pd.Series(coefs, index=names).sort_values() # importance as coefficients
        plt.figure(figsize=(7,5))
        imp.plot(kind="barh") # horizontal bar plot
        plt.title("BayesianRidge coefficients (standardized features)")
        plt.xlabel("Coefficient")
        plt.grid(True, axis="x", alpha=0.3)
        plt.tight_layout(); plt.show()

    def beta_binomial_compare(self, group_col, target_col="RTML_mean_binary", a0=1, b0=1, n_samples=50000):
        """
        Compare two groups (e.g., low vs high ESCS) using Beta–Binomial posterior.
        Computes P(p_high > p_low) where p = success probability (e.g., high reader rate).
        """
        # prepare data
        df = self.data[[group_col, target_col]].dropna()
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").astype(int)

        # create two groups: low / high (split by median)
        median_val = df[group_col].median()
        low = df[df[group_col] <= median_val][target_col]
        high = df[df[group_col] > median_val][target_col]

        s_low, f_low = low.sum(), len(low) - low.sum()
        s_high, f_high = high.sum(), len(high) - high.sum()

        # posterior params
        a_low, b_low = a0 + s_low, b0 + f_low
        a_high, b_high = a0 + s_high, b0 + f_high

        # sample from posteriors
        rng = np.random.default_rng(42)
        p_low = rng.beta(a_low, b_low, size=n_samples)
        p_high = rng.beta(a_high, b_high, size=n_samples)

        # probability that high group > low group
        prob = np.mean(p_high > p_low)
        print(f"Group split by median {group_col}: low ≤ {median_val:.3f}, high > {median_val:.3f}")
        print(f"P(p_high > p_low) ≈ {prob:.3f}")

        # small plot
        plt.figure(figsize=(6,4))
        plt.hist(p_low, bins=40, alpha=0.6, label="Low group")
        plt.hist(p_high, bins=40, alpha=0.6, label="High group")
        plt.title(f"Posterior of success rate (by {group_col})")
        plt.xlabel("p (probability of high reader)")
        plt.ylabel("Density")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.show()

        return prob
    

    # === K-MEANS CLUSTERING ===
    def kmeans_clustering(self, features, n_clusters=3):
        # K-means clustering on selected features with PCA plot
        X = self.data[features].apply(pd.to_numeric, errors='coerce').dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        self.data.loc[X.index, 'Cluster'] = clusters
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='Set1', alpha=0.7)
        plt.title(f'K-means (k={n_clusters}) — Student Profiles')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter); plt.grid(True, alpha=0.3); plt.show()
        
        print("\nCluster Profiles:")
        print(self.data.groupby('Cluster')[features + ['RTML_mean']].mean().round(2))
        return kmeans

    # === MODEL COMPARISON  ===
    def plot_model_comparison(self):
        if not self.results:
            print("Train models first!")
            return

        # Turn results dict into DataFrame
        df = pd.DataFrame(self.results).T
        print("Current results table:")
        print(df)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # === CLASSIFICATION (SVM) ===
        # we expect keys like 'svm' with 'accuracy', 'precision', 'recall'
        clf_metrics = ['accuracy', 'precision', 'recall']
        clf_cols = [c for c in clf_metrics if c in df.columns]  # only existing ones
        clf_df = df.loc[df.index.str.contains('svm', case=False), clf_cols]
        # plot if we have SVM results
        if not clf_df.empty:
            clf_df.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(clf_cols)])
            ax1.set_title('Classification (SVM)', fontsize=14, fontweight='bold')
            ax1.set_ylim(0, 1)
            ax1.set_xticklabels(clf_df.index, rotation=0)
            # add value labels
            for p in ax1.patches:
                height = p.get_height()
                ax1.annotate(f'{height:.3f}',
                            (p.get_x() + p.get_width() / 2., height + 0.02),
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No SVM\nresults', transform=ax1.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax1.set_title('Classification', fontsize=14, fontweight='bold')

        # === REGRESSION (Linear, BayesianRidge) ===
        # we expect keys like 'linear', 'bayesianridge' with 'MSE' and/or 'R²'
        reg_df = df.loc[~df.index.str.contains('svm', case=False)].copy()

        if 'R²' in reg_df.columns:   # if used capital R²
            r2_col = 'R²'
        elif 'r2' in reg_df.columns: # if used lowercase r2
            r2_col = 'r2'
        else:
            r2_col = None # no R² available

        if r2_col is not None and not reg_df.empty:
            # plot R² values for regression models
            reg_df[r2_col].plot(kind='bar', ax=ax2, color='#3498db')
            ax2.set_title('Regression (R²)', fontsize=14, fontweight='bold')
            ax2.set_ylim(min(0, reg_df[r2_col].min() - 0.05),
                        max(0.2, reg_df[r2_col].max() + 0.05))
            ax2.set_xticklabels(reg_df.index, rotation=0)
            # add value labels

            for idx, val in enumerate(reg_df[r2_col]):
                ax2.annotate(f'{val:.3f}',
                            (idx, val + 0.01),
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No regression\nresults', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title('Regression (R²)', fontsize=14, fontweight='bold')

        plt.suptitle('MODEL COMPARISON', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()




    # === FINAL PREDICTION ===
    # Predict for a new student given a dictionary of features
    def predict_student(self, student_dict):
        print("\nPREDICTION FOR NEW STUDENT:")
        print(student_dict)
        
        # Create DataFrame with correct columns
        X_new = pd.DataFrame([student_dict])
        
        # Reindex to match training columns (fill missing with 0 or median)
        if hasattr(self, 'lin_pipe') or hasattr(self, 'br_pipe'): # if models trained
            # Get feature names from regression pipeline
            reg_features = self.feature_cols  # from split_data()
        else:
            reg_features = []
         # Get feature names from classification pipeline   
        if hasattr(self, 'svm_pipe'):
            clf_features = self.feature_cols  # from split_data_clf()
        else:
            clf_features = []
        
        # SVM Prediction
        if hasattr(self, 'svm_pipe') and clf_features: # if SVM trained
            X_svm = X_new.reindex(columns=clf_features, fill_value=0)
            pred_class = self.svm_pipe.predict(X_svm)[0]
            print(f"→ High Reader? {'YES' if pred_class == 1 else 'NO'}")
        
        # Regression Prediction
        if hasattr(self, 'br_pipe') and reg_features:
            X_reg = X_new.reindex(columns=reg_features, fill_value=0)
            pred_score = self.br_pipe.predict(X_reg)[0]
            print(f"→ Predicted Score: {pred_score:.1f}")