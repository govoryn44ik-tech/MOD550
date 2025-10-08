#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

# optional imports (for Task 2.5 and 2.6)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    tf = None  # we'll warn if user calls NN without TF

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
except Exception:
    KMeans = None
    GaussianMixture = None


class DataModel:
    def __init__(self, df, target="quality_encoded"):
        # store data and target
        self.df = df.copy()
        self.target = target
        self.w = None          # weights for linear regression (train split)
        self.w_all = None      # weights for regression on all data
        self.splits = None     # dict with train/val/test
        # placeholders for optional models
        self.nn_model = None
        self.nn_history = None
        self.kmeans_model = None
        self.gmm_model = None

    # ---------------------- Task 2.3: split into train/val/test ----------------------
    def split_data(self, train_ratio=0.7, val_ratio=0.15, random_state=42):
        # shuffle first
        data = self.df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n = len(data)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        self.train = data.iloc[:n_train]
        self.val   = data.iloc[n_train:n_train + n_val]
        self.test  = data.iloc[n_train + n_val:]

        self.splits = {"train": self.train, "val": self.val, "test": self.test}
        print("✅ Data split complete:")
        print("Train:", self.train.shape, " Val:", self.val.shape, " Test:", self.test.shape)
        return self.train, self.val, self.test

    # ---------------------- Task 2.2: linear regression (normal equation) ----------------------
    def fit_linear_regression(self, features):
        # check numeric features
        non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(self.train[c])]
        if non_numeric:
            raise ValueError(f"These features are not numeric: {non_numeric}")

        # build X and y from TRAIN
        X = self.train[features].to_numpy(dtype=float)
        y = self.train[self.target].to_numpy(dtype=float).reshape(-1, 1)

        # add bias (intercept) term
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # normal equation: w = (X^T X)^(-1) X^T y
        self.w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        print("✅ Linear regression trained on TRAIN. Weights shape:", self.w.shape)
        return self.w

    # helper: predict with current linear regression weights
    def predict(self, df, features):
        X = df[features].to_numpy(dtype=float)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.w

    # ---------------------- Task 2.4: MSE ----------------------
    def mse(self, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1, 1)
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        return float(np.mean((y_true - y_pred) ** 2))

    # convenience for Task 2.9
    def evaluate_mse_on_split(self, split_name, features):
        if self.splits is None:
            raise ValueError("No splits yet. Call split_data() first.")
        if split_name not in ("train", "val", "test"):
            raise ValueError("split_name must be 'train', 'val', or 'test'.")

        df_split = self.splits[split_name]
        y_true = df_split[self.target].to_numpy().reshape(-1, 1)
        y_pred = self.predict(df_split, features)
        return self.mse(y_true, y_pred)

    # ---------------------- Task 2.7: linear regression on ALL data ----------------------
    def fit_linear_regression_on_all(self, features):
        # check numeric features
        non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(self.df[c])]
        if non_numeric:
            raise ValueError(f"These features are not numeric: {non_numeric}")

        X = self.df[features].to_numpy(dtype=float)
        y = self.df[self.target].to_numpy(dtype=float).reshape(-1, 1)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w_all = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        print("✅ Linear regression trained on ALL data. Weights shape:", self.w_all.shape)
        return self.w_all

    # ---------------------- Task 2.5: simple Neural Network (Keras) ----------------------
    def _xy_from_df(self, df, features):
        X = df[features].to_numpy(dtype=float)
        y = df[self.target].to_numpy(dtype=float).reshape(-1, 1)
        return X, y

    def build_and_train_nn(self, features, hidden_units=(16, 8), epochs=20, batch_size=32, verbose=0):
        if tf is None:
            raise ImportError("TensorFlow/Keras not installed. Install with: pip install tensorflow")

        # choose train/val
        if self.splits is None:
            X_train, y_train = self._xy_from_df(self.df, features)
            X_val, y_val = X_train, y_train
        else:
            X_train, y_train = self._xy_from_df(self.train, features)
            if hasattr(self, "val") and len(self.val) > 0:
                X_val, y_val = self._xy_from_df(self.val, features)
            else:
                X_val, y_val = self._xy_from_df(self.test, features)

        # tiny MLP
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        for hu in hidden_units:
            model.add(layers.Dense(hu, activation="relu"))
        model.add(layers.Dense(1))  # regression output

        model.compile(optimizer="adam", loss="mse")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs, batch_size=batch_size, verbose=verbose)

        self.nn_model = model
        self.nn_history = history
        print("NN trained (Keras).")
        return model, history

    def nn_predict(self, df, features):
        if self.nn_model is None:
            raise ValueError("NN model not trained yet. Call build_and_train_nn() first.")
        X = df[features].to_numpy(dtype=float)
        return self.nn_model.predict(X, verbose=0)

    # ---------------------- Task 2.6: K-Means and GMM ----------------------
    def fit_kmeans(self, features, n_clusters=3, random_state=42):
        if KMeans is None:
            raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
        X = self.df[features].to_numpy(dtype=float)
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        km.fit(X)
        self.kmeans_model = km
        self.df["kmeans_label"] = km.labels_
        print("K-Means fitted. Labels stored in df['kmeans_label'].")
        return km

    def fit_gmm(self, features, n_components=3, random_state=42):
        if GaussianMixture is None:
            raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")
        X = self.df[features].to_numpy(dtype=float)
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(X)
        labels = gmm.predict(X)
        self.gmm_model = gmm
        self.df["gmm_label"] = labels
        print("GMM fitted. Labels stored in df['gmm_label'].")
        return gmm


# In[ ]:




