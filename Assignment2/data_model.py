#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

class DataModel:
    def __init__(self, df, target="quality_encoded"):
        # save the dataframe and the target column
        self.df = df.copy()
        self.target = target
        self.w = None  # weights for regression
        self.splits = None

    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        # shuffle data before splitting
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(self.df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        self.train = self.df.iloc[:n_train]
        self.val = self.df.iloc[n_train:n_train + n_val]
        self.test = self.df.iloc[n_train + n_val:]

        print("Data split complete:")
        print("Train:", self.train.shape)
        print("Validation:", self.val.shape)
        print("Test:", self.test.shape)
        return self.train, self.val, self.test

    def fit_linear_regression(self, features):
        # check if features are numeric
        non_numeric = [col for col in features if not pd.api.types.is_numeric_dtype(self.train[col])]
        if non_numeric:
            raise ValueError(f"These columns are not numeric and cannot be used: {non_numeric}")

        # prepare X and y for training
        X = self.train[features].to_numpy(dtype=float)
        y = self.train[self.target].to_numpy(dtype=float).reshape(-1, 1)

        # add a column of 1s for bias term
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # normal equation: w = (X^T X)^(-1) X^T y
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ y
        print("Model trained. Weights shape:", self.w.shape)
        return self.w

    def predict(self, df, features):
        # prepare X for prediction
        X = df[features].to_numpy(dtype=float)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.w

    def mse(self, y_true, y_pred):
        # mean squared error
        return np.mean((y_true - y_pred) ** 2)

