#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

class DataAcquisition:
    def __init__(self, file_path):
        # save path to CSV
        self.file_path = file_path
        self.df = None

    def open_data(self):
        # read CSV
        self.df = pd.read_csv(self.file_path)
        print("First 5 rows of raw data:")
        print(self.df.head())

        # --- 1. Encode quality labels ---
        quality_map = {
            'Worst taste wine': 0,
            'Ouch, my palate': 1,
            'Meh, table wine': 2,
            'Not bad, second glass': 3,
            'Nice, dinner-worthy': 4,
            "Chef's kiss": 5,
            'Halleluya': 6
        }
        if "quality" in self.df.columns:
            self.df["quality_encoded"] = self.df["quality"].map(quality_map)

        # --- 2. Encode type (red/white) ---
        if "type" in self.df.columns:
            self.df["type_encoded"] = self.df["type"].map({"red": 0, "white": 1})

        # --- 3. Convert numeric columns ---
        numeric_cols = [
            "quality_encoded", "alcohol", "volatile acidity",
            "sulphates", "type_encoded"
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # --- 4. Drop rows with missing values ---
        self.df = self.df.dropna(subset=numeric_cols)

        print("Cleaned data preview:")
        print(self.df.head())
        print("Data types:")
        print(self.df.dtypes)

        return self.df

    def plot_histogram(self, column="quality_encoded"):
        # histogram
        if self.df is None:
            self.open_data()
        self.df[column].hist(bins=10)
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.show()

    def plot_pmf(self, column="quality_encoded"):
        # PMF (probability distribution)
        if self.df is None:
            self.open_data()
        pmf = self.df[column].value_counts(normalize=True).sort_index()
        pmf.plot(kind="bar")
        plt.title(f"PMF of {column}")
        plt.xlabel(column)
        plt.ylabel("Probability")
        plt.grid(axis="y", alpha=0.5)
        plt.show()

    def plot_cumulative(self, column="quality_encoded"):
        # cumulative distribution
        if self.df is None:
            self.open_data()
        counts = self.df[column].value_counts(normalize=True).sort_index()
        cumulative = counts.cumsum()
        cumulative.plot(drawstyle="steps-post")
        plt.title(f"Cumulative distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Cumulative Probability")
        plt.grid(axis="y", alpha=0.5)
        plt.show()


# In[ ]:




