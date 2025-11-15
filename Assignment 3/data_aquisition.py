# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class data_aquisition:
    def __init__(self, filename):
        # just the filename (CSV must be in same folder)
        self.filename = filename
        self.data = None
        
    def load_data(self):
        # load data from CSV in same folder
        self.data = pd.read_csv(self.filename)
        print("5 first rows of the dataset:")
        print(self.data.head())
        return self.data
    
    # Drop rows with missing values ---
        self.data = self.data.dropna(subset=['RTML_mean', 'ESCS', 'ST011Q01TA'])
        print("Cleaned data preview:")
        print(self.data.head())
        print("Data types:")
        print(self.data.dtypes)

        return self.data
    
    # preprocess data
    def preprocess_data(self, target='RTML_mean'):
        # Check if data is loaded
        if self.data is None:
            print("Data not loaded. Please call load_data() first.")
            return

    # 1) Delete rows with missing target
        if target in self.data.columns:
            before = len(self.data)
            self.data = self.data.dropna(subset=[target])
            after = len(self.data)
            print(f"Dropped {before - after} rows with missing target '{target}'.")
        else:
            print(f"WARNING: target column '{target}' not found. Skipping target drop.")
    # 2) Delete country column if only NOR
        if 'CNT' in self.data.columns and self.data['CNT'].nunique() == 1:
            self.data = self.data.drop(columns=['CNT'])
            print("Dropped column 'CNT' (only one country).")

            return self.data


    
    
    # Plotting functions
    def plot_histogram(self, column_name='RTML_mean'):
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.data[column_name], bins=30, color='skyblue',edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of {column_name}', fontsize=14)
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

    # scatter plot to check correlation between two variables
    def plot_scatter(self, x_column, y_column):
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[x_column], self.data[y_column], color='red', alpha=0.5)
        plt.title(f'Scatter Plot of {y_column} vs {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True)
        plt.show()

    # plot_cumulative_distibution
    def plot_cumulative_distribution(self, column_name='RTML_mean'):
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return
        # calculate cumulative distribution
        sorted_data = np.sort(self.data[column_name])
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_data, cumulative, marker='.', linestyle='none', color='purple')
        plt.title(f'Cumulative Distribution of {column_name}')
        plt.xlabel(column_name)
        plt.ylabel('Cumulative Probability')
        plt.grid(True)
        plt.show() 

    # boxplot to check reading level vs other variables
    def plot_boxplot(self, x_column, y_column="RTML_mean"):
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=self.data[x_column], y=self.data[y_column], palette='Set3')
        plt.title(f'Boxplot of {y_column} by {x_column}', frontsize=14)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True)
        plt.show()

    def plot_correlation_heatmap(self):
        if self.data is None:
            print("Data not loaded. Please load data first.")
            return
        # select only numeric columns
        numeric_data = self.data.select_dtypes(include=['int64', 'float64']) 

        #calculate correlation matrix
        corr_matrix = numeric_data.corr()

        #plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5) #annot=True to display correlation values
        plt.title('Correlation Heatmap of numerical variables', fontsize=16, fontweight='bold')
        plt.show()


    
    # ===: PISA-STYLE PLOTS ===


    def plot_pisa_trend_norway(self):
        """Norway's reading score trend over PISA cycles"""
        years = [2000, 2003, 2006, 2009, 2012, 2015, 2018]
        scores = [500, 505, 500, 503, 504, 513, 499]  # PISA 2018 ≈ 499
        plt.figure(figsize=(11, 6))
        plt.plot(years, scores, 'o-', color='darkred', linewidth=2.5, markersize=8, label='Norway')
        plt.axhline(500, color='gray', linestyle='--', alpha=0.6, label='OECD Average')
        plt.text(2016, 505, "↓ -14 points", fontsize=11, color='red', fontweight='bold')
        plt.title('Norway Reading Performance in PISA (2000–2018)', fontsize=14, fontweight='bold')
        plt.xlabel('Year'); plt.ylabel('PISA Reading Score')
        plt.ylim(480, 520); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    def plot_reading_by_digital_level(self):
        """Boxplot: Reading score by digitalization level (Low/Medium/High)"""
        if self.data is None or 'RATCMP1' not in self.data.columns:
            print("RATCMP1 not found. Skipping.")
            return
        df = self.data.copy()
        df['Digital_Level'] = pd.cut(df['RATCMP1'], bins=3, labels=['Low', 'Medium', 'High'])
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Digital_Level', y='RTML_mean', data=df, palette='viridis')
        plt.title('Reading Score by School Digitalization Level', fontsize=14, fontweight='bold')
        plt.xlabel('Digitalization Level (Computers per Student)')
        plt.ylabel('Reading Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_digital_vs_reading(self):
        """Scatter: School digitalization vs reading, colored by gender"""
        if self.data is None or 'RATCMP1' not in self.data.columns:
            print("RATCMP1 not found. Skipping.")
            return
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=self.data, x='RATCMP1', y='RTML_mean',
            hue='ST004D01T', palette={1.0: 'blue', 2.0: 'pink'}, alpha=0.6, s=60
        )
        plt.title('School Digitalization vs Reading Score\n(Colored by Gender)', fontsize=14, fontweight='bold')
        plt.xlabel('Computers per Student (RATCMP1)')
        plt.ylabel('Reading Score (RTML_mean)')
        plt.legend(title='Gender', labels=['Male', 'Female'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_reading_by_digital_level(self):
        """Boxplot: Reading score by digitalization level (Low/Medium/High)"""
        if self.data is None or 'RATCMP1' not in self.data.columns:
            print("RATCMP1 not found. Skipping.")
            return
        df = self.data.copy() # create a copy to avoid modifying original
        df['Digital_Level'] = pd.cut(df['RATCMP1'], bins=3, labels=['Low', 'Medium', 'High'])
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Digital_Level', y='RTML_mean', data=df, palette='viridis')
        plt.title('Reading Score by School Digitalization Level', fontsize=14, fontweight='bold')
        plt.xlabel('Digitalization Level (Computers per Student)')
        plt.ylabel('Reading Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_key_correlations(self):
        """Focused heatmap: only key variables"""
        if self.data is None:
            print("Data not loaded.")
            return
        key_vars = ['RTML_mean', 'ESCS', 'RATCMP1', 'RATCMP2', 'EDUSHORT', 'ST004D01T']
        df_key = self.data[key_vars].select_dtypes(include=[np.number])
        corr = df_key.corr()
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlBu_r', center=0, square=True, linewidths=0.5)
        plt.title('Key Correlations: Reading, ESCS, and Digitalization', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
            



