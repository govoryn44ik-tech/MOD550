#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataAcquisition:
    def __init__ (self, Year, Reading_score):
        """
        Defining the class function"
        """
        self.Year=Year
        self.Reading_score=Reading_score

    def Opendata(self):
        self.df = pd.read_csv('Norway.csv')
        print(self.df.head())
        return self.df

    def plot_histogram(self):
        plt.figure(figsize=(15,6))
        plt.bar(x=self.df['Year'], height=self.df['Reading score'], color='skyblue')
        plt.xlabel('Years')
        plt.ylabel('Reading score in Norway')
        plt.xticks(self.df['Year'])
        plt.title('Norway')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plot_pmf(self):
        df=self.Opendata() #get dataframe from function
        
        #calculating 
        counts = df['Reading score'].value_counts(normalize=True).sort_index()

        #create a histogram
        plt.figure(figsize=(15,6))
        plt.bar(x=counts.index, height=counts.values, color='purple')

        plt.title('PMF of Reading Scores in Norway')
        plt.xlabel('Reading Score')
        plt.ylabel('Probability')
        plt.xticks(counts.index)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def cumulative_sum(self):
        df = self.Opendata()
        df['Cumulative Reading Score'] = df['Reading score'].cumsum()
        print(df[['Reading score', 'Cumulative Reading Score']])
        return df
    
    


# In[ ]:




