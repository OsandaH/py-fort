## Correlation matrix

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet 
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("car_data.csv")

## drop a column 
df_new = df.drop(columns = ["car_name"])

## data types of columns 
df_new.dtypes

## Rows with null values
null_rows = df_new[df_new.isnull().any(axis=1)]

## filling missing values with mean 
df_mean_fill = df_new.copy()
num_cols = df_new.select_dtypes(include='number').columns
df_mean_fill[num_cols] = df_mean_fill[num_cols].fillna(df_mean_fill[num_cols].mean())

df_1 = df_mean_fill.copy()
## describe 1 element in each column 
std = df.describe().loc["std"]
maxSTD = np.max(std)

correlation_matrix = df_1.corr()
#print(correlation_matrix)
plt.figure(figsize = (10,8))
plt.imshow(correlation_matrix, cmap = "coolwarm")
plt.colorbar()
plt.xticks(range(len(correlation_matrix)),correlation_matrix.columns,rotation = 90)
plt.xticks(range(len(correlation_matrix)),correlation_matrix.columns)
plt.title('Correlation Matrix Heatmap')
plt.show()

#### Positive pairs ####
positive_pairs_df = correlation_matrix.unstack()
positive_pairs_df = positive_pairs_df[(positive_pairs_df > 0) & (positive_pairs_df.index.get_level_values(0) != positive_pairs_df.index.get_level_values(1))]
positive_pairs_df = positive_pairs_df[positive_pairs_df.index.map(lambda x: x[0] < x[1])]

#### Negative pairs ####
positive_pairs_df_neg = correlation_matrix.unstack() # Unstack the matrix
positive_pairs_df_neg = positive_pairs_df_neg[(positive_pairs_df_neg < 0) & (positive_pairs_df_neg.index.get_level_values(0) != positive_pairs_df_neg.index.get_level_values(1))]
positive_pairs_df_neg = positive_pairs_df_neg[positive_pairs_df_neg.index.map(lambda x: x[0] < x[1])]

#### Pairs Higher than a specific value ####
high_corr_pairs = positive_pairs_df[positive_pairs_df > 0.8]
