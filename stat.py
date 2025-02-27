import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv") # read the dataset

df.head(10) # 1st 10 lines of the dataset

df.shape # rows and columns 

df.shape[0] # rows 
df.shape[1] # columns 

# adding header Row 
column_headers = [
    "symboling","normalized_losses","make","fuel_type",
    "aspiration","num_doors","body_style","drive_wheels",
    "engine_location","wheel_base","length","width","height",
    "curb_weight","engine_type","num_cylinders","engine_size",
    "fuel_system","bore","stroke","compression_ratio","horsepower",
    "peak_rpm","city_mpg","highway_mpg","price"]

df = pd.read_csv("dataset.csv", names=column_headers, na_values=["?"])

# number of null values
#df.isna().sum()
null_values = df.isna()
num_nulls = null_values.sum()
#print(num_nulls)

# number of duplicates 
duplicates = df.duplicated()
num_dups = duplicates.sum()
#print("\nNumber of duplicates: ",num_dups)

#print the duplicates
duplicates = df.duplicated(keep = False)
dup_rows = df[duplicates]
#print(dup_rows)

# Drop the Duplicates 
df_unique = df.drop_duplicates() 

# Drop missing values 
df_new = df_unique.dropna()


#### numerical data ####

numerical_cols = df.select_dtypes(include = 'number').columns 

# adding mean value to missing cols 
df_mean = df_new.copy()
df_mean[numerical_cols] = df_mean[numerical_cols].fillna(df_mean[numerical_cols].mean())

# fill the missing values with median
df_median = df_new.copy()
df_median[numerical_cols] = df_median[numerical_cols].fillna(df_new[numerical_cols].median)

#### categorical data ####

string_column = df_mf.select_dtypes(include='object').columns

# adding mean value to missing cols
df_mf = df_mean.copy()
for column in string_column:
    df_mf[column].fillna(df_mf[column].mode()[0], inplace = True)

# fill the missing values with median
df_mef = df_median.copy()
string_column = df_mef.select_dtypes(include='object').columns
for column in string_column:
    df_mef[column].fillna(df_mef[column].mode()[0], inplace = True)
    
    
#### One Hot Encoding ####

categorical_columns = df_new.select_dtypes(include = 'object').columns
df_encoded = pd.get_dummies(df_new, columns = categorical_columns, dtype = int)

#### statistics ####
numericalData = df_mean.select_dtypes(include = 'number')

stats = pd.DataFrame({
    'Mean': numericalData.mean(),
    'Median': numericalData.median(),
    'Mode': numericalData.mode().iloc[0],  # Selecting the first mode
    'Variance': numericalData.var(),
    'Standard Deviation': numericalData.std(),
    'Minimum': numericalData.min(),
    'Maximum': numericalData.max()
})
## print(stats)

# ## data types
# df.info()

# ## description
# df.describe()

## selecting few columns 
df1 = df[['normalized_losses','symboling']]

## group by
df[['make','horsepower']].groupby('make').mean()

##
mean_val = df['horsepower'].mean()
mode_values = df['horsepower'].mode().iloc[0] 
median_values = df['horsepower'].median()
