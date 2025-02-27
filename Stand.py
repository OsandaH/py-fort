from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

numerical_columns = df_1.select_dtypes(include = 'number').columns
df_numerical = df_1[numerical_columns]

#### Z score scandalization ####
scaler = StandardScaler()
df_scaled_zscore = scaler.fit_transform(df_numerical)
df_scaled_zscore = pd.DataFrame(df_scaled_zscore, columns=numerical_cols)
#print(df_scaled_zscore.head())

#### Min Max Scaling ####
scaler = MinMaxScaler()
df_scaled_MinMax = scaler.fit_transform(df_numerical)
df_scaled_MinMax = pd.DataFrame(df_scaled_MinMax, columns=numerical_cols)
#print(df_scaled_MinMax.head())

#### Robust Scaling ####
scaler = RobustScaler()
df_scaled_robust = scaler.fit_transform(df_numerical)
df_scaled_robust = pd.DataFrame(df_scaled_robust, columns=numerical_cols)
#print(df_scaled_robust.head())
