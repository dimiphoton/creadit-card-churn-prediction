from sklearn.preprocessing import PowerTransformer, StandardScaler

# separate the features and target variable
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']

# get the list of numerical features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# normalize skewed features with PowerTransformer
pt = PowerTransformer()
X[num_cols] = pt.fit_transform(X[num_cols])

# scale features with StandardScaler
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
