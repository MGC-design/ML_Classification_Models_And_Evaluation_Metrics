from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return ColumnTransformer([
        ("numbers", StandardScaler(), numerical),
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ]), numerical, categorical
