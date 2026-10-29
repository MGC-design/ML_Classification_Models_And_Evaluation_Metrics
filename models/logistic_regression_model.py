from sklearn.linear_model import LogisticRegression
from .base_model import BaseClassificationModel
class LogisticRegressionModel(BaseClassificationModel):
    model_name="Logistic Regression"
    def __init__(self): super().__init__(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
