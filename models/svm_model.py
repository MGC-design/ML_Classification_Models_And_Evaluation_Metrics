from sklearn.svm import SVC
from .base_model import BaseClassificationModel
class SVMModel(BaseClassificationModel):
    model_name="Support Vector Machine"
    def __init__(self): super().__init__(SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42))
