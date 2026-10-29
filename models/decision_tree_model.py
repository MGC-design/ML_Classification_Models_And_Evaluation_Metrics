from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseClassificationModel
class DecisionTreeModel(BaseClassificationModel):
    model_name="Decision Tree"
    def __init__(self): super().__init__(DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, class_weight="balanced", random_state=42))
