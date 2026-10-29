from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseClassificationModel
class KNNModel(BaseClassificationModel):
    model_name="K-Nearest Neighbors"
    def __init__(self): super().__init__(KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1))
