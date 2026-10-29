from sklearn.naive_bayes import GaussianNB
from .base_model import BaseClassificationModel
class NaiveBayesModel(BaseClassificationModel):
    model_name="Gaussian Naive Bayes"
    def __init__(self): super().__init__(GaussianNB())
