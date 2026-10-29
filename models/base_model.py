from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from utils.preprocessing import build_preprocessor

class BaseClassificationModel:
    model_name = "Base Model"
    def __init__(self, estimator=None): self.estimator, self.pipeline = estimator, None
    def train(self, X, y):
        preprocessor, _, _ = build_preprocessor(X)
        self.pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", self.estimator)])
        self.pipeline.fit(X, y); return self
    def predict(self, X): return self.pipeline.predict(X)
    def predict_score(self, X):
        if hasattr(self.pipeline, "predict_proba"): return self.pipeline.predict_proba(X)[:, 1]
        scores=self.pipeline.decision_function(X)
        return 1/(1+__import__('numpy').exp(-scores))
    def save(self, path): Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(self.pipeline, path)
    def load(self, path): self.pipeline=joblib.load(path); return self
