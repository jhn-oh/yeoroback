import os, pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelLoader:
    _model = None

    @classmethod
    def load(cls):
        if cls._model is None:
            with open(os.path.join(BASE_DIR, 'yoroapp', 'model', 'tfidf_cluster_model.pkl'), 'rb') as f:
                cls._model = pickle.load(f)
        return cls._model
