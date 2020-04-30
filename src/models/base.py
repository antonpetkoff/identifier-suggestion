from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_x, train_y): pass

    @abstractmethod
    def predict(self, test_x): pass

    @abstractmethod
    def predict_proba(self, test_x): pass

    def get_params(self, deep = True):
        return {}

    def get_grid_params(self):
        return {}