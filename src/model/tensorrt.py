import numpy as np
import typing as tp
import torch as th

class ModelTensorRT:
    """
    Class for saving and loading modelss
    """

    def __init__(self, config: dict):
        self.path = config['path']
        self.class_names = np.array(config["class_names"])
        self.threshold = config["threshold"]
        self.model = th.jit.load(self.path)

    def predict(self, batch: np.ndarray) -> tp.List[str]:
        prediction = self.model(batch)
        return self.class_names[self.threshold < prediction].tolist()

    def predict_proba(self, batch: np.ndarray) -> tp.List[str]:
        prediction = self.model(batch)
        return prediction

    def get_classes(self) -> tp.List[str]:
        return self.class_names.tolist()
