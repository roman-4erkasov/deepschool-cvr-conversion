import numpy as np
import typing as tp
from openvino.runtime import Core


class ModelOpenVino:
    """
    Class for saving and loading modelss
    """

    def __init__(self, config: dict):
        self.path_xml = config['path_xml']
        self.path_bin = config['path_bin']
        self.class_names = np.array(config["class_names"])
        self.threshold = config["threshold"]
        ie = Core()
        model = ie.read_model(
            model=self.path_xml, 
            weights=self.path_bin
        )
        self.model = ie.compile_model(model=model)
        self.output_layer = self.model.output(0)

    def predict(self, batch: np.ndarray) -> tp.List[str]:
        prediction = self.model([batch])[self.output_layer]
        return self.class_names[self.threshold < prediction].tolist()

    def predict_proba(self, batch: np.ndarray) -> tp.List[str]:
        prediction = self.model([batch])[self.output_layer]
        return prediction

    def get_classes(self) -> tp.List[str]:
        return self.class_names.tolist()
