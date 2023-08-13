import numpy as np
import typing as tp
import onnxruntime as ort


class ModelONNX:
    """
    Class for saving and loading modelss
    """

    def __init__(self, config: dict):
        self.path = config['path']
        self.class_names = np.array(config["class_names"])
        self.threshold = config["threshold"]
        self.providers = config["providers"]
        self.input_name = config["input_name"]
        self.output_name = config["output_name"]
        self.ort_session = ort.InferenceSession(
            self.path,
            providers=self.providers
        )

    def predict(self, batch: np.ndarray) -> tp.List[str]:
        ort_inputs = {self.input_name: batch}
        [prediction] = self.ort_session.run(
            [self.output_name], ort_inputs
        )
        return self.class_names[self.threshold < prediction].tolist()

    def predict_proba(self, batch: np.ndarray) -> tp.List[str]:
        ort_inputs = {self.input_name: batch}
        [prediction] = self.ort_session.run(
            [self.output_name], ort_inputs
        )
        return prediction

    def get_classes(self) -> tp.List[str]:
        return self.class_names.tolist()
