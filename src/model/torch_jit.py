import numpy as np
import typing as tp
import torch as th



class ModelTorchJIT:
    """
    Class for saving and loading modelss
    """

    def __init__(self, config: dict):
        self.path = config['path']
        self.device = config['device']
        self.class_names = np.array(config['class_names'])
        self.threshold = config['threshold']
        self.model = th.jit.load(
            self.path, map_location=th.device(self.device)
        )
        self.model.eval()

    def predict(self, batch) -> tp.List[str]:
        batch_th = (
            th.from_numpy(batch)
            .to(th.device(self.device))
        )
        with th.no_grad():
            prediction = (
                self.model(batch_th).
                sigmoid().
                squeeze().
                detach().
                cpu().
                numpy()
            )
        return self.class_names[self.threshold < prediction].tolist()

    def predict_proba(self, batch) -> tp.List[str]:
        batch_th = (
            th.from_numpy(batch)
            .to(th.device(self.device))
        )
        with th.no_grad():
            prediction = (
                self.
                model(batch_th).
                sigmoid().
                squeeze().
                detach().
                cpu().
                numpy()
            )
        return prediction

    def get_classes(self) -> tp.List[str]:
        return self.class_names.tolist()

