import numpy as np
import typing as tp
import torch as th
import timm


class ModelTorch:
    """
    Class for saving and loading models
    """

    def __init__(self, config: dict):
        self.path = config['path']
        self.device = th.device(config['device'])
        self.class_names = np.array(config['class_names'])
        self.model_kwargs = config['model_kwargs']
        self.threshold = config['threshold']
        # self.model = th.load(self.path)
        state_dict = th.load(
            self.path, map_location=th.device(self.device),
        )
        self.model = timm.create_model(
            num_classes=len(self.class_names),
            **self.model_kwargs,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, batch) -> tp.List[str]:
        batch_th = th.from_numpy(batch)
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
        batch_th = th.from_numpy(batch)
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