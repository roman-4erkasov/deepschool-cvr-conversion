import os
import sys
import timeit
import numpy as np
import cv2
import typing as tp
import torch as th  # for torch script and torch trace
import timeit
import pandas as pd
import timm


class ModelTorch:
    """
    Class for saving and loading models
    """

    def __init__(self, config: dict):
        self.path = config['path']
        self.device = config['device']
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
        with th.no_grad():
            prediction = (
                self.model(batch).
                sigmoid().
                squeeze().
                detach().
                cpu().
                numpy()
            )
        return self.class_names[self.threshold < prediction].tolist()

    def predict_proba(self, batch) -> tp.List[str]:
        with th.no_grad():
            prediction = (
                self.
                model(batch).
                sigmoid().
                squeeze().
                detach().
                cpu().
                numpy()
            )
        return prediction

    def get_classes(self) -> tp.List[str]:
        return self.class_names.tolist()


class ModelTorchJIT:
    """
    Class for saving and loading modelss
    """

    def __init__(self, config: dict):
        self.path = config['path']
        self.device = config['device']
        self.class_names = np.array(config['class_names'])
        self.model_kwargs = config['model_kwargs']
        self.threshold = config['threshold']
        self.model = th.jit.load(self.path)
        self.model.eval()

    def predict(self, batch) -> tp.List[str]:
        with th.no_grad():
            prediction = (
                self.model(batch).
                sigmoid().
                squeeze().
                detach().
                cpu().
                numpy()
            )
        return self.class_names[self.threshold < prediction].tolist()

    def predict_proba(self, batch) -> tp.List[str]:
        with th.no_grad():
            prediction = (
                self.
                model(batch).
                sigmoid().
                squeeze().
                detach().
                cpu().
                numpy()
            )
        return prediction

    def get_classes(self) -> tp.List[str]:
        return self.class_names.tolist()


class ModelONNX:
    """
    Class for saving and loading modelss
    """

    def __init__(self, config: dict):
        self.path = config['path']
        self.class_names = np.array(config['class_names'])
        self.threshold = config['threshold']
        self.providers = config['providers']
        # self.model = th.jit.load(self.path)
        ort_session = ort.InferenceSession(
            self.path,
            providers=self.providers
        )
        self.model.eval()

    def predict(self, batch) -> tp.List[str]:
        with th.no_grad():
            prediction = (
                self.model(batch).
                sigmoid().
                squeeze().
                detach().
                cpu().
                numpy()
            )
        return self.class_names[self.threshold < prediction].tolist()

    def predict_proba(self, batch) -> tp.List[str]:
        with th.no_grad():
            prediction = (
                self.
                model(batch).
                sigmoid().
                squeeze().
                detach().
                cpu().
                numpy()
            )
        return prediction

    def get_classes(self) -> tp.List[str]:
        return self.class_names.tolist()
