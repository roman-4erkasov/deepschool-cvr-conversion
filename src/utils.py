import os
import cv2
import time
import numpy as np
import typing as tp
import torch as th

MAX_UINT8 = 255


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess_image(
    image: np.ndarray, 
    target_image_size: tp.Tuple[int, int] = (MAX_UINT8, MAX_UINT8)
):
    """
    Препроцессинг имаджнетом.
    :param image: RGB изображение;
    :param target_image_size: целевой размер изображения;
    :return: батч с одним изображением.
    """
    image = image.astype(np.float32)
    image = cv2.resize(image, target_image_size) / MAX_UINT8
    image = np.transpose(image, (2, 0, 1))
    image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    image /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return image


def get_image(path):
    image = load_image(path)
    img = preprocess_image(image, (MAX_UINT8, MAX_UINT8))
    return img


def get_batch(dir_path, n, ext="jpg"):
    pathes = [
        os.path.join(dir_path, x)
        for x in os.listdir(dir_path) if x.endswith(ext)
    ][:n]
    images = []
    for path in pathes:
        image = get_image(path)
        images.append(image)
    batch = np.stack(images)
    return batch


def benchmark(
    model: tp.Any,
    input_shape: tp.Tuple[int] = (1, 3, MAX_UINT8, MAX_UINT8),
    nwarmup: int = 50,
    nruns: int = 10000,
    print_step: int = 1000,
    verbose: bool = False
):
    if verbose:
        print("Warm up ...")
    for _ in range(nwarmup):
        input_data = np.random.rand(*input_shape).astype(np.float32)
        features = model.predict_proba(input_data)
    if verbose:
        print("Start timing ...")

    timings = []
    for i in range(1, nruns + 1):
        input_data = np.random.rand(*input_shape).astype(np.float32)
        start_time = time.time()
        features = model.predict_proba(input_data)
        end_time = time.time()
        timings.append(end_time - start_time)
        if verbose and i % print_step == 0:
            print(
                f"Iteration {i}/{nruns}, "
                f"avg batch time {np.mean(timings) * 1000:.2f} "
                f"± {np.std(timings) * 1000:.2f} ms."
            )
    if verbose:
        print(f'Input shape: {input_data.shape}')
        print(f'Output features size: {features.shape}')
        print(f'Average throughput: {input_shape[0] / np.mean(timings):.3f} images/second')
    return np.mean(timings), np.std(timings)