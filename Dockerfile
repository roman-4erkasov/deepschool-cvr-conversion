ARG TRT_CONTAINER_VERSION=23.04
FROM nvcr.io/nvidia/pytorch:${TRT_CONTAINER_VERSION}-py3

RUN apt-get update
RUN apt-get install libgl1-mesa-glx  -y
# RUN pip install --no-cache-dir ipywidgets
# RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install -U pip --no-cache-dir
#RUN pip install --no-cache-dir pycuda==2022.1 segmentation_models_pytorch timm opencv-python-headless==4.4.0.44 opencv-python==4.4.0.44 numpy==1.23
RUN pip install --no-cache-dir timm==0.9.2 pyyaml==6.0 
RUN pip install --no-cache-dir onnxruntime-gpu==1.14.1 torchmetrics==0.11.4


WORKDIR /workspace/project
CMD [ "/bin/bash" ]
