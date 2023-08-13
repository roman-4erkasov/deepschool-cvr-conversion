# docker version
VERSION:=23.04
IMAGE_NAME=trt
CONTAINER_NAME=trt

.PHONY: *

prep_cuda:
	echo "Please do the following commands:"
	echo " 1) wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run"
	echo " 2) sudo sh cuda_11.6.0_510.39.01_linux.run" 

venv_cpu:
	rm -rf venv
	python3.8 -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements_cpu.txt


venv_gpu:
	rm -rf venv
	python3.8 -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install -r requirements_gpu.txt


make_docker:
	docker build -t trt .

download_weights:
	dvc pull


docker_lab:
ifdef DIR_DATA
	docker run \
	  --net host \
	  --gpus all \
	  --rm \
	  -v ${CURDIR}:/workspace/project \
	  -v ${DIR_DATA}:/workspace/data \
	  ${IMAGE_NAME} jupyter lab
else
	docker run \
	  --net host \
	  --gpus all \
	  --rm \
	  -v ${CURDIR}:/workspace/project \
	  ${IMAGE_NAME} jupyter lab
endif

docker_list:
	docker run \
	  --net host \
	  --gpus all \
	  --rm \
	  -v ${CURDIR}:/workspace/project \
	  ${IMAGE_NAME} jupyter notebook list

docker_ls:
	sudo docker run \
	  --net host \
	  --gpus all \
	  --rm \
	  -v ${CURDIR}:/workspace/project \
	  ${IMAGE_NAME} ls ${ARGS}


