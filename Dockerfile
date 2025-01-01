############################################################
# Configure global parameters.
############################################################
ARG UBUNTU_VERSION=22.04
# CUDA 12.X results in compilation errors with COLMAP 3.8.
ARG NVIDIA_CUDA_VERSION=11.7.1
# TODO: Choose the right CUDA_ARCHITECTURES for your GPU according to https://developer.nvidia.com/cuda-gpus.
# Some versions may not be supported by CUDA 11.7.1. E.g. 89.
ARG CUDA_ARCHITECTURES=86



############################################################
# Set the base image.
############################################################
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder
# Set the frontend to noninteractive to prevent prompts like selecting a geographic area or accepting confirmations
ENV DEBIAN_FRONTEND=noninteractive
# Set the default package directory.
WORKDIR /home/ubuntu/packages



############################################################
# Build Ceres 2.1.0 from source.
############################################################
RUN apt-get update && \
	apt-get install -y \
	git \
	cmake \
	libgoogle-glog-dev \
	libgflags-dev \
	libatlas-base-dev \
	libeigen3-dev \
	libsuitesparse-dev
RUN git clone https://github.com/ceres-solver/ceres-solver.git && \
	cd ceres-solver && \
	# Ceres 2.1.0 is compatible with COLMAP 3.8 and PixelSfM 1.0 at the same time.
	git checkout 2.1.0 && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j3 && \
	# make test && \
	make install && \
	cd ../.. && \
	rm -rf ceres-solver



############################################################
# Build and install COLMAP 3.8 from source.
############################################################
RUN apt-get update && \
	apt-get install -y \
	# git \
	# cmake \
	ninja-build \
	build-essential \
	libboost-program-options-dev \
	libboost-filesystem-dev \
	libboost-graph-dev \
	libboost-system-dev \
	libboost-test-dev \
	# libeigen3-dev \
	libflann-dev \
	libfreeimage-dev \
	libmetis-dev \
	# libgoogle-glog-dev \
	# libgflags-dev \
	libsqlite3-dev \
	libglew-dev \
	qtbase5-dev \
	libqt5opengl5-dev \
	libcgal-dev
	# libceres-dev
RUN git clone https://github.com/colmap/colmap.git && \
	cd colmap && \
	git checkout 3.8 && \
	mkdir build && \
	cd build && \
	cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
	ninja && \
	ninja install && \
	cd ../.. && \
	rm -rf colmap



############################################################
# Build and install LightGlue 0.1 from source.
############################################################
RUN apt-get update && \
	apt-get install -y \
	python3 \
	python3-pip
RUN git clone https://github.com/cvg/LightGlue.git && \
	cd LightGlue && \
	git checkout v0.1_arxiv && \
	pip3 install -e .[extra]



############################################################
# Build and install hloc 1.4 from source.
############################################################
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization.git && \
	cd Hierarchical-Localization && \
	git checkout v1.4 && \
	# We have manually installed LightGlue, so we remove it from the requirements.
	sed -i '$ d' requirements.txt && \
	pip3 install -e .



############################################################
# Build and install pixelsfm from source.
############################################################
RUN git clone --recursive https://github.com/cvg/pixel-perfect-sfm && \
	cd pixel-perfect-sfm && \
	pip3 install -r requirements.txt && \
	pip3 install -e .



############################################################
# Setup other dependencies.
############################################################
# To avoid the error "AttributeError: np.NaN was removed in the NumPy 2.0 release. Use np.nan instead." in PixelSfM.
RUN pip install "numpy<2.0" debugpy
RUN apt-get update && \
	apt-get install -y \
	gdb \
	wget \
	sqlite3 \
	libsqlite3-dev
RUN ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
RUN git clone https://github.com/p-ranav/argparse.git && \
	cd argparse && \
	git checkout v3.1 && \
	cp -R ./include/argparse /usr/include && \
	cd .. && \
	rm -rf argparse
RUN git clone https://github.com/rogersce/cnpy.git && \
	cd cnpy && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make && \
	make install && \
	cd ../.. && \
	rm -rf cnpy



############################################################
# Clean up cache to reduce the image size.
############################################################
RUN apt-get clean && \
	rm -rf /var/lib/apt/lists/*



############################################################
# Set the default working directory.
############################################################
WORKDIR /home/ubuntu/