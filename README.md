# Featuremetric-Calibration

The official implementation of (Name TBD).

## About



## Installation

We recommend running the project in a docker container. All dependencies will be installed if you build the docker image using our `Dockerfile`.

Git clone the repository to the local machine:

```bash
cd /path/to/your/workspace
git clone https://github.com/YJJfish/Featuremetric-Calibration.git
cd Featuremetric-Calibration
```

Choose the right `CUDA_ARCHITECTURES` for your GPU according to https://developer.nvidia.com/cuda-gpus, and set the value in `Dockerfile`:
```dockerfile
# TODO: Choose the right CUDA_ARCHITECTURES for your GPU according to https://developer.nvidia.com/cuda-gpus.
# Some versions may not be supported by CUDA 11.7.1. E.g. 89.
ARG CUDA_ARCHITECTURES=86
```

Also set the value in `src/CMakeLists.txt`:

```cmake
set_target_properties(calibration PROPERTIES CUDA_ARCHITECTURES "86")
```

Build a docker image with name `calibration` and tag `1.0`, using the provided `Dockerfile`:

```bash
docker build -t calibration:1.0 .
```

Run a Docker container with the following options:

 - Mount the workspace folder in the host machine to `/home/ubuntu/workspace` in the container.
 - Specify the maximum RAM available to the container (e.g., 8GB) according to your dataset.
 - Include GPU support for the container by using `--gpus all`.
```bash
cd /path/to/your/workspace
docker run -dit \
	--name calibration-container \
	-v $(pwd):/home/ubuntu/workspace \
	--memory=8g \
	--gpus all
	calibration:1.0
docker exec -it calibration-container bash
```

Inside the container, use `CMake` to generate the project files and use `make` to compile the project:

```bash
cd workspace/Featuremetric-Calibration/src
mkdir build
cd build
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release
make
```

## Tutorial

### Dataset

Download the [multiface dataset](https://github.com/facebookresearch/multiface) (or its mini-dataset).

### Data Preprocessing

The images of multiface dataset are gathered per camera. However, for SfM applications, images are required to be gathered per frame.

Reorder the dataset images using `script/gather_images.py`:

```bash
python3 script/gather_images.py /path/to/dataset/folder /path/to/reordered/dataset/folder
```

Our project functions as an add-on to other SfM pipelines. It uses the sparse reconstruction results of other SfM pipelines and refines the model. We provide `colmap_batch.py` or `pixelsfm_batch.py` to run COLMAP or Pixel-Perfect SfM. We use `pixelsfm_batch.py` as an example:

```bash
python3 script/pixelsfm_batch.py /path/to/reordered/dataset/folder /path/to/sfm/output/folder
```

Optionally, run `extract_dense_features.py` to extract dense features and store them in database. These features will be used to compute featuremetric costs in our objective function if you enable featuremetric refinement.

```bash
python3 script/extract_dense_features.py /path/to/reordered/dataset/folder /path/to/sfm/output/folder /path/to/feature/database --mode pixelsfm
```

Finally, run our project:

```bash
./src/build/calibration /path/to/sfm/output/folder /path/to/feature/database /path/to/dataset/KRT --mode pixelsfm
```