FROM continuumio/miniconda3:latest

RUN apt update && apt -y install libusb-1.0-0 libgl1-mesa-glx build-essential
RUN conda install python=3.8.11  # later versions are not compatible to Open3D 0.13
RUN conda install cython numpy scipy matplotlib pandas pyyaml numba sphinx numpydoc pdoc3 nose coverage
RUN pip install gmr pytransform3d[all] tqdm sphinx-gallery sphinx-bootstrap-theme

# docker build . -t conda_py38
# docker tag conda_py38 af01/conda_py38
# docker push af01/conda_py38
