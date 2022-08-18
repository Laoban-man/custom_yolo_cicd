# syntax=docker/dockerfile:1
FROM ubuntu:20.04
WORKDIR /code
RUN apt update
RUN apt-get install -y \
    libpng-dev \
    freetype* \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran
RUN apt-get install -y gcc musl-dev python3-pip libgl1
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install -U torch torchvision scikit-learn wandb
RUN apt-get install git -y
RUN git clone https://github.com/ultralytics/yolov5  # clone
RUN cd yolov5
RUN pip install -r requirements.txt  # install
COPY ./code /code
ENTRYPOINT [ "python3" ]
CMD [ "main.py" ]
