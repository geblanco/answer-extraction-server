MAINTAINER 'gblanco@lsi.uned.es'
FROM python:3.7

# use nvidia if possible

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /usr/app/
COPY /src /usr/app/
COPY ./requirements.txt /usr/app

# models should be grabbed from local dir

# RUN mkdir -p /usr/models && \
#   cd /app/models && \
#   wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip && \
#   unzip uncased_L-12_H-768_A-12.zip && \
#   mv /app/models/uncased_L-12_H-768_A-12/config.json /app/models/uncased_L-12_H-768_A-12/bert_config.json

RUN pip install -r requirements.txt
EXPOSE 9000

CMD ["bert-serving-start" "-num_worker=2" "-cpu" "-fp16" "-http_port 9000" "-model_dir /app/models/bert/uncased_L-12_H-768_A-12/"]
