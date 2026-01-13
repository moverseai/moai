FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y -qq \
    wget \
    zip \
    libgl1 \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

COPY . /moai

# Install torch and torchvision
RUN conda install -c conda-forge -c nvidia \
   --override-channels -y \
  pytorch torchvision
#  pymomentum

WORKDIR /moai
RUN pip install -e . --config-settings editable_mode=compat
# CHECK https://github.com/facebookresearch/hydra/issues/2359
# CHECK https://setuptools.pypa.io/en/latest/userguide/development_mode.html#legacy-behavior
# CHECK https://github.com/facebookresearch/hydra/blob/abb36af33e83f391a22d6269fd2f317378171ec9/hydra/core/plugins.py

# newer setuptools uses "strict" editable installs via virtual hooks) 
# while Hydra’s discovery mechanism relies on scanning physical directory structures.

RUN python -m pip uninstall -y torchtext
RUN pip install \
    torchserve==0.10.0 \
    torch-model-archiver==0.10.0

COPY ./docker/serve/config.properties /workspace/config.properties

EXPOSE 7070/tcp 7071/tcp 8080/tcp 8081/tcp 8082/tcp

COPY docker/serve/serve.sh /workspace/serve.sh
RUN chmod +x /workspace/serve.sh

RUN useradd -m worker \
    && mkdir -p /workspace/models \
    && mkdir -p /workspace/logs \
    && mkdir -p /workspace/data \
    && mkdir -p /workspace/run \
    && chown -R worker /workspace \
    && chmod 755 /workspace \
    && chown -R worker /workspace/data \
    && chmod 755 /workspace/data \
    && chown -R worker /workspace/logs \
    && chmod 755 /workspace/logs \
    && chown -R worker /workspace/run \
    && chmod 755 /workspace/run

ENV LOG_LOCATION=/workspace/logs
ENV METRICS_LOCATION=/workspace/logs

RUN pip cache purge && apt-get clean

USER worker
WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "/workspace/serve.sh"]
