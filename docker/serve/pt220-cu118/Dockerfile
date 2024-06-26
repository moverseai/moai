FROM  pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

COPY . /moai

RUN apt-get update \
    && apt-get install -y -qq \
    zip \
    libgl1 \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip uninstall -y torchtext
RUN pip install \
    -e /moai \
    torchserve==0.10.0 \
    # onnx==1.8.1 \
    torch-model-archiver==0.10.0
    # onnxruntime==1.6.0

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
