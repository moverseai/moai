ARG VERSION
FROM ghcr.io/moverseai/moai/serve:${VERSION}-pt220-cu118

USER root

RUN apt-get update && apt-get install -y -qq \
    curl \
    apt-transport-https \
    lsb-release \
    gnupg \
    git \
    && rm -rf /var/lib/apt/lists/*

# install iopath python sdk with verbose output to see what's failing
RUN pip install --verbose git+https://github.com/moverseai/iopath[gcs,az] || \
    (echo "Failed to install with extras, trying base package..." && \
     pip install git+https://github.com/moverseai/iopath && \
     pip install google-cloud-storage azure-storage-blob)


RUN curl -sL https://packages.microsoft.com/keys/microsoft.asc | \
    gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/azure-cli.list

RUN apt-get update && apt-get install -y -qq azure-cli

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

RUN apt-get update && apt-get install -y -qq google-cloud-sdk

# cache cleanup
RUN chown -R worker /workspace \
    && chmod 755 /workspace \
    && pip cache purge && apt-get clean

USER worker
WORKDIR /workspace
