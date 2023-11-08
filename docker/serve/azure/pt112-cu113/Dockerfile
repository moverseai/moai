FROM ghcr.io/ai-in-motion/moai/serve:v1.0.1-pt112-cu113

USER root

RUN apt-get update && apt-get install -y -qq \
    curl \
    apt-transport-https \
    lsb-release \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sL https://packages.microsoft.com/keys/microsoft.asc | \
    gpg --dearmor -o /etc/apt/trusted.gpg.d/microsoft.gpg \
    && echo "deb [arch=amd64] https://packages.microsoft.com/repos/azure-cli/ $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/azure-cli.list

RUN apt-get update && apt-get install -y -qq azure-cli

# install azure python sdk
RUN pip install azure-core \
    azure-storage-blob

# cache cleanup
RUN chown -R worker /workspace \
    && chmod 755 /workspace \
    && pip cache purge && apt-get clean

USER worker
WORKDIR /workspace
