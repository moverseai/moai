torchserve --start --ncs --ts-config /workspace/config.properties \
  --model-store /workspace/models --models all
tail -f /dev/null