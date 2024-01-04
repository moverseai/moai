# Hydra-Graylog Plugin

## Description
The Hydra-Graylog plugin integrates Hydra with Graylog for advanced logging capabilities. It enhances Hydra applications with efficient log management and visualization.

## Installation
Install the plugin using pip:
```bash
pip install hydra-graylog
```

## Usage
Configure Hydra to use the plugin by adding it to your Hydra configuration:
```yaml
defaults:
  - graylog/job_logging: graylog
  - graylog/hydra_logging: graylog
```

## Features
- Seamless integration with Graylog for logging
- Supports custom log formats and levels
- Easy configuration within Hydra's ecosystem

## Requirements
- Hydra-core (version specific)
- graypy

## Author
Georgios Albanis

## License
MIT License

For more details and updates, visit our [GitHub repository](https://github.com/ai-in-motion/moai).