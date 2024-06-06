# _moai_ - Accelerating modern data-driven workflows

[![Documentation Status](https://readthedocs.org/projects/moai/badge/?version=latest)](https://moai.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fmoverseai%2Fmoai%2Fbadge%3Fref%3Dmain&style=popout)](https://actions-badge.atrox.dev/moverseai/moai/goto?ref=main)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)
![TorchServe](https://img.shields.io/badge/TorchServe-%23EE4C2C.svg?logo=PyTorch&logoColor=white)

<!-- https://actions-badge.atrox.dev/ & https://github.com/atrox/github-actions-badge?tab=readme-ov-file -->

# Overview

**_moai_** is a [PyTorch](https://pytorch.org/)-based AI Model Development Kit (MDK) that aims to **improve data-driven model workflows, design and understanding**.
Since it is based on established [open-source packages](#Dependencies), it can be readily used to improve most AI workflows. To explore _moai_, simply [install](#Installation) the package and follow the [examples](https://github.com/moverseai/moai/tree/master/conf/examples), having in mind that it is in early development _alpha_ version, thus new features will be available soon.

![Overview](https://github.com/moverseai/moai/raw/main/docs/diagrams/overview_light.png#gh-light-mode-only)
![Overview](https://github.com/moverseai/moai/raw/main/docs/diagrams/overview_dark.png#gh-dark-mode-only)

# Features & Design Goals

- **Modularity via Monads**: Use _moai_'s existing pool of modular _model building blocks_.
- **Reproducibility via Configuration**: _moai_ manages the hyper-parameter sensitive AI R&D workflows via its built-in _configuration-based design_.
- **Productivity via Minimizing Coding**: _moai_ offers a _data-driven domain modelling language_ (DML) that can facilitate quick & easy model design.
- **Extensibility via Plugins**: Easily integrate external code using _moai_'s built-in metaprogramming and _external code integration_. 
- **Understanding via Analysis**: _moai_ supports _inter-model performance and design aggregation_ [**actions**](#Actions) to consolidate knowledge between models and query differences.

# Actions

**_moai_** offers a set of data-driven workflow functionalities through specific integrated **actions**. These consume _moai_ **configuration files** that describe each action's executed context.
As _moai_ is built around these configuration files that define its context and describe each model's details, it offers actions that support heavy data-driven workflows with inter-model analytics, knowledge extraction and meticulous reproduction.

Details for each action follow:

- _`moai`_ **`play`** `CONFIG_PATH`

![Play Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/play_light.png#gh-light-mode-only "Play Action")
![Play Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/play_dark.png#gh-dark-mode-only "Play Action")

Using the `play` action, _moai_ starts the playback of a dataset's `train\val\test` splits. _moai_'s exporters can be used to the extract dataset specific statistics. _moai_'s visualization engine can be used to showcase the dataset. Optionally, monad processing graphs can be defined to transform the data.

- _`moai`_ **`train`** `CONFIG_PATH`

![Train Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/train_light.png#gh-light-mode-only "Train Action")
![Train Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/train_dark.png#gh-dark-mode-only "Train Action")

The `train` action consumes a configuration file that defines the model that will be trained, the data that will be used to train and validate it, as well as configurating the engine around the training process.
The results include model states across training and logs including validation metrics and losses.

- _`moai`_ **`evaluate`** `CONFIG_PATH`

![Evaluate Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/evaluate_light.png#gh-light-mode-only "Evaluate Action")
![Evaluate Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/evaluate_dark.png#gh-dark-mode-only "Evaluate Action")

The `evaluate` action consumes a configuration file that defines the trained model that will be tested, the test data, as well as configurating the engine around the testing process.
The results include model aggregated and/or detailed metrics, and inference samples.

- _`moai`_ **`plot`** `PATH_TO_EXPERIMENTS`

![Plot Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/plot_light.png#gh-light-mode-only "Plot Action")
![Plot Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/plot_dark.png#gh-dark-mode-only "Plot Action")

The `plot` action consumes various configuration files - usually from different versions of the same model - and generates a visualization consolidating and aggregating inter-model performance, providing the necessary means to analyze the behaviour of different hyper-parameters or model configurations.

- _`moai`_ **`diff`** `lhs=PATH_TO_CONFIG_A` `rhs=PATH_TO_CONFIG_B`

![Diff Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/diff_light.png#gh-light-mode-only "Diff Action")
![Diff Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/diff_dark.png#gh-dark-mode-only "Diff Action")

The `diff` action consumes two different configuration file - usually from different versions of the same model - and reports their differences related to hyper-parameterization, processing graph variations, etc..

- _`moai`_ **`reprod`** `PATH_TO_RESOLVED_CONFIG`

![Reprod Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/reprod_light.png#gh-light-mode-only "Reprod Action")
![Reprod Action](https://github.com/moverseai/moai/raw/main/docs/diagrams/reprod_dark.png#gh-dark-mode-only "Reprod Action")

The `reprod` action consumes a previously logged and resolved configuration file, and facilitates its reproduction by re-executing it while adjusting to development environment differences.

# Dependencies

_moai_ stands on the shoulders of giants as it relies on various large scale open-source projects:

1. [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/) `> 1.7.0` needs to be customly installed on your system/environment.
2. [![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai) `> 1.0.0` is the currently supported training backend.
3. [Hydra](https://hydra.cc/) `> 1.0` drives _moai_'s DML that sets up model configurations, and additionally manages the hyper-parameter complexity of modern AI models.
4. [![TorchServe](https://img.shields.io/badge/TorchServe-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/serve/) `> 0.5.3` is needed to deploy models as services.
5. [![ONNX](https://img.shields.io/badge/ONNX-%23EE4C2C.svg?logo=onnx&logoColor=white&color=0EB0EC)](https://onnx.ai/) `> 1.11.0` is needed to export models in an exchangeable format.
6. [Visdom](https://github.com/fossasia/visdom) is the currently supported visualization engine.
7. [HiPlot](https://github.com/facebookresearch/hiplot) drives _moai_'s inter-model analytics.
8. [Various PyTorch Open Source Projects](#Dependencies):
    
    - [Kornia](https://github.com/kornia/kornia) for a set of computer vision operations integrated as _moai_ monads.
    - [Albumentations](https://github.com/albumentations-team/albumentations) as the currently supported data augmentation framework.

7. [The Wider Open Source Community](https://www.github.com) that conducts accessible R&D and drives most of _moai_'s capabilities.

8. [A set of awesome Python libraries](https://github.com/moverseai/moai/tree/master/requirements.txt).

# Installation

## Package
To install the latest **released** _moai_ package run:

`pip install moai-mdk`

## Source
Download the master branch source and install it by opening a command line on the source directory and running:

`pip install .` or `pip install -e .` (in editable form)

# Getting Started

Visit the [**documentation**](https://moai.readthedocs.io/) site to learn about _moai_'s DML and the overall MDK design and usage.

Examples can be found at [**conf/examples**](https://github.com/moverseai/moai/tree/main/moai/conf/examples). 

# Licence

_moai_ is Apache 2.0 licenced, as found in the corresponding [LICENCE](https://github.com/moverseai/moai/blob/main/LICENSE) file.

However, some code integrated from external projects may carry their own licences.

# PyTorch Developer's Day 2021
![PTDD21](https://github.com/moverseai/moai/raw/main/docs/images/poster.jpeg#gh-light-mode-only)
![PTDD21](https://github.com/moverseai/moai/raw/main/docs/images/poster.jpeg#gh-dark-mode-only)

# Citation
If you use _moai_ in your R&D workflows or find its code useful please consider citing:

```
@misc{moai,
    key = {moai: PyTorch Model Development Kit},
    title = {{\textit{moai}: Accelerating modern data-driven workflows}},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/moverseai/moai}},
}
```

# Contact

Use a [GitHub issue tracker](https://github.com/moverseai/moai/issues)
