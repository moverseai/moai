<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/2.0.2/anime.min.js"></script>

<h1 class="ml4">
  <span class="letters letters-1">Modular</span>
  <span class="letters letters-2">Reproducible</span>
  <span class="letters letters-3">Minimal</span>
  <span class="letters letters-4">Extensible</span>
  <span class="letters letters-5">Insightful</span>
</h1>

# ***moai***: Accelerating modern data-driven workflows

**_moai_** is a [PyTorch](https://pytorch.org/)-based AI Model Development Kit (MDK) that aims to **improve data-driven model workflows, design and understanding**.
Since it is based on established [open-source packages](#Dependencies), it can be readily used to improve most AI workflows. To explore _moai_, simply [install](#Installation) the package and follow the [examples](https://github.com/ai-in-motion/moai/tree/master/conf/examples), having in mind that it is in early development _alpha_ version, thus new features will be available soon.

## Features & Design Goals

- **Modularity via Monads**: Use _moai_'s existing pool of modular _model building blocks_.
- **Reproducibility via Configuration**: _moai_ manages the hyper-parameter sensitive AI R&D workflows via its built-in _configuration-based design_.
- **Productivity via Minimizing Coding**: _moai_ offers a _data-driven domain modelling language_ (DML) that can facilitates quick & easy model design.
- **Extensibility via Plugins**: Easily integrate external code using _moai_'s built-in metaprogramming and _external code integration_. 
- **Understanding via Analysis**: _moai_ supports _inter-model performance and design aggregation_ [**actions**](#) to consolidate knowledge between models and query differences.

## Dependencies

_moai_ stands on the shoulders of giants as it relies on various large scale open-source projects:

1. [PyTorch](https://pytorch.org/) `> 1.7.0` needs to be customly installed on your system/environment.
2. [Lightning](https://www.pytorchlightning.ai/) `> 1.0.0` is the currently supported training backend.
3. [Hydra](https://hydra.cc/) `> 1.0` drives _moai_'s DML that sets up model configurations, and additionally manages the hyper-parameter complexity of modern AI models.
4. [Visdom](https://github.com/fossasia/visdom) is the currently supported visualization engine.
5. [HiPlot](https://github.com/facebookresearch/hiplot) drives _moai_'s inter-model analytics.
6. [Various PyTorch Open Source Projects](#Dependencies):
    
    - [Kornia](https://github.com/kornia/kornia) for a set of computer vision operations integrated as _moai_ monads.
    - [Albumentations](https://github.com/albumentations-team/albumentations) as the currently supported data augmentation framework.

7. [The Wider Open Source Community](www.github.com) that conducts accessible R&D and drives most of _moai_'s capabilities.

8. [A set of awesome Python libraries](#Dependencies) as found in our [**requirements**](https://github.com/ai-in-motion/moai/tree/master/requirements.txt) file.

## Installation

### Package
To install the currently **released** _moai_ version package run:

`pip install moai-mdk`

### Source
Download the master branch source and install it by opening a command line on the source directory and running:

`pip install .` or `pip install -e .` (in editable form)

## Getting Started

Visit the [**documentation**](#) site to learn about _moai_'s DML and the overall MDK design and usage.

The examples found at [**conf/examples**](https://github.com/ai-in-motion/moai/tree/master/conf/examples) will be documented **soon**.

## Documentation
[Documentation](https://moai.readthedocs.io/en/latest/) is currently a **work-in-progress**.

## API
The API documentation is currently a **work-in-progress**.

## How-to extend moai with custom functionalities (external project)
In the case one needs to create a project beyond the existing functionality of the framework, he/she can go beyond the standard configuration.
Custom modules can be developed and integrated to the framework.

To do so, the steps below should be followed:

*  Create a folder where the custom code lives ({custom_project}).
*  In this folder, create a configuration folder (e.g. '{custom_project}/conf'), to place the .yaml files of the new moduldes.
*  Create a source folder (e.g. '{custom_project}/src'), to place the implementations of the new modules.
*  