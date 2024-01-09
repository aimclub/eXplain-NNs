<p align="center">
    <img src="/docs/logo.png" width="500">
</p>

[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg)](https://en.itmo.ru/en/)

[![Documentation](https://github.com/aimclub/eXplain-NNs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://med-ai-lab.github.io/eXplain-NNs-documentation/)
[![license](https://img.shields.io/github/license/aimclub/eXplain-NNs)](https://github.com/aimclub/eXplain-NNs/blob/main/LICENSE)
[![Rus](https://img.shields.io/badge/lang-ru-yellow.svg)](/README.md)
[![Mirror](https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162)](https://gitlab.actcognitive.org/itmo-sai-code/eXplain-NNs)

# eXplain-NNs
This repository contains eXplain-NNs Library - an open-source library with explainable AI (XAI) methods for analyzing neural networks. This library provides several XAI methods for latent spaces analysis and uncertainty estimation.

## Project Description

### eXplain-NNs methods
XAI methods implemented in the library
1. visualization of latent spaces
1. homology analysis of latent spaces
1. uncertainty estimation via bayesianization

Thus compared to other XAI libraries, eXplain-NNs Library:
* Provides homology analysis of latent spaces
* Impelemnts a novel method of uncertainty estimation via bayesianization

Details of [implemented methods](/docs/methods.md).

### Data Requirement
* The library supports only models that are:
    * fully connected or convolutional
    * designed for classification task

## Installation
Requirements: Python 3.8
1. [optional] create Python environment, e.g.
    ```
    $ conda create -n eXNN python=3.8
    $ conda activate eXNN
    ```
1. install requirements from [requirements.txt](/requirements.txt)
    ```
    $ pip install -r requirements.txt
    ```
1. install the library as a package
    ```
    $ python -m pip install git+ssh://git@github.com/Med-AI-Lab/eXplain-NNs
    ```


## Development
Requirements: Python 3.8
1. [optional] create Python environment, e.g.
    ```
    $ conda create -n eXNN python=3.8
    $ conda activate eXNN
    ```
1. clone repository and install all requirements
    ```
    $ git clone git@github.com:Med-AI-Lab/eXplain-NNs.git
    $ cd eXplain-NNs
    $ pip install -r requirements.txt
    ```
1. run tests
    ```
    $ pytest tests/tests.py
    ```
1. fix code style to match PEP8 automatically
    ```
    $ make format
    ```
1. check that code style matches PEP8
    ```
    $ make check
    ```
1. build a PyPi package locally
    ```
    $ python3 -m pip install --upgrade build
    $ python3 -m build
    ```

## Documentation
The general description is available [here](https://med-ai-lab.github.io/eXplain-NNs-documentation/).

eXplain-NNs Library API is available [here](https://med-ai-lab.github.io/eXplain-NNs-documentation/api_docs/eXNN.html)


## Examples & Tutorials
We provides examples of different levels of complexity:
* [minimal] minimalistic examples presenting our API
* [basic] applying eXNN to simple tasks like MNIST classification
* [use cases] demonstation of eXplain-NN usage for solving different use cases in industrial tasks

### Minimal
This colab contains minimalistic demonstration of our API on dummy objects:

[![minimal](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lOiB50LppDiiRHTv184JMuQ2IvZ4I4rp?usp=sharing)

### Basic
Here are colabs demonstrating how to work with different modules of our API on simple tasks:
| Colab Link | Module |
| ------------- | ------------- |
| [![bayes](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ayd0IronxUIfnbAmWQLHiILG2qtBBpF4?usp=sharing)| bayes |
| [![topology](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1T5ENfNaCIRI61LM2ZhtU8lfmvRmlfiEo?usp=sharing)| topology |
| [![visualization](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LJVdWTv-wcASSMX4is_E15TR7XJsT7W3?usp=sharing)| visualization |

### Use Cases
This block provides examples how eXplain-NNs can be used to solve different use cases in industrial tasks. For demonstration purposed 3 tasks are used:
* [satellite] landscape classification from satellite imagery
* [electronics] electronic components and devices classification
* [ECG] ECG diagnostics

| Colab Link | Task | Use Case |
| ------------- | ------------- | ------------- |
| [![CNN_viz](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12ZJigH-0geGTefNXnCM5dQ71d4tqlf6L?usp=sharing)| satellite | Visualization of data manifold evolution from layer to layer |
| [![adv](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n50WUu2ZKZ6nrT9DuFD3q87m3yZvxkwm?usp=sharing) | satellite | Detecting adversarial examples |
| [![generalize](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mG-VrP7J7OoCvIQDl7n5YWEIdyfFg_0I?usp=sharing) | electronics | Estimating generalization of a NN |
| TBD | ECG | Visualization of data manifold evolution from layer to layer |

## Contribution Guide
The contribution guide is available in the [repository](/docs/contribution.md).

## Acknowledgments

### Affiliation
[ITMO University](https://en.itmo.ru/).

### Supported by
The study is supported by the [Research Center Strong Artificial Intelligence in Industry](<https://sai.itmo.ru/>) 
of [ITMO University](https://en.itmo.ru/) as part of the plan of the center's program: Development and testing of an experimental prototype of a library of strong AI algorithms in terms of algorithms for explaining the results of modeling on data using the semantics and terminology of the subject and problem areas in tasks with high uncertainty, including estimation of the uncertainty of neural network models predictions, analysis and visualization of interlayer transformations of the input variety inside neural networks.

### Developers
* A. Vatyan - team leader
* N. Gusarova - chief scientific advisor
* I. Tomilov
* T. Polevaya
* Ks. Nikulina

## Contacts
* Alexandra Vatyan alexvatyan@gmail.com for collaboration suggestions
* Tatyana Polevaya tpolevaya@itmo.ru for technical questions
