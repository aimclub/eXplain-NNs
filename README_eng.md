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
1. fix code style to match PEP8 automatically with autopep8
    ```
    $ pip install autopep8==1.6.0
    $ autopep8 --max-line-length=90 -i -r eXNN
    $ autopep8 --max-line-length=90 -i -r tests
    ```
1. check that code style matches PEP8
    ```
    $ pip install pycodestyle==2.8.0
    $ pycodestyle --max-line-length=90 --ignore=E266 eXNN
    $ pycodestyle --max-line-length=90 --ignore=E266 tests
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
We provide tutorials demonstrating suggested usage pipeline:
* [Visualization](/examples/minimal/Visualization.ipynb)
* [Homologies](/examples/minimal/Homologies.ipynb)
* [Bayesianization](/examples/minimal/Bayesianization.ipynb)

We also provide examples of application of our library to different tasks:
* [CIFAR10 classification](/examples/CIFAR10)
* [casting defect detection](/examples/casting)

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