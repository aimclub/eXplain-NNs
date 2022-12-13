![logo](docs/logo.png)

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

Details of [implemented methods](docs/methods.md).

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
1. install requirements from [requirements.txt](requirements.txt)
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

## Documentation
TBD

## Examples & Tutorials
TBD

## Contribution Guide
TBD

## Acknowledgments
### Affiliation
![itmo_logo](docs/itmo_logo_small.png)

The library was developed in [ITMO University](https://en.itmo.ru/).

### Developers
* A. Vatyan - team leader
* N. Gusarova - chief scientific advisor
* I. Tomilov
* T. Polevaya
* Ks. Nikulina

## Contacts
* Alexandra Vatyan alexvatyan@gmail.com for collaboration suggestions
* Tatyana Polevaya tpolevaya@itmo.ru for technical questions