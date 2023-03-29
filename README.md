<p align="center">
    <img src="/docs/logo.png" width="500">
</p>

[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg)](https://en.itmo.ru/en/)

[![Documentation](https://github.com/aimclub/eXplain-NNs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://med-ai-lab.github.io/eXplain-NNs-documentation/)
[![license](https://img.shields.io/github/license/aimclub/eXplain-NNs)](https://github.com/aimclub/eXplain-NNs/blob/main/LICENSE)
[![Eng](https://img.shields.io/badge/lang-en-red.svg)](/README_eng.md)
[![Mirror](https://camo.githubusercontent.com/9bd7b8c5b418f1364e72110a83629772729b29e8f3393b6c86bff237a6b784f6/68747470733a2f2f62616467656e2e6e65742f62616467652f6769746c61622f6d6972726f722f6f72616e67653f69636f6e3d6769746c6162)](https://gitlab.actcognitive.org/itmo-sai-code/eXplain-NNs)

# eXplain-NNs
Этот репозиторий содержит библиотеку eXplain-NNs — библиотеку с открытым исходным кодом с методами объяснимого ИИ (XAI) для анализа нейронных сетей. Эта библиотека предоставляет несколько методов XAI для анализа латентных пространств и оценки неопределенности.

## Описание проекта

### Методы
Методы XAI, реализованные в библиотеке
1. визуализация латентных пространств
1. гомологический анализ латентных пространств
1. оценка неопределенности с помощью байесианизации

Таким образом, по сравнению с другими библиотеками объяснимого ИИ библиотека eXplain-NNs:
* Обеспечивает анализ гомологий латентных пространств
* Внедряет новый метод оценки неопределенности с помощью байесианизации XAI для анализа латентных пространств и оценки неопределенности.

Детали [реализации методов](/docs/methods.md).

### Data Requirement
* The library supports only models that are:
    * fully connected or convolutional
    * designed for classification task

## Installation
Требования: Python 3.8
1. [optional] создайте среду окружения Python, e.g.
    ```
    $ conda create -n eXNN python=3.8
    $ conda activate eXNN
    ```
1. установите зависимости из [requirements.txt](/requirements.txt)
    ```
    $ pip install -r requirements.txt
    ```
1. установите библиотеку как пакет
    ```
    $ python -m pip install git+ssh://git@github.com/Med-AI-Lab/eXplain-NNs
    ```


## Development
Требования: Python 3.8
1. [optional] создайте среды окружения Python, e.g.
    ```
    $ conda create -n eXNN python=3.8
    $ conda activate eXNN
    ```
1. клонируйте репозиторий и установите зависимости
    ```
    $ git clone git@github.com:Med-AI-Lab/eXplain-NNs.git
    $ cd eXplain-NNs
    $ pip install -r requirements.txt
    ```
1. запуск тестов
    ```
    $ pytest tests/tests.py
    ```
1. приведение стиля кода в соотвествие с PEP8 автоматически с помощью autopep8
    ```
    $ pip install autopep8==1.6.0
    $ autopep8 --max-line-length=90 -i -r eXNN
    $ autopep8 --max-line-length=90 -i -r tests
    ```
1. проверка стиля кода на соотвествие с PEP8
    ```
    $ pip install pycodestyle==2.8.0
    $ pycodestyle --max-line-length=90 --ignore=E266 eXNN
    $ pycodestyle --max-line-length=90 --ignore=E266 tests
    ```
1. создание PyPi пакета локально
    ```
    $ python3 -m pip install --upgrade build
    $ python3 -m build
    ```

## Документация
[Документация](https://med-ai-lab.github.io/eXplain-NNs-documentation/)

[API](https://med-ai-lab.github.io/eXplain-NNs-documentation/api_docs/eXNN.html)


## Примеры и тьюториалы
Примеры использования API библиотеки:
* [Визуализация](/examples/minimal/Visualization.ipynb)
* [Гомологии](/examples/minimal/Homologies.ipynb)
* [Байесианизация](/examples/minimal/Bayesianization.ipynb)

Примеры использования библиотеки для индустриальных задач:
* [CIFAR10 classification](/examples/CIFAR10)
* [casting defect detection](/examples/casting)

## Как помочь проекту
[Инструкции](/docs/contribution.md).

## Организационное

### Аффилиация
[Университет ИТМО](https://en.itmo.ru/).

### Поддержка
Исследование проводится при поддержке [Исследовательского центра сильного искусственного интеллекта в промышленности](<https://sai.itmo.ru/>) [Университета ИТМО](https://itmo.ru) в рамках мероприятия программы центра: Разработка и испытания экспериментального образца библиотеки алгоритмов сильного ИИ в части алгоритмов объяснения результатов моделирования на данных с использованием семантики и терминологии предметной и проблемной областей в задачах с высокой неопределенностью, включая оценку неопределенности предсказаний моделей нейронных сетей, а также анализ и визуализацию межслойных трансформаций входного многообразия внутри нейронных сетей.

### Разработчики
* А. Ватьян - тим лид
* Н. Гусарова - научный руководитель
* И. Томилов
* Т. Полевая
* К. Никулина

## Контакты
* Александра Ватьян alexvatyan@gmail.com по вопросам сотрудничества
* Татьяна Полевая tpolevaya@itmo.ru по техническим вопросам