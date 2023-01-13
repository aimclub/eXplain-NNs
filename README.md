![logo](/docs/logo.png)

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
![itmo_logo](/docs/itmo_logo_small.png)

Библиотка разработана [Университетом ИТМО](https://itmo.ru/).

### Разработчики
* А. Ватьян - тим лид
* Н. Гусарова - научный руководитель
* И. Томилов
* Т. Полевая
* К. Никулина

## Контакты
* Александра Ватьян alexvatyan@gmail.com по вопросам сотрудничества
* Татьяна Полевая tpolevaya@itmo.ru по техническим вопросам