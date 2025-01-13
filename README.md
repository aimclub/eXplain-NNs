<p align="center">
    <img src="/docs/banner.png">
</p>

[![SAI](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/SAI_badge_flat.svg)](https://sai.itmo.ru/)
[![ITMO](https://github.com/ITMO-NSS-team/open-source-ops/blob/master/badges/ITMO_badge_flat_rus.svg)](https://en.itmo.ru/en/)

[![Documentation](https://github.com/aimclub/eXplain-NNs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://med-ai-lab.github.io/eXplain-NNs-documentation/)
[![license](https://img.shields.io/github/license/aimclub/eXplain-NNs)](https://github.com/aimclub/eXplain-NNs/blob/main/LICENSE)
[![Eng](https://img.shields.io/badge/lang-en-red.svg)](/README_eng.md)
[![Mirror](https://img.shields.io/badge/mirror-GitLab-orange)](https://gitlab.actcognitive.org/itmo-sai-code/eXplain-NNs)

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

Видео с процессом установки можно посмотреть [здесь](https://drive.google.com/file/d/1Sv8UiRwWfMLJ0kOSYHB_PgILHzNcqfs0/view?usp=sharing).


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
1. приведение стиля кода в соотвествие с PEP8 автоматически
    ```
    $ make format
    ```
1. проверка стиля кода на соотвествие с PEP8
    ```
    $ make check
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
Мы предоставляем примеры разного уровня сложности:
* [минимальные] минималистичные примеры, представляющие наш API
* [базовые] применение eXNN для простых задач, таких как классификация MNIST
* [сценарии использования] демонстрация использования eXplain-NN для решения различных проблем, возникающих в промышленных задачах.

### Минимальные
Этот колаб содержит минималистическую демонстрацию нашего API на фиктивных объектах:

[![minimal](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lOiB50LppDiiRHTv184JMuQ2IvZ4I4rp?usp=sharing)

### Базовые
Вот колабы, демонстрирующие, как работать с разными модулями нашего API на простых задачах:

| Colab Link | Module |
| ------------- | ------------- |
| [![bayes](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ayd0IronxUIfnbAmWQLHiILG2qtBBpF4?usp=sharing)| bayes |
| [![topology](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1T5ENfNaCIRI61LM2ZhtU8lfmvRmlfiEo?usp=sharing)| topology |
| [![visualization](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LJVdWTv-wcASSMX4is_E15TR7XJsT7W3?usp=sharing)| visualization |

### Сценарии использования
В этом блоке представлены примеры использования eXplain-NN для решения различных вариантов использования в промышленных задачах. Для демонстрационных целей используются 4 задачи:
* [спутник] классификация ландшафтов по спутниковым снимкам.
* [электроника] классификация электронных компонентов и устройств
* [ЭКГ] диагностика ЭКГ
* [полупроводники] детекция дефектов при тестировании полупроводниковых пластин

| Colab Link | Task | Use Case |
| ------------- | ------------- | ------------- |
| [![CNN_viz](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12ZJigH-0geGTefNXnCM5dQ71d4tqlf6L?usp=sharing)| спутник | Визуализация изменения многообразия данных от слоя к слою |
| [![adv](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n50WUu2ZKZ6nrT9DuFD3q87m3yZvxkwm?usp=sharing) | спутник | Детекция adversarial данных |
| [![generalize](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mG-VrP7J7OoCvIQDl7n5YWEIdyfFg_0I?usp=sharing) | электроника | Оценка обобщающей способности нейронной сети |
| [![RNN_viz](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aAtqxQLcOsSJJumfsmS9HGLgHrOFHlfk?usp=sharing) | ЭКГ | Визуализация изменения многообразия данных от слоя к слою |
| [![wafer](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nGpnO1wC7jp3fW4rjVfucvKyz_FA8hYp?usp=sharing) | полупроводники | Детекция дефектов при тестировании полупроводниковых пластин |

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
