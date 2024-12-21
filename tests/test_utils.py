from collections import OrderedDict

import torch
import torch.nn as nn


def _form_message_header(message_header=None):
    return "Value mismatch" if message_header is None else message_header


def compare_values(expected, got, message_header=None):
    assert (
        expected == got
    ), f"{_form_message_header(message_header)}: expected {expected}, got {got}"


def create_testing_data(architecture='fcn'):
    architecture = architecture.lower()
    if architecture == 'fcn':
        return torch.randn((20, 256))
    elif architecture == 'cnn':
        return torch.randn((20, 3, 32, 32))
    else:
        raise Exception(f'Unsupported architecture type: {architecture}')


def create_testing_model(architecture='fcn', num_classes=10):
    architecture = architecture.lower()
    if architecture == 'fcn':
        return nn.Sequential(
            OrderedDict(
                [
                    ("first_layer", nn.Linear(256, 128)),
                    ("second_layer", nn.Linear(128, 64)),
                    ("third_layer", nn.Linear(64, num_classes)),
                ],
            ),
        )
    elif architecture == 'cnn':
        return nn.Sequential(
            OrderedDict(
                [
                    ("first_layer", nn.Conv2d(in_channels=3, out_channels=10, kernel_size=7)),
                    ("second_layer", nn.Conv2d(in_channels=10, out_channels=20, kernel_size=7)),
                    ("avgpool", nn.AdaptiveAvgPool2d(1)),
                    ("flatten", nn.Flatten()),
                    ("fc", nn.Linear(20, num_classes)),
                ],
            ),
        )
    elif architecture == 'rnn':
        return nn.Sequential(
        OrderedDict(
            [
                ('first_layer', nn.LSTM(256, 128, 1, batch_first=True)),
                ('extract', ExtractTensor()),
                ('second_layer', nn.Linear(128, 64)),
                ('third_layer', nn.Linear(64, num_classes)),
            ],
        ),
    )
    else:
        raise Exception(f'Unsupported architecture type: {architecture}')


class ExtractTensor(nn.Module):
    def forward(self, x):
        tensor, _ = x
        x = x.to(torch.float32)
        return tensor[:, :]