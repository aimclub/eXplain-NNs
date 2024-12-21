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