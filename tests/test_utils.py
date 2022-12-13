from collections import OrderedDict
import torch
import torch.nn as nn


def _form_message_header(message_header=None):
    return 'Value mismatch' if message_header is None else message_header

def compare_values(expected, got, message_header=None):
    assert expected == got, f"{_form_message_header(message_header)}: expected {expected}, got {got}"

def create_testing_data():
    N = 20
    dim = 256
    data = torch.randn((N, dim))
    return N, dim, data

def create_testing_model():
    return nn.Sequential(
        OrderedDict([
            ('first_layer', nn.Linear(256, 128)), 
            ('second_layer', nn.Linear(128, 64)), 
            ('third_layer', nn.Linear(64, 10))
        ])
    )