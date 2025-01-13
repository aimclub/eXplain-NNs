from collections import OrderedDict

import torch
import torch.nn as nn


def _form_message_header(message_header=None):
    """
    Forms the header for assertion messages.

    Args:
        message_header (str, optional): Custom message header. Defaults to None.

    Returns:
        str: The custom header if provided, otherwise "Value mismatch".
    """
    return "Value mismatch" if message_header is None else message_header


def compare_values(expected, got, message_header=None):
    """
    Compares two values and raises an assertion error if they do not match.

    Args:
        expected: The expected value.
        got: The value to compare against the expected value.
        message_header (str, optional): Custom header for the assertion message. Defaults to None.

    Raises:
        AssertionError: If the values do not match, an error with the message header is raised.
    """
    assert (expected == got), \
        f"{_form_message_header(message_header)}: expected {expected}, got {got}"


def create_testing_data():
    """
    Creates synthetic testing data.

    Returns:
        tuple: A tuple containing:
            - N (int): Number of samples.
            - dim (int): Dimensionality of each sample.
            - data (torch.Tensor): Randomly generated data tensor of shape (N, dim).
    """
    n = 20
    dim = 256
    data = torch.randn((n, dim))
    return n, dim, data


def create_testing_model(num_classes=10):
    """
    Creates a simple feedforward neural network for testing.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.

    Returns:
        nn.Sequential: A sequential model with three layers:
            - Linear layer (input_dim=256, output_dim=128).
            - Linear layer (input_dim=128, output_dim=64).
            - Linear layer (input_dim=64, output_dim=num_classes).
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("first_layer", nn.Linear(256, 128)),
                ("second_layer", nn.Linear(128, 64)),
                ("third_layer", nn.Linear(64, num_classes)),
            ],
        ),
    )


class ExtractTensor(nn.Module):
    """
    A custom PyTorch module to extract and process tensors.

    This module extracts the first tensor from a tuple, converts it to
    float32, and returns its values.
    """

    @staticmethod
    def forward(x):
        """
        Forward pass for tensor extraction and processing.

        Args:
            x (tuple): Input tuple where the first element is the tensor to be processed.

        Returns:
            torch.Tensor: Processed tensor (converted to float32).
        """
        tensor, _ = x
        tensor = tensor.to(torch.float32)
        return tensor[:, :]


def create_testing_model_lstm(num_classes=10):
    """
    Creates a recurrent neural network with LSTM layers for testing.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.

    Returns:
        nn.Sequential: A sequential model with the following layers:
            - LSTM layer (input_dim=256, hidden_dim=128, num_layers=1).
            - ExtractTensor layer to process LSTM output.
            - Linear layer (input_dim=128, output_dim=64).
            - Linear layer (input_dim=64, output_dim=num_classes).
    """
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
