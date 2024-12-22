import matplotlib

import eXNN.topology as topology_api
import tests.utils.test_utils as utils


def test_data_barcode():
    """
    Test generating a barcode from data.

    Verifies:
        - Output is a dictionary.
    """
    n, dim, data = utils.create_testing_data()
    res = topology_api.get_data_barcode(data, "standard", "3")
    utils.compare_values(dict, type(res), "Wrong result type")


def test_nn_barcodes():
    """
    Test generating barcodes for a neural network.

    Verifies:
        - Output is a dictionary.
        - Dictionary keys match the specified layers.
        - Each dictionary value is a dictionary.
    """
    n, dim, data = utils.create_testing_data()
    model = utils.create_testing_model()
    layers = ["second_layer", "third_layer"]

    res = topology_api.get_nn_barcodes(model, data, layers, "standard", "3")

    utils.compare_values(dict, type(res), "Wrong result type")
    utils.compare_values(2, len(res), "Wrong dictionary length")
    utils.compare_values(set(layers), set(res.keys()), "Wrong dictionary keys")

    for layer, barcode in res.items():
        utils.compare_values(
            dict,
            type(barcode),
            f"Wrong result type for key {layer}",
        )


def test_barcode_plot():
    """
    Test generating a barcode plot.

    Verifies:
        - Output is a matplotlib.figure.Figure.
    """
    n, dim, data = utils.create_testing_data()
    barcode = topology_api.get_data_barcode(data, "standard", "3")
    plot = topology_api.plot_barcode(barcode)
    utils.compare_values(matplotlib.figure.Figure, type(plot), "Wrong result type")

