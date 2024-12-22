import eXNN.topology as topology_api
import tests.utils.test_utils as utils


def test_barcode_evaluate_all_metrics():
    """
    Test evaluating all metrics for a barcode.

    Verifies:
        - Output is a dictionary.
        - Dictionary keys match the expected metric names.
        - Each metric value is a float.
    """
    n, dim, data = utils.create_testing_data()
    barcode = topology_api.get_data_barcode(data, "standard", "3")
    result = topology_api.evaluate_barcode(barcode)
    utils.compare_values(dict, type(result), "Wrong result type")
    all_metric_names = [
        "h",
        "max_length",
        "mean_birth",
        "mean_death",
        "mean_length",
        "median_length",
        "normh",
        "ratio_2_1",
        "ratio_3_1",
        "snr",
        "stdev_birth",
        "stdev_death",
        "stdev_length",
        "sum_length",
    ]
    utils.compare_values(all_metric_names, sorted(result.keys()))
    for name, value in result.items():
        utils.compare_values(float, type(value), f"Wrong result type for metric {name}")


def test_barcode_evaluate_one_metric():
    """
    Test evaluating a single metric for a barcode.

    Verifies:
        - Output is a float.
    """
    n, dim, data = utils.create_testing_data()
    barcode = topology_api.get_data_barcode(data, "standard", "3")
    result = topology_api.evaluate_barcode(barcode, metric_name="mean_length")
    utils.compare_values(float, type(result), "Wrong result type")

