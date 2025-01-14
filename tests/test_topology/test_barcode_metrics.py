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


def test_barcode_evaluate_all_metrics_and_individual():
    """
    Test evaluating all metrics for a barcode and individual metrics evaluation.

    Verifies:
        - Output for evaluating all metrics is a dictionary.
        - Dictionary keys match the expected metric names.
        - Each metric value is a float.
        - Individual metrics can be correctly evaluated.
    """
    # Test for all metrics
    n, dim, data = utils.create_testing_data()
    barcode = topology_api.get_data_barcode(data, "standard", "3")
    result = topology_api.evaluate_barcode(barcode)

    # Check that the result is a dictionary
    utils.compare_values(dict, type(result), "Wrong result type")

    # List of expected metric names
    all_metric_names = [
        "h", "max_length", "mean_birth", "mean_death", "mean_length",
        "median_length", "normh", "ratio_2_1", "ratio_3_1", "snr",
        "stdev_birth", "stdev_death", "stdev_length", "sum_length"
    ]

    # Check that the dictionary keys match the expected metric names
    utils.compare_values(
        all_metric_names,
        sorted(result.keys()),
        "Wrong dictionary keys"
    )

    # Ensure all metric values are floats
    for name, value in result.items():
        utils.compare_values(
            float, type(value),
            f"Wrong result type for metric {name}"
        )

    # Test for evaluating individual metrics
    for metric_name in all_metric_names:
        individual_result = topology_api.evaluate_barcode(barcode, metric_name=metric_name)
        utils.compare_values(
            float,
            type(individual_result),
            f"Wrong result type for individual metric {metric_name}"
        )
