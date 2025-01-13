import heapq
from typing import Dict

import numpy as np


def _get_available_metrics():
    """
    Returns a dictionary mapping metric names to their respective computation functions.

    Returns:
        Dict[str, callable]: A dictionary of metric computation functions.
    """
    return {
        # Absolute length-based metrics
        "max_length": _compute_longest_interval_metric,
        "mean_length": _compute_length_mean_metric,
        "median_length": _compute_length_median_metric,
        "stdev_length": _compute_length_stdev_metric,
        "sum_length": _compute_length_sum_metric,
        # Relative length-based metrics
        "ratio_2_1": _compute_two_to_one_ratio_metric,
        "ratio_3_1": _compute_three_to_one_ratio_metric,
        # Entropy-based metrics
        "h": _compute_entropy_metric,
        "normh": _compute_normed_entropy_metric,
        # Signal-to-noise ratio
        "snr": _compute_snr_metric,
        # Birth-death based metrics
        "mean_birth": _compute_births_mean_metric,
        "stdev_birth": _compute_births_stdev_metric,
        "mean_death": _compute_deaths_mean_metric,
        "stdev_death": _compute_deaths_stdev_metric,
    }


def compute_metric(barcode: Dict[str, np.ndarray], metric_name: str = None):
    """
    Computes specified or all metrics for a given barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to compute metrics for.
        metric_name (str, optional): The specific metric name to compute.
        If None, all metrics are computed.

    Returns:
        float or Dict[str, float]: The computed metric(s).
    """
    metrics = _get_available_metrics()
    if metric_name is None:
        return {name: fn(barcode) for name, fn in metrics.items()}
    return metrics[metric_name](barcode)


def _get_lengths(barcode: Dict[str, np.ndarray]):
    """
    Extracts lengths of intervals from a barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        List[float]: A list of interval lengths.
    """
    diag = barcode["H0"]
    return [d[1] - d[0] for d in diag]


def _compute_longest_interval_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the maximum interval length in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The maximum interval length.
    """
    lengths = _get_lengths(barcode)
    return float(np.max(lengths))


def _compute_length_mean_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the mean interval length in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The mean interval length.
    """
    lengths = _get_lengths(barcode)
    return float(np.mean(lengths))


def _compute_length_median_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the median interval length in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The median interval length.
    """
    lengths = _get_lengths(barcode)
    return float(np.median(lengths))


def _compute_length_stdev_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the standard deviation of interval lengths in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The standard deviation of interval lengths.
    """
    lengths = _get_lengths(barcode)
    return float(np.std(lengths))


def _compute_length_sum_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the sum of all interval lengths in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The sum of all interval lengths.
    """
    lengths = _get_lengths(barcode)
    return float(np.sum(lengths))


def _compute_two_to_one_ratio_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the ratio of the second largest to the largest interval length.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The ratio of the second largest to the largest interval length.
    """
    lengths = _get_lengths(barcode)
    value = heapq.nlargest(2, lengths)[1] / lengths[0]
    return float(value)


def _compute_three_to_one_ratio_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the ratio of the third largest to the largest interval length.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The ratio of the third largest to the largest interval length.
    """
    lengths = _get_lengths(barcode)
    value = heapq.nlargest(3, lengths)[2] / lengths[0]
    return float(value)


def _get_entropy(values: np.ndarray, normalize: bool) -> float:
    """
    Computes the entropy of a given distribution.

    Args:
        values (np.ndarray): The values to compute entropy for.
        normalize (bool): Whether to normalize the entropy.

    Returns:
        float: The computed entropy.
    """
    values_sum = np.sum(values)
    entropy = (-1) * np.sum(np.divide(values, values_sum) * np.log(np.divide(values, values_sum)))
    if normalize:
        entropy = entropy / np.log(values_sum)
    return float(entropy)


def _compute_entropy_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the persistent entropy of intervals in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The persistent entropy.
    """
    return _get_entropy(_get_lengths(barcode), normalize=False)


def _compute_normed_entropy_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the normalized persistent entropy of intervals in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The normalized persistent entropy.
    """
    return _get_entropy(_get_lengths(barcode), normalize=True)


def _get_births(barcode: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extracts the birth times from the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        np.ndarray: An array of birth times.
    """
    diag = barcode["H0"]
    return np.array([x[0] for x in diag])


def _get_deaths(barcode: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Extracts the death times from the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        np.ndarray: An array of death times.
    """
    diag = barcode["H0"]
    return np.array([x[1] for x in diag])


def _compute_snr_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the signal-to-noise ratio (SNR) for the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The computed SNR.
    """
    births = _get_births(barcode)
    deaths = _get_deaths(barcode)
    signal = np.mean(deaths - births)
    noise = np.std(births)
    snr = signal / noise
    return float(snr)


def _compute_births_mean_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the mean of birth times in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The mean of birth times.
    """
    return float(np.mean(_get_births(barcode)))


def _compute_births_stdev_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the standard deviation of birth times in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The standard deviation of birth times.
    """
    return float(np.std(_get_births(barcode)))


def _compute_deaths_mean_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the mean of death times in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The mean of death times.
    """
    return float(np.mean(_get_deaths(barcode)))


def _compute_deaths_stdev_metric(barcode: Dict[str, np.ndarray]) -> float:
    """
    Computes the standard deviation of death times in the barcode.

    Args:
        barcode (Dict[str, np.ndarray]): The barcode to process.

    Returns:
        float: The standard deviation of death times.
    """
    return float(np.std(_get_deaths(barcode)))
