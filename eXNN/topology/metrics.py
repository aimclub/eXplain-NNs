import heapq
import numpy as np


def _get_available_metrics():
    """
    Returns a dictionary of available metric names and their corresponding functions.

    Returns:
        dict: Mapping of metric names to functions.
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


def compute_metric(barcode, metric_name=None):
    """
    Compute the specified metric or all available metrics for a given barcode.

    Args:
        barcode (dict): The barcode data containing persistent homology intervals.
        metric_name (str, optional): The name of the metric to compute. Defaults to None.

    Returns:
        dict or float: A dictionary of all metrics if metric_name is None, otherwise the value of the specified metric.
    """
    metrics = _get_available_metrics()
    if metric_name is None:
        return {name: fn(barcode) for name, fn in metrics.items()}
    return metrics[metric_name](barcode)


def _get_lengths(barcode):
    """
    Compute lengths of intervals in the H0 component of the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        list: A list of interval lengths.
    """
    diag = barcode["H0"]
    return [d[1] - d[0] for d in diag]


def _compute_longest_interval_metric(barcode):
    """
    Compute the length of the longest interval in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The length of the longest interval.
    """
    lengths = _get_lengths(barcode)
    return np.max(lengths).item()


def _compute_length_mean_metric(barcode):
    """
    Compute the mean length of intervals in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The mean length of intervals.
    """
    lengths = _get_lengths(barcode)
    return np.mean(lengths).item()


def _compute_length_median_metric(barcode):
    """
    Compute the median length of intervals in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The median length of intervals.
    """
    lengths = _get_lengths(barcode)
    return np.median(lengths).item()


def _compute_length_stdev_metric(barcode):
    """
    Compute the standard deviation of interval lengths in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The standard deviation of interval lengths.
    """
    lengths = _get_lengths(barcode)
    return np.std(lengths).item()


def _compute_length_sum_metric(barcode):
    """
    Compute the sum of interval lengths in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The sum of interval lengths.
    """
    lengths = _get_lengths(barcode)
    return np.sum(lengths).item()


def _compute_two_to_one_ratio_metric(barcode):
    """
    Compute the ratio of the second longest to the longest interval in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The 2-to-1 ratio of interval lengths.
    """
    lengths = _get_lengths(barcode)
    value = heapq.nlargest(2, lengths)[1] / lengths[0]
    return value


def _compute_three_to_one_ratio_metric(barcode):
    """
    Compute the ratio of the third longest to the longest interval in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The 3-to-1 ratio of interval lengths.
    """
    lengths = _get_lengths(barcode)
    value = heapq.nlargest(3, lengths)[2] / lengths[0]
    return value


def _get_entropy(values, normalize):
    """
    Compute the entropy of a set of values.

    Args:
        values (list): The values for which to compute entropy.
        normalize (bool): Whether to normalize the entropy.

    Returns:
        float: The computed entropy.
    """
    values_sum = np.sum(values)
    entropy = -np.sum(np.divide(values, values_sum) * np.log(np.divide(values, values_sum)))
    if normalize:
        entropy /= np.log(values_sum)
    return entropy


def _compute_entropy_metric(barcode):
    """
    Compute the persistent entropy of the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The persistent entropy.
    """
    return _get_entropy(_get_lengths(barcode), normalize=False).item()


def _compute_normed_entropy_metric(barcode):
    """
    Compute the normalized persistent entropy of the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The normalized persistent entropy.
    """
    return _get_entropy(_get_lengths(barcode), normalize=True).item()


def _get_births(barcode):
    """
    Extract the birth times from the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        np.ndarray: The birth times.
    """
    diag = barcode["H0"]
    return np.array([x[0] for x in diag])


def _get_deaths(barcode):
    """
    Extract the death times from the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        np.ndarray: The death times.
    """
    diag = barcode["H0"]
    return np.array([x[1] for x in diag])


def _compute_snr_metric(barcode):
    """
    Compute the signal-to-noise ratio (SNR) of the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The SNR value.
    """
    births = _get_births(barcode)
    deaths = _get_deaths(barcode)
    signal = np.mean(deaths - births)
    noise = np.std(births)
    return (signal / noise).item()


def _compute_births_mean_metric(barcode):
    """
    Compute the mean of birth times in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The mean birth time.
    """
    return np.mean(_get_births(barcode)).item()


def _compute_births_stdev_metric(barcode):
    """
    Compute the standard deviation of birth times in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The standard deviation of birth times.
    """
    return np.std(_get_births(barcode)).item()


def _compute_deaths_mean_metric(barcode):
    """
    Compute the mean of death times in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The mean death time.
    """
    return np.mean(_get_deaths(barcode)).item()


def _compute_deaths_stdev_metric(barcode):
    """
    Compute the standard deviation of death times in the barcode.

    Args:
        barcode (dict): The barcode data.

    Returns:
        float: The standard deviation of death times.
    """
    return np.std(_get_deaths(barcode)).item()
