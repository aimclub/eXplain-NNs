import heapq
import numpy as np

def _get_available_metrics():
    return {
        # absolute length based metrics
        'max_length': _compute_longest_interval_metric,
        'mean_length': _compute_length_mean_metric,
        'median_length': _compute_length_median_metric,
        'stdev_length': _compute_length_stdev_metric,
        'sum_length': _compute_length_sum_metric,
        # relative length based metrics
        'ratio_2_1': _compute_two_to_one_ratio_metric,
        'ratio_3_1': _compute_three_to_one_ratio_metric,
        # entopy based metrics
        'h': _compute_entropy_metric,
        'normh': _compute_normed_entropy_metric,
        # signal to noise ration
        'snr': _compute_snr_metric,
        # birth-death based metrics
        'mean_birth': _compute_births_mean_metric,
        'stdev_birth': _compute_births_stdev_metric,
        'mean_death': _compute_deaths_mean_metric,
        'stdev_death': _compute_deaths_stdev_metric,
    }



def compute_metric(barcode, metric_name=None):
    metrics = _get_available_metrics()
    if metric_name is None:
        return {name: fn(barcode) for (name, fn) in metrics.items()}
    else:
        return metrics[metric_name](barcode)


def _get_lengths(barcode):
    diag = barcode['H0']
    return [d[1] - d[0] for d in diag]


def _compute_longest_interval_metric(barcode):
    lengths = _get_lengths(barcode)
    return max(lengths)


def _compute_length_mean_metric(barcode):
    lengths = _get_lengths(barcode)
    return np.mean(lengths)


def _compute_length_median_metric(barcode):
    lengths = _get_lengths(barcode)
    return np.median(lengths)


def _compute_length_stdev_metric(barcode):
    lengths = _get_lengths(barcode)
    return np.std(lengths)


def _compute_length_sum_metric(barcode):
    lengths = _get_lengths(barcode)
    return np.sum(lengths)


# Proportion between the longest intervals: 2/1 ratio, 3/1 ratio
def _compute_two_to_one_ratio_metric(barcode):
    lengths = _get_lengths(barcode)
    return heapq.nlargest(2, lengths)[1] / lengths[0]


def _compute_three_to_one_ratio_metric(barcode):
    lengths = _get_lengths(barcode)
    return heapq.nlargest(3, lengths)[2] / lengths[0]


# Compute the persistent entropy and normed persistent entropy
def _get_entropy(values, normalize:bool):
    values_sum = np.sum(values)
    entropy = (-1) * np.sum(np.divide(values, values_sum) * np.log(np.divide(values, values_sum)))
    if normalize:
        entropy = entropy / np.log(values_sum)
    return entropy

def _compute_entropy_metric(barcode):
    return _get_entropy(_get_lengths(barcode), normalize=False)


def _compute_normed_entropy_metric(barcode):
    return _get_entropy(_get_lengths(barcode), normalize=True)


# Compute births
def _get_births(barcode):
    diag = barcode['H0']
    return np.array([x[0] for x in diag])


# Comput deaths
def _get_deaths(barcode):
    diag = barcode['H0']
    return np.array([x[1] for x in diag])


# def _get_birth(barcode, dim):
#     diag = barcode['H0']
#     temp = np.array([x[0] for x in diag if x[2] == dim])
#     return temp[0]


# def _get_death(barcode, dim):
#     diag = barcode['H0']
#     temp = np.array([x[1] for x in diag if x[2] == dim])
#     return temp[-1]


# Compute SNR
def _compute_snr_metric(barcode):
    births = _get_births(barcode)
    deaths = _get_deaths(barcode)
    signal = np.mean(deaths - births)
    noise = np.std(births)
    snr = signal / noise
    return snr


# Compute the birth-death pair indices: Birth mean, birth stdev, death mean, death stdev
def _compute_births_mean_metric(barcode):
    return np.mean(_get_births(barcode))


def _compute_births_stdev_metric(barcode):
    return np.std(_get_births(barcode))


def _compute_deaths_mean_metric(barcode):
    return np.mean(_get_deaths(barcode))


def _compute_deaths_stdev_metric(barcode):
    return np.std(_get_deaths(barcode))
