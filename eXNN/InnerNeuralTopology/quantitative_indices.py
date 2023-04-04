import heapq
import numpy as np

def compute_length(diag):
    return [d[1] - d[0] for d in diag[0]]

def compute_longest_interval(lengths):
    return max(lengths)

def compute_length_mean(lengths):
    return np.mean(lengths)

def compute_length_median(lengths):
    return np.median(lengths)

def compute_length_stdev(lengths):
    return np.std(lengths)

def compute_length_sum(lengths):
    return np.sum(lengths)
    
# Proportion between the longest intervals: 2/1 ratio, 3/1 ratio

def compute_two_to_one_ratio(lengths):
    return heapq.nlargest(2, lengths)[1] / lengths[0]

def compute_three_to_one_ratio(lengths):
    return heapq.nlargest(3, lengths)[2] / lengths[0]
    
# Compute the persistent entropy and normed persistent entropy
def compute_entropy(lengths, length_sum):
    return (-1) * np.sum(np.divide(lengths, length_sum)*np.log(np.divide(lengths, length_sum)))

def compute_normed_entropy(entropy, length_sum):
    return entropy / np.log(length_sum)
    
# Compute births
def compute_births(diag):
    return np.array([x[0] for x in diag[0]])

# Comput deaths
def compute_deaths(diag):
    return np.array([x[1] for x in diag[0]])

def compute_birth(diag, dim):
    temp = np.array([x[0] for x in diag[0] if x[2]==dim])   
    return temp[0]

def compute_death(diag, dim):
    temp = np.array([x[1] for x in diag[0] if x[2]==dim])
    return temp[-1]

# Compute SNR
def compute_snr(births, deaths):
    signal = np.mean(deaths - births)
    noise = np.std(births)
    snr = signal / noise
    return snr
    
# Compute the birth-death pair indices: Birth mean, birth stdev, death mean, death stdev
def compute_births_mean(births):
    return np.mean(births)

def compute_births_stdev(births):
    return np.std(births)

def compute_deaths_mean(deaths):
    return np.mean(deaths)

def compute_deaths_stdev(deaths):
    return np.std(deaths)
