import numpy as np
from random import choice

def data_check(data):
    nItems = data.shape[1]
    nSubs = data.shape[0]

    for row in data:
        if sum(row) == nItems or sum(row) == 0:
            print("One or more rows are completely 0 or 1! Remove said rows to fix.")
            break
    for column in data.transpose():
        if sum(column) == nSubs or sum(column) == 0:
            print("One or more columns are completely 0 or 1! Remove said columns to fix.")
            break

def generate_sample_data(items, subjects, item_stats, subject_stats, item_params = None, subject_params = None):
    def logistic(z):
        return 1/(1+np.exp(-z))

    t1, t2, tn = type(item_params), type(subject_params), type(None)

    if t1 == tn and t2 == tn:
        mu_item, std_item = item_stats
        mu_subject, std_subject = subject_stats
        item_params = np.random.normal(loc = mu_item, scale = std_item, size = items)
        subject_params = np.random.normal(loc = mu_subject, scale = std_subject, size = subjects)
    elif t1 == tn and t2 != tn:
        mu_item, std_item = item_stats
        item_params = np.random.normal(loc = mu_item, scale = std_item, size = items)
    elif t1 != tn and t2 == tn:
        mu_subject, std_subject = subject_stats
        subject_params = np.random.normal(loc = mu_subject, scale = std_subject, size = subjects)

    sample_data = np.zeros((subjects, items))

    for i, subject in enumerate(subject_params):
        for j, item in enumerate(item_params):
            p = logistic(subject - item)
            sample_data[i][j] = np.random.choice([0, 1], p = [1-p, p])
    
    if items > 1:
        for i, row in enumerate(sample_data):
            if sum(row) == 0 or sum(row) == items:
                replacement = choice(sample_data)
                while sum(replacement) == 0 or sum(replacement) == items:
                    replacement = choice(sample_data)
                sample_data[i] = replacement

    return sample_data
