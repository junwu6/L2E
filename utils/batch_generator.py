import numpy as np


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(raw_data, batch_size, shuffle=True, ind=None):
    if ind is None:
        data = [raw_data['X'], raw_data['Y']]
    else:
        data = [raw_data['X'][ind], raw_data['Y'][ind]]
    if shuffle:
        data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield data[0][start:end], data[1][start:end]
