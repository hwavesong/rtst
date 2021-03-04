# -*- coding: utf-8 -*-
import numpy as np
from absl import logging
from tqdm import tqdm

import model as cm
from utils import calculate_cosine_score


def grid_search_threshold(short_AB, short_AC, raw_A):
    A_losses = list()
    for threshold in np.linspace(-0.99, 0.99, 199):
        bin_AB = binarize_vectors(short_AB, threshold)
        bin_AC = binarize_vectors(short_AC, threshold)

        bin_A = calculate_cosine_score(bin_AB, bin_AC)

        A_loss = np.mean(np.square(bin_A - raw_A))
        A_losses.append(A_loss)

    return np.array(A_losses)


def binarize_vectors(real_vectors, threshold):
    bin_vectors = np.ones_like(real_vectors)
    bin_vectors[real_vectors < threshold] = -1.

    return bin_vectors


def solve_truncate_threshold(datasets, cfg):
    logging.info('solving threshold')

    model = cm.CompressModel()
    model.load_weights(cfg['compress_model_path'])

    losses_via_linespace = list()
    for ix, (raw_AB, raw_AC, raw_A) in enumerate(datasets):
        raw_AB = np.array(raw_AB, np.float32)
        raw_AC = np.array(raw_AC, np.float32)
        raw_A = np.array(raw_A, np.float32)

        short_AB = model(raw_AB, training=False).numpy()
        short_AC = model(raw_AC, training=False).numpy()

        losses_via_linespace.append(grid_search_threshold(short_AB, short_AC, raw_A))

    losses_via_linespace = np.array(losses_via_linespace)
    mean_loss_via_linespace = np.mean(losses_via_linespace, axis=0)

    threshold = np.linspace(-0.99, 0.99, 199)[np.argmin(mean_loss_via_linespace)]

    logging.info('threshold is {:.2f}'.format(threshold))

    return threshold


def produce_short_vectors(w_v_dct, cfg):
    model_path = cfg['compress_model_path']
    vec_path = cfg['vec_path']
    vec_size = cfg['vec_size']

    logging.info('generating vector to ' + vec_path)

    model = cm.CompressModel()
    model.load_weights(model_path)

    with open(vec_path, 'w', encoding='utf8') as file:
        file.write(str(len(w_v_dct)) + ' ' + vec_size + '\n')

        words, vectors = list(), list()
        total_len = len(w_v_dct)
        for ix, (k, v) in enumerate(w_v_dct.items()):
            print('\r {: >7,d} / {:d}'.format(ix, total_len), end='')

            if len(words) >= 1024:
                vectors = np.array(vectors)
                np.clip(vectors, -1., 1., vectors)
                short_vectors = np.around(model(vectors, training=False).numpy(), decimals=5)

                for word, vector in zip(words, short_vectors):
                    file.write(word + ' ' + ' '.join([str(v) for v in vector]) + '\n')

                words, vectors = list(), list()

            words.append(k)
            vectors.append(v)

        # process remaining data
        vectors = np.array(vectors)
        np.clip(vectors, -1., 1., vectors)
        short_vectors = np.around(model(vectors, training=False).numpy(), decimals=5)
        for word, vector in zip(words, short_vectors):
            file.write(word + ' ' + ' '.join([str(v) for v in vector]) + '\n')

    print()


def produce_binary_vectors(threshold, cfg):
    logging.info('binarized vector to ' + cfg['bin_path'])

    with open(cfg['bin_path'], 'w', encoding='utf8') as bin_file:
        with open(cfg['vec_path'], 'r', encoding='utf8') as real_file:
            bin_file.write(next(real_file))

            for line in tqdm(real_file):
                splited_line = line.rstrip().split(' ')
                real_vector = np.asarray(splited_line[1:], np.float32)

                binary_vector = binarize_vectors(real_vector, threshold)

                bin_file.write(splited_line[0] + ' ' + ' '.join([str(x) for x in binary_vector]) + '\n')
