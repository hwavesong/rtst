# -*- coding: utf-8 -*-
import configparser
import logging
import os

import numpy as np
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def calculate_cosine_score(vectors1, vectors2):
    sum1 = np.sum(vectors1 * vectors1, axis=-1)
    sum2 = np.sum(vectors2 * vectors2, axis=-1)

    np.clip(sum1, 1.e-8, None, sum1)
    np.clip(sum2, 1.e-8, None, sum2)

    return np.sum(vectors1 * vectors2, axis=-1) / np.sqrt(sum1 * sum2)


def read_config(filename, section):
    cp = configparser.ConfigParser()
    cp.read(filename)

    return cp[section]


def load_wv_dct(path):
    logging.info('loading word vector from ' + path)

    w_v_dct = dict()
    with open(path, 'r', encoding='utf8') as file:
        n, d = next(file).split()  # skip the first line
        d = int(d)

        for line in tqdm(file):
            splited_line = line.rstrip().split(' ')

            if len(splited_line) < d + 1:
                logging.info('error word ' + splited_line[0])
                continue

            w_v_dct[splited_line[0]] = np.asarray(splited_line[1:], dtype=np.float32)

    logging.info('loaded : ' + str(len(w_v_dct)) + ' words')

    return w_v_dct


def load_bucket(path):
    logging.info('loading bucket data from ' + path)

    bucket_words = list()
    last_is_blank_line = False
    with open(path, 'r', encoding='utf8') as file:
        for line in tqdm(file):
            strip_line = line.strip()

            if last_is_blank_line:
                if len(strip_line) < 1:
                    bucket_words.append(list())
                    bucket_words[-1].append(list())
                else:
                    bucket_words[-1].append(list())
                    bucket_words[-1][-1].append(strip_line)
                last_is_blank_line = False
                continue
            if len(strip_line) < 1:
                last_is_blank_line = True
                continue

            bucket_words[-1][-1].append(strip_line)

    logging.info('loaded : ' + str(len(bucket_words)) + ' buckets')

    return bucket_words


def flatten_bucket(bucket_words):
    logging.info('flatting bucket')

    flatten_list = list()
    for one_bucket in bucket_words:
        flatten_list.extend(one_bucket)

    logging.info('flatted :{} buckets'.format(len(flatten_list)))

    return flatten_list


def load_filter_set(dir_path):
    logging.info('loading filter table from: ' + dir_path)

    filter_set = set()

    for dataset_name in os.listdir(dir_path):
        dataset_path = os.path.join(dir_path, dataset_name)

        with open(dataset_path, 'r', encoding='utf8') as file:
            for line in file:
                splited_line = line.strip().split('\t')

                filter_set.add((splited_line[0], splited_line[1]))

    logging.info('loaded: {} filter cases'.format(len(filter_set)))

    return filter_set


def load_bucket_pairs(path):
    logging.info('loading bucket pairs from ' + path)

    bucket_pairs = list()
    # sim_bucket_pairs
    with open(path, 'r', encoding='utf8') as file:
        for line in tqdm(file):
            splited_line = line.strip().split('\t')
            bucket_pairs.append(tuple(eval(x) for x in splited_line))

    logging.info('loaded : {} bucket paris'.format(len(bucket_pairs)))

    return bucket_pairs
