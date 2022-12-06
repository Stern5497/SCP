import datetime
import json as js
import os
import re
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from datasets import Dataset, concatenate_datasets, load_dataset, load_metric
from scipy.special import expit


def get_dataset(experimental_samples, train_arg, split):
    if train_arg:
        if experimental_samples:
            return load_dataset("Stern5497/SCP", 'bge_scp', split=split + '[:5%]',
                                cache_dir="datasets_caching")
        elif not experimental_samples:
            return load_dataset("Stern5497/SCP", 'bge_scp', split=split,
                                cache_dir="datasets_caching")
    return datasets.Dataset.from_pandas(pd.DataFrame())


def remove_unused_features(dataset, feature_col, label):
    columns_to_remove = ['origin_chamber', 'legal_area', 'considerations', 'facts', 'origin_region', 'court', 'year',
                         'chamber', 'origin_court', 'origin_canton', 'canton', 'decision_id', 'region', 'language',
                         'bge_label', 'citation_label']
    columns_to_keep = [feature_col, label, 'language']
    columns_to_remove = [ele for ele in columns_to_remove if ele not in columns_to_keep]
    for column in columns_to_remove:
        if column in dataset.column_names:
            dataset = dataset.remove_columns(column)
    return dataset


# rename to input, label and language
def rename_features(dataset, feature_col, label_name):
    dataset = dataset.rename_column(feature_col, "input")
    dataset = dataset.rename_column(label_name, 'label')
    return dataset


# filter out cases with too short input or too long inout
def filter_by_length(example):
    """
    Removes examples that are too short
    :param example:    the example to check
    :return:
    """
    # TODO check if this can be removed, because we want to use all cases
    if 10 < len(str(example["input"])):
        return True
    return False


def get_labels(critical_only):
    if critical_only:
        return ['critical-1', 'critical-2', 'critical-3', 'critical-4']
    else:
        return ['critical-1', 'critical-2', 'critical-3', 'critical-4', 'non-critical']


def add_oversampling_to_multiclass_dataset(train_dataset, id2label, data_args):
    """
    get the biggest class and oversample all other classes to have the same size
    """
    # get the biggest class
    max_class = ['', 0]
    label_datasets = dict()
    for label_id in id2label.keys():
        label_datasets[label_id] = train_dataset.filter(lambda item: item['label'] == label_id,
                                                        load_from_cache_file=data_args.overwrite_cache)
        if len(label_datasets[label_id]) > max_class[1]:
            max_class = [label_id, len(label_datasets[label_id])]

    for k in label_datasets.keys():
        label_datasets[k] = do_oversampling_to_multiclass_dataset(label_datasets[k], max_class[1])

    train_dataset = concatenate_datasets(label_datasets.values())
    return train_dataset


def do_oversampling_to_multiclass_dataset(train_dataset_class, majority_len):
    # NOTE: this is not tested yet
    datasets = [train_dataset_class]
    minority_len = len(train_dataset_class)
    if minority_len is None or minority_len == 0:  # avoid division with 0 or none
        return train_dataset_class
    num_full_minority_sets = int(majority_len / minority_len)
    for i in range(num_full_minority_sets - 1):  # -1 because one is already included in the training dataset
        datasets.append(train_dataset_class)

    remaining_minority_samples = majority_len % minority_len
    random_ids = np.random.choice(minority_len, remaining_minority_samples, replace=False)
    datasets.append(train_dataset_class.select(random_ids))

    train_dataset = concatenate_datasets(datasets)
    return train_dataset
