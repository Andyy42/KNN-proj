#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description:
# Author: Ondřej Odehnal <xodehn09@vutbr.cz>
# =============================================================================
"""Create dev and test data split"""
# =============================================================================
# Imports
# =============================================================================
import random
from dataclasses import dataclass, astuple
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import argparse
import sys

@dataclass
class Data:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    # Allows unpacking of dataclasses
    def __iter__(self):
        return iter(astuple(self))


def _save_split(X, y, file: Path, extension: str) -> None:
    """
    Save the split data and labels to a new file.

    Args:
        X (numpy.ndarray): The data to be saved.
        y (numpy.ndarray): The labels to be saved.
        file (Path): The path to the original file.
        extension (str): The extension to be added to the new file.
    """
    new_file = file.parent / f"{file.stem}_{extension}{file.suffix}"

    with h5py.File(new_file, "w") as hf:
        hf.create_dataset(
            "Data",
            data=X,
            dtype=X.dtype,
            compression="gzip",
        )
        hf.create_dataset(
            "Name",
            data=y,
            dtype=y.dtype,
            compression="gzip",
        )


def save_splits(X_dev, y_dev, X_test, y_test, original_file: Path):
    _save_split(X_dev, y_dev, original_file, extension="dev")
    _save_split(X_test, y_test, original_file, extension="test")


def check_split(y_dev, y_test) -> None:
    # NOTE: np.unique sorts values from lower to higher by "unique" values
    dev_unique, dev_counts = np.unique(y_dev, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    assert (dev_unique == test_unique).all()
    total = dev_counts + test_counts
    test_split = test_counts / total
    # test_split = 1 - dev_split


    df = pd.DataFrame(dict(labels=dev_unique, dev_counts=dev_counts, test_counts=test_counts, total=total, test_percentage=test_split))
    print(df)

# Split strategy
def _pick_n_from_each_group(df, groups, n: int) -> tuple:
    """
    Picks 'n' names from each group in the given DataFrame based on the count column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        groups (list): The list of unique group values.
        n (int): The number of names to pick from each group.
        
    Returns:
        tuple: A tuple containing two lists - dev_names and test_names.
            dev_names (list): The list of names not picked from each group.
            test_names (list): The list of names picked from each group.
    """
    
    q_higher = df["count"].quantile(q=0.75)
    q_lower = df["count"].quantile(q=0.25)
    dev_names = []
    test_names = []
    for group in groups:
        group_df = df[df["unique"] == group]
        print(f"{q_higher= }")
        print(f"{q_lower= }")
        # Remove values lower than the 25th quantile or higher than the 75th quantile
        quantile_mask = (group_df["count"] > q_lower) & (group_df["count"] < q_higher)
        names = group_df[quantile_mask]["name"][:n]

        test_names.extend(group_df[group_df["name"].isin(names)]["name"])
        dev_names.extend(group_df[~group_df["name"].isin(names)]["name"])

    return dev_names, test_names

def average_data(data, group_id, names, n: int = 2):
    """
    A function that averages the data based on the given group_id and names.

    Args:
        data (numpy.ndarray): The input data array.
        group_id (numpy.ndarray): The group IDs for each data point.
        names (numpy.ndarray): The names for each data point.
        n (int, optional): The number of samples to average. Defaults to 2.

    Returns:
        Data: A Data object containing the averaged data.

    """
    new_names = []
    new_groups = []
    new_data = []

    for name in np.unique(names):
        mask = names == name
        masked_data = data[mask, :]
        modulo = (len(masked_data) % n)
        to_pad = n - modulo if modulo else 0
        masked_data = np.pad(masked_data, ((0,to_pad),(0,0)), 'constant', constant_values=np.nan)

        emb_len = masked_data.shape[1]

        split_data = np.reshape(masked_data, (-1,n,emb_len))
        new_d = np.nanmean(split_data,axis=1)

        s = len(new_d)

        new_data.append(new_d)
        new_names.extend([name] * s)
        new_groups.extend([group_id[mask][0]] * s)

    return Data(X=np.concatenate(new_data, axis=0), y=np.array(new_groups), ids=np.array(new_names))


def create_split(
    name: np.ndarray, group_id: np.ndarray, data: np.ndarray, ignore_groups, average_n:int=1
):
    """_summary_

    Args:
        name (np.ndarray): _description_
        group_id (np.ndarray): _description_
        data (np.ndarray): _description_
        ignore_groups (_type_): _description_
        average_n (int): Averages consecutive data samples into one sample

    Returns:
        tuple: X_dev, y_dev, X_test, y_test
    """
    if average_n > 1:
        dataset = average_data(
            data, group_id, name, n=average_n
        )  # TODO: Test for different N
        data, group_id, name 
    df = pd.DataFrame(dict(name=name, group_id=group_id))

    # Step 1: Create a list of unique "names" and their corresponding "group" and shuffle them.
    unique_names_groups_df = (
        df.groupby("name")["group_id"]
        .agg(["unique", "count"])
        .sample(frac=1)
        .reset_index()
    )

    # Step 2: Sampling based on Groups.
    groups = set(group_id) - set(ignore_groups)
    print(unique_names_groups_df)
    dev_names, test_names = _pick_n_from_each_group(
        df=unique_names_groups_df, groups=groups, n=15
    )

    # Step 3: Expand to original data and return data with group_id labels.
    dev_data = Data(
        X=data[df["name"].isin(dev_names)],
        y=group_id[df["name"].isin(dev_names)],
        ids=name[df["name"].isin(dev_names)],
    )
    test_data = Data(
        X=data[df["name"].isin(test_names)],
        y=group_id[df["name"].isin(test_names)],
        ids=name[df["name"].isin(test_names)],
    )

    assert len(dev_data.X) + len(test_data.X)
    +len(df[df["group_id"].isin(ignore_groups)]) == len(data)

    return dev_data, test_data 

    # dev_names = []
    # test_names = []
    # for group in set(group_id) - set(ignore_groups):
    #     group_df = unique_names_groups_df[
    #         unique_names_groups_df["unique"] == group
    #     ].copy()
    #     group_df["cumulative_percentage"] = (
    #         group_df["count"].cumsum() / group_df["count"].sum()
    #     )
    #     # Assign according to desired split percentage. For example, 80-20 split.
    #     dev_names.extend(group_df[group_df["cumulative_percentage"] <= 0.8]["name"])
    #     test_names.extend(group_df[group_df["cumulative_percentage"] > 0.8]["name"])


def load_h5_file(file, group_index="SubgroupID"):
    """Reads h5 file with specific group index and returns loaded data as numpy arrays

    Returns:
        tuple: name, group_id, data
    """
    with h5py.File(file, "r") as hf:
        name = np.array(hf["Name"]).flatten()
        group_id = np.array(hf[group_index]).flatten()
        data = np.array(hf["Data"])
    return name, group_id, data


def create_split_from_file(file: Path, group_index, ignore_groups):
    name, group_id, data = load_h5_file(file=file, group_index=group_index)
    return create_split(
        name=name,
        group_id=group_id,
        data=data,
        ignore_groups=ignore_groups,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dev and test data split")
    parser.add_argument(
        "file", type=Path, help="Path to the directory containing .gz files"
    )
    args = parser.parse_args()

    dev_data, test_data = create_split_from_file(
        args.file, group_index="SubgroupID", ignore_groups=[b"5", b"7"]
    )
    check_split(y_dev=dev_data.y, y_test=test_data.y)
    save_splits(
        X_dev=dev_data.X, y_dev=dev_data.y, X_test=test_data.X, y_test=test_data.y, original_file=args.file
    )


