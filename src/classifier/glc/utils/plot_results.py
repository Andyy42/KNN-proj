#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description:
# Author: Ondřej Odehnal <xodehn09@vutbr.cz>
# =============================================================================
"""Plot accuracy and cprim over time."""
# =============================================================================
# Imports
# =============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fire

SPLIT_S=40

def plot_comparison(result_dir1: str, result_dir2: str):
    """
    Plots the comparison of accuracy and C_prim over time for results.

    Args:
        result_dir1 (str): The path to the first result directory.
        result_dir2 (str): The path to the second result directory.
    """

    result_dir1 = Path(result_dir1)
    result_dir2 = Path(result_dir2)

    def make_df(result_dir):
        """
        Creates a DataFrame from the result directory.

        Args:
            result_dir (Path): The path to the result directory.

        Returns:
            pd.DataFrame: The DataFrame containing accuracy, C_prim, and time values.
        """
        acc = np.fromfile(result_dir / "accuracy.txt", sep="\n")
        cprim = np.fromfile(result_dir / "cprim.txt", sep="\n")
        ts = np.arange(start=SPLIT_S, stop=SPLIT_S*len(acc)+1, step=SPLIT_S, dtype=int)

        df = pd.DataFrame({
            "Accuracy": acc,
            r"$C_{prim}$": cprim,
            "Time [s]": ts
        })

        df = df[df["Time [s]"] < 900]
        df.set_index("Time [s]")
        return df

    df1 = make_df(result_dir1)
    df1["Model"] = "ECAPA-TDNN"
    df2 = make_df(result_dir2)
    df2["Model"] = "ResNet-18"

    df = pd.concat([df1, df2])

    fig, ax = plt.subplots(1,1, figsize=(9,3))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(SPLIT_S))
    sns.lineplot(ax=ax, data=df, x="Time [s]", y="Accuracy", hue="Model", markers=True)

    plt.legend()
    # plt.legend(labels=[r'$C_{prim}$', ])
    plt.grid()
    # ax.axvline(df["Time [s]"].iloc[df['Accuracy'].argmax()])
    plt.tight_layout()
    plt.savefig(result_dir1/"accuracy.pdf", dpi=300)
    plt.cla()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(SPLIT_S))
    sns.lineplot(data=df ,x="Time [s]", y=r"$C_{prim}$" , hue="Model", markers=True)
    plt.legend()
    # ax.axvline(df["Time [s]"].iloc[df["$C_{prim}$"].argmin()], color='tab:blue')
    plt.grid()
    plt.tight_layout()
    plt.savefig(result_dir1/"cprim.pdf", dpi=300)





def plot_results(result_dir: str):
    """
    Plots the results of a classification experiment.

    Args:
        result_dir (str): The directory path where the result files are located.

    Returns:
        None
    """

    result_dir = Path(result_dir)
    
    acc = np.fromfile(result_dir / "accuracy.txt", sep="\n")
    cprim = np.fromfile(result_dir / "cprim.txt", sep="\n")

    ts = np.arange(start=SPLIT_S, stop=SPLIT_S*len(acc)+1, step=SPLIT_S, dtype=int)

    df = pd.DataFrame({
        "Accuracy": acc,
        r"$C_{prim}$": cprim,
        "Time [s]": ts
    })

    df = df[df["Time [s]"] < 900]

    df.set_index("Time [s]")

    fig, ax = plt.subplots(1,1, figsize=(9,3))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(SPLIT_S))
    sns.lineplot(ax=ax, data=df, x="Time [s]", y="Accuracy", color='tab:red', markers=True)
    plt.legend(labels=['Accuracy', ])
    # plt.legend(labels=[r'$C_{prim}$', ])
    plt.grid()
    # ax.axvline(df["Time [s]"].iloc[df['Accuracy'].argmax()])
    plt.tight_layout()
    plt.savefig(result_dir/"accuracy.png", dpi=300)
    plt.cla()


    ax.xaxis.set_major_locator(ticker.MultipleLocator(SPLIT_S))
    sns.lineplot(data=df ,x="Time [s]", y=r"$C_{prim}$" , color='tab:blue', markers=True)
    plt.legend(labels=[r'$C_{prim}$', ])
    # ax.axvline(df["Time [s]"].iloc[df["$C_{prim}$"].argmin()], color='tab:blue')
    plt.grid()
    plt.tight_layout()
    plt.savefig(result_dir/"cprim.png", dpi=300)



if __name__ == "__main__":
    # fire.Fire(plot_results)
    fire.Fire(plot_comparison)

