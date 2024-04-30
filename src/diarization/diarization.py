#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description:
# Author: Simon Košina <simonkosina@gmail.com>
# =============================================================================
"""Diarization."""
# =============================================================================
# Imports
# =============================================================================
import torch
import argparse
import os
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from pyannote.audio import Pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create speaker labels for audio files.")
    parser.add_argument(
        "files",
        type=Path,
        help="Audio files to create label",
        nargs="+"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Output directory to save labels",
        default=".",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pipeline = Pipeline.from_pretrained("config.yaml")
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for file in tqdm(args.files):
        diarization = pipeline(file)

        start = []
        end = []
        speakers = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start.append(turn.start)
            end.append(turn.end)
            speakers.append(speaker)

        df = pd.DataFrame({
            'Start [s]': start,
            'End [s]': end,
            'Speaker': speakers
        })

        df.to_csv(os.path.join('labels', f"{os.path.basename(
            file).split('.')[0]}.csv"), index=False)

        df2 = df[['Speaker']].copy()
        df2['Duration'] = df['End [s]'] - df['Start [s]']
        df2 = df2.groupby(by='Speaker', as_index=False).sum('Duration')

        min_speaker_id = df2['Speaker'][df2['Duration'].idxmin()]
        single_speaker = df['Speaker'].nunique() == 1

        if single_speaker:
            df_filtered = pd.DataFrame(columns=df.columns)
        else:
            df_filtered = df.drop(df[df['Speaker'] != min_speaker_id].index)
        df_filtered.to_csv(os.path.join('labels', f"{os.path.basename(
            file).split('.')[0]}_unwanted.csv"), index=False)
