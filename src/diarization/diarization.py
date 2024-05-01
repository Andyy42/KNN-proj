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
import yaml
import os
import pandas as pd
import torchaudio

from tqdm import tqdm
from pathlib import Path
from pyannote.audio import Pipeline


def mute_overlapped(waveform, sample_rate, overlapps, fade_samples=800):
    fade_in = torch.linspace(0, 1, fade_samples)
    fade_out = torch.linspace(1, 0, fade_samples)

    for overlapp in overlapps:
        start_idx = int(overlapp['Start [s]'] * sample_rate)
        end_index = int(overlapp['End [s]'] * sample_rate)

        fade_in_end_idx = max(0, min(start_idx + fade_samples, waveform.shape[1]))
        fade_in_samples = fade_in_end_idx - start_idx
        fade_out_start_idx = max(0, min(end_index - fade_samples, waveform.shape[1]))
        fade_out_samples = end_index - fade_out_start_idx
        waveform[:, start_idx:fade_in_end_idx] *= fade_in[fade_in.shape[0] - fade_in_samples:]
        waveform[:, fade_in_end_idx:fade_out_start_idx] = 0
        waveform[:, fade_out_start_idx:end_index] *= fade_out[:-fade_out_samples+1]

    return waveform, sample_rate


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

    script_path = os.path.dirname(__file__)

    # Load the existing configuration
    with open(os.path.join(script_path, "diarization_config.yaml"), 'r') as file:
        diarization_config = yaml.safe_load(file)

    with open(os.path.join(script_path, "overlapped_config.yaml"), 'r') as file:
        overlapped_config = yaml.safe_load(file)

    # Update the segmentation path to be relative to the script folder
    diarization_config['pipeline']['params']['segmentation'] = os.path.join(script_path, diarization_config['pipeline']['params']['segmentation'])
    overlapped_config['pipeline']['params']['segmentation'] = os.path.join(script_path, overlapped_config['pipeline']['params']['segmentation'])

    tmp_diarization_config = os.path.join(script_path, '.diarization_config.yaml.tmp')
    with open(tmp_diarization_config, 'w') as file:
        yaml.safe_dump(diarization_config, file)

    tmp_overlapped_config = os.path.join(script_path, '.overlapped_config.yaml.tmp')
    with open(tmp_overlapped_config, 'w') as file:
        yaml.safe_dump(overlapped_config, file)

    diarization_pipeline = Pipeline.from_pretrained(tmp_diarization_config)
    overlapped_pipeline = Pipeline.from_pretrained(tmp_overlapped_config)

    diarization_pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    overlapped_pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for file in tqdm(args.files):
        waveform, sample_rate = torchaudio.load(file)

        overlapped = overlapped_pipeline({'waveform': waveform, 'sample_rate': sample_rate})

        overlapps = []
        for speech in overlapped.get_timeline().support():
            overlapps.append({
                'Start [s]': speech.start,
                'End [s]': speech.end,
                'Speaker': 'OVERLAPPED'
            })

        muted_waveform, _ = mute_overlapped(waveform, sample_rate, overlapps)

        diarization = diarization_pipeline({'waveform': muted_waveform, 'sample_rate': sample_rate})

        start = []
        end = []
        speakers = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.end - turn.start < 0.8:
                continue

            start.append(turn.start)
            end.append(turn.end)
            speakers.append(speaker)

        df = pd.DataFrame({
            'Start [s]': start,
            'End [s]': end,
            'Speaker': speakers
        })

        df.to_csv(os.path.join(args.out_dir, f"{os.path.basename(file).split('.')[0]}.csv"), index=False)

        df2 = df[['Speaker']].copy()
        df2['Duration'] = df['End [s]'] - df['Start [s]']
        df2 = df2.groupby(by='Speaker', as_index=False).sum('Duration')

        min_speaker_id = df2['Speaker'][df2['Duration'].idxmin()]
        single_speaker = df['Speaker'].nunique() == 1

        if single_speaker:
            df_filtered = pd.DataFrame(columns=df.columns)
        else:
            df_filtered = df.drop(df[df['Speaker'] != min_speaker_id].index)

        df_filtered = pd.concat([df_filtered, pd.DataFrame(overlapps)], ignore_index=True)
        df_filtered.to_csv(os.path.join(args.out_dir, f"{os.path.basename(file).split('.')[0]}_unwanted.csv"), index=False)
