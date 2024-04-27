#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description: H5 Convert
# Author: Ondřej Odehnal <xodehn09@vutbr.cz>
# =============================================================================
"""Converts embeddings from scp to H5 format"""
# =============================================================================
# Imports
# =============================================================================

import gzip
import h5py
from pathlib import Path
import numpy as np
import logging
import argparse
import kaldiio

logging.basicConfig(
    filename="conversion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    # stream=sys.stdout
)

def parse_labels(embd_file: Path) -> (str, str, str):
    """
    Expects filename in format:
    "XXXXXX_XXXX_04-Podkrkonosi_SPLIT_010.i.gz"
    where
    "XXXXXX_XXXX_04-Podkrkonosi"

    ## Groups (Oblasti)
    1-n Česká 
    2-n Středomoravská 
    3-n Východomoravská
    4-n Slezkomoravská
    5 Pohranici nareci (nemeckojazycna)
    7 Pohranici nareci
    
    ## Subgroups (Podoblasti)
    1-1 Severovýchodočeská
    1-2 Středočeská
    1-3 Jihozápadočeská
    1-4 Českomoravská
    
    2-1 Jižní
    2-2 Západní
    2-3 Východní
    
    3-1 Slovácko 
    3-2 Zlinsko
    3-3 Valašsko
    
    4-1 Slezskomoravská
    4-2 Slezskopolská
    """
    parent_dir = (
        embd_file.parent.name
    )  # Contains group and subgroup ID in format: [0-4]-[0-4]
    name = embd_file.name
    if "SPLIT" not in name:
        raise ValueError("Filename does not contain 'SPLIT' keyword")
    recording_name = name.split("_SPLIT")[0]
    # if ".wav" not in name:
    #      raise ValueError("Filename does not contain '.wav' keyword")
    # recording_name = name.split(".wav")[0]
    group_id = parent_dir[0]
    subgroup_id = parent_dir

    return recording_name, group_id, subgroup_id


def h5_store_string(label: str, data: str, hf: h5py.File):
    try:
      new_data = np.array([data.encode('utf-8')], dtype="S")
    except UnicodeEncodeError as e:
      print(data)
      raise e

    if label not in hf:
        hf.create_dataset(
            label,
            dtype=new_data.dtype,
            shape=(1, 1),
            # data=new_data,
            compression="gzip",
            chunks=True,
            maxshape=(None, 1),
        )
    else:
        hf[label].resize((hf[label].shape[0] + 1), axis=0)
    hf[label][-1] = new_data[0]


def h5_store_embedding(label, embeddings, hf: h5py.File):
    if label not in hf:
        hf.create_dataset(
            label,
            shape=(1, embeddings.size),
            dtype=embeddings.dtype,
            compression="gzip",
            chunks=True,
            maxshape=(None, embeddings.size),
        )
    else:
        hf[label].resize((hf[label].shape[0] + 1), axis=0)
    hf[label][-1] = embeddings


def load_embd(embd_file: Path, hf: h5py.File) -> None:
    try:
        with gzip.open(embd_file, "rb") as gzfile:
            data = gzfile.read().decode("utf-8").strip()
            if data:
                # Convert the single line directly to a NumPy array
                embeddings_array = np.fromstring(data, sep=" ")  # [:, np.newaxis]
                h5_store_embedding("Data", embeddings_array, hf=hf)

                # Store the file name in a vector
                recording_name, group_id, subgroup_id = parse_labels(
                    embd_file=embd_file
                )
                h5_store_string("Name", recording_name, hf=hf)
                h5_store_string("GroupID", group_id, hf=hf)
                h5_store_string("SubgroupID", subgroup_id, hf=hf)

                logging.info(f"Converted {embd_file} and added to {hf.filename}")
            else:
                logging.warning(f"Empty data in {embd_file}")

    except Exception as e:
        logging.error(f"Error converting {embd_file}: {str(e)}")
        raise e

def load_embd_from_scp(embd_file, embeddings_array, hf):
   # Convert the single line directly to a NumPy array
   # embeddings_array = np.fromstring(data, sep=" ")  # [:, np.newaxis]
   h5_store_embedding("Data", embeddings_array, hf=hf)

   # Store the file name in a vector
   recording_name, group_id, subgroup_id = parse_labels(
       embd_file=embd_file
   )
   h5_store_string("Name", recording_name, hf=hf)
   h5_store_string("GroupID", group_id, hf=hf)
   h5_store_string("SubgroupID", subgroup_id, hf=hf)

   logging.info(f"Converted {embd_file} and added to {hf.filename}")


def process_directory(data_dir: Path, recursive: bool, output_h5: str):
    with h5py.File(output_h5, "w") as hf:
        for gz_file in data_dir.rglob("*.gz") if recursive else data_dir.glob("*.gz"):
            load_embd(gz_file, hf)


def process_scp_file(scp_file: Path, output_h5):
    with h5py.File(output_h5, "w") as hf:
      for utt, emb in kaldiio.load_scp_sequential(str(scp_file)):
        load_embd_from_scp(Path(utt), emb, hf)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert compressed .gz files to a single HDF5 file with data and names."
    )
    parser.add_argument(
        "scp_file", type=Path, help="Path to the scp file"
    )

    parser.add_argument(
        "output_h5", type=str, help="Output HDF5 file to save data and names"
    )

    args = parser.parse_args()

    process_scp_file(args.scp_file, args.output_h5)

