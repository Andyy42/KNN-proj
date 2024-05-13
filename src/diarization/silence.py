#!/usr/bin/env python
import csv
from pathlib import Path
from pydub import AudioSegment, silence
import tqdm
import fire

def process_audio(recordings_path, diarization_path, output_path,min_silence_len=1000, silence_thresh=-40, csv_delimiter="tab"):
    recordings_path = Path(recordings_path)
    diarization_path = Path(diarization_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    length_summary = {}
    old_length_summary = {}

    if csv_delimiter == "tab":
        delimiter="\t"
    elif csv_delimiter == ",":
        delimiter=","

    # Iterate over all subdirectories in the recordings directory
    for subdir in recordings_path.glob('*'):
        if subdir.is_dir():
            output_subdir = output_path / subdir.name
            output_subdir.mkdir(parents=True, exist_ok=True)

            total_length = 0
            old_total_length = 0

            # Process each WAV file in the directory
            for wav_file in tqdm.tqdm(list(subdir.glob('*.wav'))):
                print("Proccesing:", wav_file)
                diarization_file = diarization_path / subdir.name / f"{wav_file.stem}_unwanted.csv"
                
                audio = AudioSegment.from_wav(wav_file)
                filtered_audio = AudioSegment.silent(duration=0)

                # Read diarization labels and exclude the unwanted parts
                if diarization_file.exists():
                    with open(diarization_file, 'r') as f:
                        csv_reader = csv.reader(f,delimiter=delimiter )
                        next(csv_reader)  # Skip header if it exists
                        unwanted_segments = [(1_000*float(start), 1_000*float(end)) for start, end, _ in csv_reader]
                        
                    # Include segments not listed as unwanted
                    last_end = 0
                    for start, end in sorted(unwanted_segments):
                        if last_end < start:  # Avoid overlapping with previous unwanted segment
                            filtered_audio += audio[last_end:start]
                        last_end = max(last_end, end)
                    if last_end < len(audio):
                        filtered_audio += audio[last_end:]
                else:
                    filtered_audio = audio

                # Silence removal
                chunks = silence.split_on_silence(
                    filtered_audio,
                    min_silence_len=min_silence_len,  # minimum length of a silence to be considered for splitting
                    silence_thresh=silence_thresh,    # silence threshold
                    keep_silence=200 # keep 200 ms of silence at the start and end of the chunks
                )

                # Combine non-silent chunks
                filtered_audio = sum(chunks, AudioSegment.silent(duration=0))

                # Export filtered audio
                filtered_file = output_subdir / wav_file.name
                filtered_audio.export(filtered_file, format='wav')
                
                # Update total length for the directory
                total_length += len(filtered_audio)
                old_total_length += len(audio)

            # Store the total length in the summary dictionary
            length_summary[subdir.name] = total_length
            old_length_summary[subdir.name] = old_total_length

    # Write the length summary to a text file
    with open(output_path / 'lengths.txt', 'w') as f:
        f.write(f"Dir, New len, Old len\n")
        for (dir_name, total_len), (_, old_total_len) in zip(length_summary.items(), old_length_summary()):
            f.write(f"{dir_name}, {total_len // 1000}, {old_total_len // 1000}\n")

def main():
    fire.Fire(process_audio)

if __name__ == "__main__":
    main()

