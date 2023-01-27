import json
import os
import librosa
import math
import audioread
import numpy as np


SAMPLE_RATE = 22050
DURATION = 30  # in seconds


def save_mfcc(dataset_path, op_json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {"mapping": [], "mfcc": [], "labels": []}

    num_samples_per_segment = SAMPLE_RATE * DURATION // num_segments
    expected_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    for ind, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("/")
            data["mapping"].append(dirpath_components[-1])  # Add genre name to mapping
            print(f"Processing {dirpath_components[-1]}")
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path)
                except audioread.exceptions.NoBackendError:
                    print("FREAD ERROR: ", file_path)
                    continue
                if sr != SAMPLE_RATE:
                    print(
                        f"File load Error: {file_path} has sample rate {sr} but we need {SAMPLE_RATE}"
                    )
                    continue

                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(
                        y=signal[start_sample:finish_sample],
                        sr=sr,
                        n_fft=n_fft,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length,
                    )
                    mfcc = mfcc.T

                    if len(mfcc) == expected_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(
                            ind - 1
                        )  # -1 cause the first dirpath is dataset_path
                        print(f"{file_path}: segment {s + 1}")
                    else:
                        print(
                            f"MFCC Error: {file_path} has {len(mfcc)} but we need {expected_mfcc_vectors_per_segment}"
                        )

    with open(op_json_path, "w") as fp:
        json.dump(data, fp)


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    labels = np.array(data["labels"])

    return inputs, labels


if __name__ == "__main__":
    save_mfcc(
        "datasets/archive/Data/genres_original",
        "preprocessed_mfcc_data.json",
        num_segments=5,
    )
