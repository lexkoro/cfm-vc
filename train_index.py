import argparse
import concurrent.futures
import os
import pickle

import utils

sovits_config_path = "/home/alexander/Projekte/so-vits-svc/logs/22k_cfm/config.json"
hps = utils.get_hparams_from_file(sovits_config_path, True)
speaker_dict = {v: k for k, v in hps.spk.items()}


def process_speaker_data(item):
    k, v = item
    print(f"now, index {k} feature...")
    index = utils.train_index(v)
    return k, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/config.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "--output_dir", type=str, default="logs/22k", help="path to output dir"
    )

    args = parser.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    spk_dic = hps.spk
    spk_dic_inv = {v: k for k, v in spk_dic.items()}

    result = {}

    files_per_speaker = {}

    with open(
        "/home/alexander/Projekte/so-vits-svc/filelists/gametts_train.txt", "r"
    ) as rf:
        for line in rf:
            wav_path, speaker_id = line.strip().split("|")

            speaker_name = speaker_dict[int(speaker_id)]

            if (
                any([x in speaker_name for x in ["male", "KCD"]])
                and "WoW" not in speaker_name
                and "child" not in speaker_name
                and "SVM" not in speaker_name
            ):
                if speaker_id not in files_per_speaker:
                    files_per_speaker[speaker_id] = []

                soft_path = wav_path.replace(".wav", ".soft.pt")
                files_per_speaker[speaker_id].append(soft_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        result = dict(executor.map(process_speaker_data, files_per_speaker.items()))

    with open(os.path.join(args.output_dir, "feature_and_index.pkl"), "wb") as f:
        pickle.dump(result, f)
