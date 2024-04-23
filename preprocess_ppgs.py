import ppgs

if __name__ == "__main__":
    # build paths
    wav_paths = []
    with open("/workspace/vc_train.csv", "r") as f:
        for line in f:
            file_path = line.split("|")[0]
            wav_paths.append(file_path.strip())
    ppgs_paths = [path.replace(".wav", ".ppg.pt") for path in wav_paths]

    # compute ppgs
    ppgs.from_files_to_files(wav_paths, ppgs_paths, gpu=0)
