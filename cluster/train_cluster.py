import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from kmeans import KMeansGPU
from sklearn.cluster import KMeans, MiniBatchKMeans
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_cluster(
    in_dir, n_clusters, use_minibatch=True, verbose=False, use_gpu=False
):  # gpu_minibatch真拉，虽然库支持但是也不考虑
    if str(in_dir).endswith(".ipynb_checkpoints"):
        logger.info(f"Ignore {in_dir}")

    features = []
    nums = 0
    for path in tqdm.tqdm(in_dir):
        # for name in os.listdir(in_dir):
        #     path="%s/%s"%(in_dir,name)
        features.append(torch.load(path, map_location="cpu").squeeze(0).numpy().T)
        # print(features[-1].shape)
    features = np.concatenate(features, axis=0)

    features = features.astype(np.float32)
    logger.info(f"Clustering features of shape: {features.shape}")
    t = time.time()
    if use_gpu is False:
        if use_minibatch:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, verbose=verbose, batch_size=4096, max_iter=80
            ).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters, verbose=verbose).fit(features)
    else:
        kmeans = KMeansGPU(
            n_clusters=n_clusters,
            mode="euclidean",
            verbose=2 if verbose else 0,
            max_iter=500,
            tol=1e-2,
        )  #
        features = torch.from_numpy(features)  # .to(device)
        kmeans.fit_predict(features)  #

    print(time.time() - t, "s")

    x = {
        "n_features_in_": kmeans.n_features_in_
        if use_gpu is False
        else features.shape[1],
        "_n_threads": kmeans._n_threads if use_gpu is False else 4,
        "cluster_centers_": kmeans.cluster_centers_
        if use_gpu is False
        else kmeans.centroids.cpu().numpy(),
    }
    print("end")

    return x


def process_speaker_data(item):
    k, v = item
    print(f"now, index {k} feature...")
    x = train_cluster(v, 10000, use_minibatch=False, verbose=False, use_gpu=use_gpu)
    return k, x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default="logs/22k", help="path of model output directory"
    )
    parser.add_argument("--gpu", action="store_true", default=False, help="to use GPU")

    args = parser.parse_args()

    checkpoint_dir = args.output
    use_gpu = args.gpu
    n_clusters = 10000

    result = {}

    files_per_speaker = {}

    with open(
        "/home/alexander/Projekte/so-vits-svc/filelists/gametts_train.txt", "r"
    ) as rf:
        for line in rf:
            wav_path, speaker_id = line.strip().split("|")

            if speaker_id not in files_per_speaker:
                files_per_speaker[speaker_id] = []

            soft_path = wav_path.replace(".wav", ".soft.pt")
            files_per_speaker[speaker_id].append(soft_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        result = dict(executor.map(process_speaker_data, files_per_speaker.items()))

    checkpoint_path = os.path.join(checkpoint_dir, f"kmeans_{n_clusters}.pt")
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        result,
        checkpoint_path,
    )
