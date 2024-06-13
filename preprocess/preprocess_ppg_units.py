import numpy as np
import torch
import tqdm
from sklearn.cluster import KMeans
import concurrent.futures
from glob import glob
from itertools import groupby
import ppgs

np.random.seed(1234)
kmeans = KMeans(n_clusters=40, verbose=0)


def collate_1d_or_2d(
    values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1
):
    if len(values[0].shape) == 1:
        return collate_1d(values, pad_idx, left_pad, shift_right, max_len, shift_id)
    else:
        return collate_2d(values, pad_idx, left_pad, shift_right, max_len)


def collate_1d(
    values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def dedup_seq(seq):
    B, L = seq.shape
    vals, counts = [], []
    for i in range(B):
        val, count = zip(*[(k.item(), sum(1 for _ in g)) for k, g in groupby(seq[i])])
        vals.append(torch.LongTensor(val))
        counts.append(torch.LongTensor(count))
    vals = collate_1d_or_2d(vals, 0)
    counts = collate_1d_or_2d(counts, 0)
    return vals, counts


def process_ppg_unit(file):
    ppg_unit_path = file.replace(".ppg.pt", ".ppg_unit.pt")
    feature = torch.load(file)

    sparse_ppg = ppgs.sparsify(
        ppg=feature, method="percentile", threshold=torch.Tensor([0.85])
    ).squeeze(1)
    most_probable_ppg = torch.argmax(sparse_ppg, dim=1)
    torch_features, features_dur = dedup_seq(most_probable_ppg)

    to_store = {
        "ppg_unit": torch_features.squeeze(0),
        "ppg_unit_dur": features_dur.squeeze(0),
    }

    torch.save(to_store, ppg_unit_path)


if __name__ == "__main__":
    gametts_ppgs = glob("/workspace/dataset/de/GameTTS/**/*.ppg.pt", recursive=True)

    # pool executor with tqdm
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        result = list(
            tqdm.tqdm(
                executor.map(process_ppg_unit, gametts_ppgs), total=len(gametts_ppgs)
            )
        )
