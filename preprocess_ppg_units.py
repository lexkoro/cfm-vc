import concurrent.futures
from glob import glob

import ppgs
import torch
import tqdm

import utils
from modules.commons import dedup_seq


def process_ppg_unit(filename):
    ppg_unit_path = filename.replace(".ppg.pt", ".ppg_unit.pt")
    f0_path = filename.replace(".ppg.pt", ".rmvpe.pt")
    loaded_data = torch.load(f0_path)
    f0 = loaded_data["f0"].unsqueeze(0)
    ppg = torch.load(filename)

    ppg = utils.repeat_expand_2d(ppg.squeeze(0), f0.shape[1], mode="nearest")

    sparse_ppg = ppgs.sparsify(
        ppg=ppg, method="percentile", threshold=torch.Tensor([0.85])
    )
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
