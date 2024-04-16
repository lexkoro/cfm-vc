import os
from concurrent.futures import ProcessPoolExecutor

import librosa
import ppgs
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

import utils

sampling_rate = 22050
hop_length = 256
speech_encoder = "vec768l12"
device = "cuda:0"
f0p = "fcpe"


def process_one(filename, hmodel, f0p, device):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    audio_norm = torch.FloatTensor(wav)
    audio_norm = audio_norm.unsqueeze(0)

    soft_path = filename.replace(".wav", ".soft.pt")
    if not os.path.exists(soft_path):
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        wav16k = torch.from_numpy(wav16k).to(device)
        c = hmodel.encoder(wav16k)
        torch.save(c.cpu(), soft_path)

    f0_path = filename.replace(".wav", ".rmvpe.pt")
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(
            f0p,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            device=device,
            threshold=0.05,
        )

        f0, uv = f0_predictor.compute_f0_uv(wav)

        # Assuming f0 and uv are numpy arrays
        f0_tensor = torch.from_numpy(f0)
        uv_tensor = torch.from_numpy(uv)

        # Save as a dictionary for clarity
        data_to_save = {"f0": f0_tensor, "uv": uv_tensor}
        torch.save(data_to_save, f0_path)


def process_batch(file_chunk, f0p, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    if torch.cuda.is_available():
        gpu_id = rank % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Rank {rank} uses device {device}")
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    for filename in tqdm(file_chunk, position=rank):
        process_one(filename, hmodel, f0p, device)


def parallel_process(filenames, num_processes, f0p, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            tasks.append(executor.submit(process_batch, file_chunk, f0p, device=device))
        for task in tqdm(tasks, position=0):
            task.result()


def preprocess_ppgs(file_paths):
    ppgs_paths = [path.replace(".wav", ".ppg.pt") for path in file_paths]
    ppgs.from_files_to_files(file_paths, ppgs_paths, gpu=0)


if __name__ == "__main__":
    wav_paths = []
    with open("/workspace/vc_train.csv", "r") as f:
        for line in f:
            file_path = line.split("|")[0]
            wav_paths.append(file_path.strip())

    # preprocess f0 and hubert
    parallel_process(wav_paths, 6, f0p, device)

    # preprocess ppgs
    preprocess_ppgs(wav_paths)
