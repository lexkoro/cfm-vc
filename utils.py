import argparse
import glob
import json
import logging
import math
import os
import re
import subprocess
import sys

import numpy as np
import torch
from scipy.io.wavfile import read
from torch.nn import functional as F

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logger = logging


def _get_matplotlib_pyplot():
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt

    return plt


def _figure_to_numpy(fig, close=True):
    fig.canvas.draw()
    # Keep image output consistent as RGB for tensorboard HWC logging.
    data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    if close:
        plt = _get_matplotlib_pyplot()
        plt.close(fig)
    return data


def plot_data_to_numpy(x, y):
    plt = _get_matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(x)
    ax.plot(y)
    fig.tight_layout()

    return _figure_to_numpy(fig)


def get_content(cmodel, y):
    with torch.no_grad():
        c = cmodel.extract_features(y.squeeze(1))[0]
    c = c.transpose(1, 2)
    return c


def mle_loss(z, m, logs, mask):
    l = torch.sum(logs) + 0.5 * torch.sum(
        torch.exp(-2 * logs) * ((z - m) ** 2)
    )  # neg normal likelihood w/o the constant term
    l = l / torch.sum(
        torch.ones_like(z) * mask
    )  # averaging across batch, channel and time axes
    l = l + 0.5 * math.log(2 * math.pi)  # add the remaining constant term
    return l


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    return loss


def calc_same_padding(kernel_size: int):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "dec" in k or "disc" in k
            # print("load", k)
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
                print(
                    "%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning."
                    % k
                )
                logger.info("%s is not in the checkpoint" % k)
                new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("load ")
    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x):
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (
            x_sorted("G_default")[:-n_ckpts_to_keep]
            + x_sorted("G_ema")[:-n_ckpts_to_keep]
        )
    ]

    def del_info(fn):
        return logger.info(f".. Free up space by deleting ckpt {fn}")

    def del_routine(x):
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_default_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram, return_figure=False):
    plt = _get_matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Channels")
    fig.tight_layout()

    data = _figure_to_numpy(fig, close=not return_figure)

    if return_figure:
        return data, fig
    return data


def plot_alignment_to_numpy(alignment, info=None):
    plt = _get_matplotlib_pyplot()

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Encoder timestep")
    fig.tight_layout()

    return _figure_to_numpy(fig)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/config.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path, infer_mode=False):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config) if not infer_mode else InferHParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def repeat_expand_2d(content, target_len, mode="left"):
    # content : [h, t]
    return (
        repeat_expand_2d_left(content, target_len)
        if mode == "left"
        else repeat_expand_2d_other(content, target_len, mode)
    )


def repeat_expand_2d_left(content, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(
        content.device
    )
    temp = torch.arange(src_len + 1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos + 1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(content, target_len, mode="nearest"):
    # content : [h, t]
    content = content[None, :, :]
    target = F.interpolate(content, size=target_len, mode=mode)[0]
    return target


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, index):
        return self.__dict__.get(index)


class InferHParams(HParams):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = InferHParams(**v)
            self[k] = v

    def __getattr__(self, index):
        return self.get(index)


class Volume_Extractor:
    def __init__(self, hop_size=512):
        self.hop_size = hop_size

    def extract(self, audio):  # audio: 2d tensor array
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio**2
        audio2 = torch.nn.functional.pad(
            audio2,
            (int(self.hop_size // 2), int((self.hop_size + 1) // 2)),
            mode="reflect",
        )
        volume = torch.nn.functional.unfold(
            audio2[:, None, None, :], (1, self.hop_size), stride=self.hop_size
        )[:, :, :n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume)
        return volume
