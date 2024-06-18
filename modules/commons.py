import math
import random
from itertools import groupby

import numpy as np
import torch
from einops import repeat
from torch.nn import functional as F


def rand_mel_segment(mel, mel_lengths, min_out_size):
    feats = mel.size(1)
    max_mel_length = max(mel_lengths)
    out_size = max(min_out_size, max_mel_length // 2)
    out_size = int(
        min(out_size, max_mel_length)
    )  # if max length < out_size, then decrease out_size

    # adjust out size by finding the largest multiple of 4 which is smaller than it
    max_offset = (mel_lengths - out_size).clamp(0)
    offset_ranges = list(
        zip([0] * max_offset.shape[0], max_offset.cpu().numpy().astype(int))
    )
    out_offset = torch.LongTensor(
        [
            torch.tensor(random.choice(range(start, end)) if end > start else 0)
            for start, end in offset_ranges
        ]
    ).to(mel_lengths)

    y_cut = torch.zeros(
        mel.shape[0], feats, out_size, dtype=mel.dtype, device=mel.device
    )
    y_cut_lengths = []
    for i, (y_, out_offset_) in enumerate(zip(mel, out_offset)):
        y_cut_length = out_size + (mel_lengths[i] - out_size).clamp(None, 0)
        y_cut_lengths.append(y_cut_length)
        cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
        y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]

    y_cut_lengths = torch.LongTensor(y_cut_lengths).to(mel_lengths.device)

    return y_cut, y_cut_lengths


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


def update_adversarial_weight(iteration, warmup_steps=10000, adv_max_weight=1e-2):
    """Update adversarial weight value based on iteration"""
    weight_iter = iteration * warmup_steps**-1.5 * adv_max_weight / warmup_steps**-0.5
    weight = min(adv_max_weight, weight_iter)

    return weight


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def slice_pitch_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments_with_pitch(x, pitch, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    ret_pitch = slice_pitch_segments(pitch, ids_str, segment_size)
    return ret, ret_pitch, ids_str


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if "Depthwise_Separable" in classname:
        m.depth_conv.weight.data.normal_(mean, std)
        m.point_conv.weight.data.normal_(mean, std)
    elif classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def rand_spec_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


@torch.jit.script
def simplified_gated_activation(input_a, n_channels):
    n_channels_int = n_channels[0]
    t_act = torch.tanh(input_a[:, :n_channels_int, :])
    s_act = torch.sigmoid(input_a[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = (
        path
        - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[
            :, :-1
        ]
    )
    path = path * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def fix_y_by_max_length(y, y_max_length):
    B, D, L = y.shape
    assert y_max_length >= L
    if y_max_length == L:
        return y
    else:
        new_y = torch.zeros(size=(B, D, y_max_length)).to(y.device)
        new_y[:, :, :L] = y
        return new_y


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def maximum_path_numpy(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool_)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[
            :, :-1
        ]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


# def average_over_durations(values, durs):
#     """
#     - in:
#         - values: B, 1, T_de
#         - durs: B, T_en
#     - out:
#         - avg: B, 1, T_en
#     """
#     durs_cums_ends = torch.cumsum(durs, dim=1).long()
#     durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
#     values_nonzero_cums = torch.nn.functional.pad(
#         torch.cumsum(values != 0.0, dim=2), (1, 0)
#     )
#     values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

#     bs, l = durs_cums_ends.size()
#     n_formants = values.size(1)
#     dcs = repeat(durs_cums_starts, "bs l -> bs n l", n=n_formants)
#     dce = repeat(durs_cums_ends, "bs l -> bs n l", n=n_formants)

#     values_sums = (
#         torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
#     ).to(values.dtype)
#     values_nelems = (
#         torch.gather(values_nonzero_cums, 2, dce)
#         - torch.gather(values_nonzero_cums, 2, dcs)
#     ).to(values.dtype)

#     avg = torch.where(
#         values_nelems == 0.0, values_nelems, values_sums / values_nelems
#     ).to(values.dtype)
#     return avg


def average_over_durations(values: torch.Tensor, durs: torch.Tensor) -> torch.Tensor:
    """
    - in:
        - values: B, C, T_de (B: batch size, C: number of channels, T_de: decoder timesteps)
        - durs: B, T_en (B: batch size, T_en: encoder timesteps)
    - out:
        - avg: B, C, T_en (B: batch size, C: number of channels, T_en: encoder timesteps)
    """
    # Compute cumulative sums of durations to get start and end indices
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))

    # Compute cumulative sums of values and non-zero values
    values_nonzero_cums = torch.nn.functional.pad(
        torch.cumsum(values != 0.0, dim=2), (1, 0)
    )
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    n_channels = values.size(1)
    dcs = repeat(durs_cums_starts, "bs l -> bs n l", n=n_channels)
    dce = repeat(durs_cums_ends, "bs l -> bs n l", n=n_channels)

    # Compute sums and counts of elements within the durations
    values_sums = (
        torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
    ).to(values.dtype)
    values_nelems = (
        torch.gather(values_nonzero_cums, 2, dce)
        - torch.gather(values_nonzero_cums, 2, dcs)
    ).to(values.dtype)

    # Prevent division by zero
    values_nelems = torch.where(
        values_nelems == 0, torch.ones_like(values_nelems), values_nelems
    )

    # Compute the average values
    avg = values_sums / values_nelems

    return avg


def mel2unit(features, durations):
    """
    Reverts the process of expanding a token sequence to a mel spectrogram.

    Args:
    mel_spectrogram (torch.Tensor): The input mel spectrogram tensor of shape (batch, n_mel_bins, frame_length).
    durations (torch.Tensor): The tensor of durations of shape (batch, num_tokens).

    Returns:
    torch.Tensor: The aggregated token-level tensor of shape (batch, n_mel_bins, num_tokens).
    """
    batch_size, n_mel_bins, frame_length = features.shape
    _, num_tokens = durations.shape

    # Initialize the output tensor
    token_level_spectrogram = torch.zeros(
        (batch_size, n_mel_bins, num_tokens), device=features.device
    )

    for b in range(batch_size):
        current_frame = 0
        for t in range(num_tokens):
            duration = durations[b, t].item()
            if duration > 0:
                # Aggregate the frames for the current token
                token_frames = features[b, :, current_frame : current_frame + duration]
                token_level_spectrogram[b, :, t] = token_frames.mean(dim=1)
                current_frame += duration

    return token_level_spectrogram
