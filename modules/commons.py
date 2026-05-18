import math

import einx
import numpy as np
import torch
from einops import repeat
from torch.nn import functional as F


def update_adversarial_weight(iteration, warmup_steps, adv_max_weight=1e-2):
    """Update adversarial weight value based on iteration"""
    weight_iter = iteration * warmup_steps**-1.5 * adv_max_weight / warmup_steps**-0.5
    weight = min(adv_max_weight, weight_iter)

    return weight


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    if input_b is not None:
        in_act = input_a
    else:
        in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
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


def slice_segments(x, ids_str, segment_size=4, pad_short=False):
    # pad the input tensor if it is shorter than the segment size
    if pad_short and x.shape[-1] < segment_size:
        x = torch.nn.functional.pad(x, (0, segment_size - x.size(2)))

    segments = torch.zeros_like(x[:, :, :segment_size])

    for i in range(x.size(0)):
        index_start = ids_str[i]
        index_end = index_start + segment_size
        x_i = x[i]
        if pad_short and index_end >= x.size(2):
            # pad the sample if it is shorter than the segment size
            x_i = torch.nn.functional.pad(x_i, (0, (index_end + 1) - x.size(2)))
        segments[i] = x_i[:, index_start:index_end]
    return segments


def rand_slice_segments(
    x, x_lengths=None, segment_size=4, let_short_samples=False, pad_short=False
):
    _x_lenghts = x_lengths.clone()
    B, _, T = x.size()
    if pad_short:
        if T < segment_size:
            x = torch.nn.functional.pad(x, (0, segment_size - T))
            T = segment_size
    if _x_lenghts is None:
        _x_lenghts = T
    len_diff = _x_lenghts - segment_size
    if let_short_samples:
        _x_lenghts[len_diff < 0] = segment_size
        len_diff = _x_lenghts - segment_size
    else:
        assert all(len_diff > 0), (
            f" [!] At least one sample is shorter than the segment size ({segment_size}). \n {_x_lenghts}"
        )
    segment_indices = (torch.rand([B]).type_as(x) * (len_diff + 1)).long()
    ret = slice_segments(x, segment_indices, segment_size, pad_short=pad_short)

    return ret, segment_indices


# def rand_mel_segment(mel, mel_lengths, frac_lengths_mask=(0.5, 0.7)):
#     batch, feats, _ = mel.size()

#     frac_lengths = torch.zeros((batch,)).float().uniform_(*frac_lengths_mask)
#     rand_span_mask = (
#         mask_from_frac_lengths(mel_lengths, frac_lengths).unsqueeze(1).to(mel.dtype)
#     )

#     max_mel_length = max(mel_lengths)
#     out_size = max(min_out_size, max_mel_length // 2)
#     out_size = min(
#         out_size, max_mel_length
#     )  # if max length < out_size, then decrease out_size

#     # adjust out size by finding the largest multiple of 4 which is smaller than it
#     max_offset = (mel_lengths - out_size).clamp(0)
#     offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
#     out_offset = torch.LongTensor(
#         [
#             torch.tensor(random.choice(range(start, end)) if end > start else 0)
#             for start, end in offset_ranges
#         ]
#     ).to(mel_lengths)

#     y_cut = torch.zeros(
#         mel.shape[0], feats, out_size, dtype=mel.dtype, device=mel.device
#     )
#     y_cut_lengths = []
#     for i, (y_, out_offset_) in enumerate(zip(mel, out_offset)):
#         y_cut_length = out_size + (mel_lengths[i] - out_size).clamp(None, 0)
#         y_cut_lengths.append(y_cut_length)
#         cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
#         y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]

#     y_cut_lengths = torch.LongTensor(y_cut_lengths).to(mel_lengths.device)
#     y_cut_mask = sequence_mask(y_cut_lengths, y_cut.size(2)).unsqueeze(1).to(mel.dtype)

#     return y_cut, y_cut_lengths, y_cut_mask


def rand_mel_segment(mel, mel_lengths, frac_lengths_mask=(0.5, 0.7)):
    batch, feats, _ = mel.size()

    frac_lengths = (
        torch.zeros((batch,), device=mel.device).float().uniform_(*frac_lengths_mask)
    )
    rand_span_mask = mask_from_frac_lengths(mel_lengths, frac_lengths).unsqueeze(1)

    # Calculate new lengths for each segment
    cut_lengths = (rand_span_mask.squeeze(1).sum(dim=-1)).long()

    # Find the maximum length in the batch
    max_length = cut_lengths.max()

    # Create a new tensor to store extracted segments
    cut_mel = torch.zeros(batch, feats, max_length, device=mel.device, dtype=mel.dtype)

    # Extract the segments where mask is True
    for i in range(batch):
        mask = rand_span_mask[i, 0]
        segment = mel[i, :, mask.bool()]
        cut_mel[i, :, : cut_lengths[i]] = segment

    cut_mask = sequence_mask(cut_lengths, max_length).unsqueeze(1).to(mel.dtype)

    return cut_mel, cut_lengths, cut_mask, rand_span_mask


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


def shift_1d(x):
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_len = torch.max(lengths).item()
    ids = (
        torch.arange(0, max_len, device=lengths.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


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


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    seq_range = torch.arange(
        max_len, dtype=sequence_length.dtype, device=sequence_length.device
    )
    # B x T_max
    mask = seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)
    return mask


def lengths_from_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.dim() == 3:
        masks = masks.squeeze(1)
    lengths = torch.sum(masks, dim=1).int()
    return lengths


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    length = (length / factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length


def fix_y_by_max_length(y, y_max_length):
    B, D, L = y.shape
    assert y_max_length >= L
    if y_max_length == L:
        return y
    else:
        new_y = torch.zeros(size=(B, D, y_max_length)).to(y.device)
        new_y[:, :, :L] = y
        return new_y


def generate_path(duration, mask):
    """
    Shapes:
        - duration: :math:`[B, T_en]`
        - mask: :math:'[B, T_en, T_de]`
        - path: :math:`[B, T_en, T_de]`
    """
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


# @jit(nopython=True)
# def _average_by_duration(ds, xs, text_lengths, feats_lengths):
#     B = ds.shape[0]
#     xs_avg = np.zeros_like(ds)
#     ds = ds.astype(np.int32)
#     for b in range(B):
#         t_text = text_lengths[b]
#         t_feats = feats_lengths[b]
#         d = ds[b, :t_text]
#         d_cumsum = d.cumsum()
#         d_cumsum = [0] + list(d_cumsum)
#         x = xs[b, :t_feats]
#         for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])):
#             if len(x[start:end]) != 0:
#                 xs_avg[b,n] = x[start:end].mean()
#             else:
#                 xs_avg[b,n] = 0
#     return xs_avg

# def average_by_duration(ds, xs, text_lengths, feats_lengths):
#     """
#     Args:
#         ds (Tensor): Batched token duration (B,T_text)
#         xs (Tensor): Batched feature sequences to be averaged (B,T_feats)
#         text_lengths (Tensor): Text length tensor (B,)
#         feats_lengths (Tensor): Feature length tensor (B,)
#     Returns:
#         Tensor: Batched feature averaged according to the token duration (B, T_text)
#     """
#     device = ds.device
#     args = [ds, xs, text_lengths, feats_lengths]
#     args = [arg.detach().cpu().numpy() for arg in args]
#     xs_avg = _average_by_duration(*args)
#     xs_avg = torch.from_numpy(xs_avg).to(device)
#     return xs_avg


def average_over_durations(values, durs):
    """
    - in:
        - values: B, 1, T_de
        - durs: B, T_en
    - out:
        - avg: B, 1, T_en
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(
        torch.cumsum(values != 0.0, dim=2), (1, 0)
    )
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = repeat(durs_cums_starts, "bs l -> bs n l", n=n_formants)
    dce = repeat(durs_cums_ends, "bs l -> bs n l", n=n_formants)

    values_sums = (
        torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)
    ).to(values.dtype)
    values_nelems = (
        torch.gather(values_nonzero_cums, 2, dce)
        - torch.gather(values_nonzero_cums, 2, dcs)
    ).to(values.dtype)

    avg = torch.where(
        values_nelems == 0.0, values_nelems, values_sums / values_nelems
    ).to(values.dtype)
    return avg


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


# def normalize_f0(f0, x_mask, uv, random_scale=True):
#     # calculate means based on x_mask
#     uv_sum = torch.sum(uv, dim=1, keepdim=True)
#     uv_sum[uv_sum == 0] = 9999
#     means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

#     if random_scale:
#         factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
#     else:
#         factor = torch.ones(f0.shape[0], 1).to(f0.device)
#     # normalize f0 based on means and factor
#     f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
#     if torch.isnan(f0_norm).any():
#         exit(0)
#     return f0_norm * x_mask


def normalize_f0(f0, uv, random_scale=True):
    # calculate means based on x_mask
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum

    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    # normalize f0 based on means and factor
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        exit(0)
    return f0_norm


def create_partial_mask(mel_lengths, fraction=1 / 3):
    max_len = mel_lengths.max()
    batch_size = mel_lengths.shape[0]

    # Create the original mask
    original_mask = sequence_mask(mel_lengths, max_len).unsqueeze(1).to(torch.float32)

    # Calculate the lengths to mask
    mask_lengths = (mel_lengths * fraction).long()

    # Create a mask for the portion to be masked out
    partial_mask = torch.ones((batch_size, 1, max_len), device=mel_lengths.device)

    for i, (length, mask_length) in enumerate(zip(mel_lengths, mask_lengths)):
        # Generate random start point
        start = torch.randint(0, length - mask_length + 1, (1,)).item()
        end = start + mask_length

        partial_mask[i, :, start:end] = 0

    # Combine the masks
    loss_mask = original_mask.bool() & partial_mask.bool()

    return original_mask, loss_mask


def temporal_avg_pooling(x, mask):
    len_ = mask.sum(dim=2)
    x = torch.sum(x * mask, dim=2)
    out = torch.div(x, len_)
    return out


def mask_from_start_end_indices(seq_len, start, end):
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    return einx.greater_equal("n, b -> b n", seq, start) & einx.less(
        "n, b -> b n", seq, end
    )


def mask_from_frac_lengths(seq_len, frac_lengths, from_start=True):
    if from_start:
        lengths = (frac_lengths * seq_len).long()
        start = torch.zeros_like(lengths)
        end = lengths
    else:
        lengths = (frac_lengths * seq_len).long()
        max_start = seq_len - lengths
        rand = torch.rand_like(frac_lengths)
        start = (max_start * rand).long().clamp(min=0)
        end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def rand_span_mask(x, x_lengths, frac_lengths=(0.2, 0.4), from_start=True):
    batch = x.size(0)
    frac_lengths = (
        torch.zeros((batch,), device=x.device).float().uniform_(*frac_lengths)
    )
    rand_span_mask = mask_from_frac_lengths(
        x_lengths, frac_lengths, from_start=from_start
    ).unsqueeze(1)

    return rand_span_mask
