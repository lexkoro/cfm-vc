import logging
import multiprocessing
import os
import time
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader, get_weighted_sampler
from models import SynthesizerTrn

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()


class ModelEmaV2(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        if hasattr(model, "module"):
            self.model_state_dict = deepcopy(model.module.state_dict())
        else:
            self.model_state_dict = deepcopy(model.state_dict())
        self.decay = decay
        self.device = device  # perform ema on different device from model if set

    def _update(self, model, update_fn):
        model_values = (
            model.module.state_dict().values()
            if hasattr(model, "module")
            else model.state_dict().values()
        )
        with torch.no_grad():
            for ema_v, model_v in zip(self.model_state_dict.values(), model_values):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.model_state_dict


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = hps.train.port

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # for pytorch on win, backend use gloo
    dist.init_process_group(
        backend="gloo" if os.name == "nt" else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem  # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    train_dataset = TextAudioSpeakerLoader(
        hps.data.training_files, hps, all_in_mem=all_in_mem
    )
    num_workers = 8 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0

    sampler = get_weighted_sampler(train_dataset.audiopaths)

    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        batch_size=hps.train.batch_size,
        collate_fn=collate_fn,
    )
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(
            hps.data.validation_files, hps, all_in_mem=all_in_mem
        )
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=1,
            shuffle=False,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    net_g = SynthesizerTrn(
        hps.data.n_mel_channels,
        **hps.model,
    ).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    skip_optimizer = True
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_default_*.pth"),
            net_g,
            optim_g,
            skip_optimizer,
        )

        epoch_str = max(epoch_str, 1)
        name = utils.latest_checkpoint_path(hps.model_dir, "G_default_*.pth")
        global_step = int(name[name.rfind("_") + 1 : name.rfind(".")]) + 1
        # global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        print("load old checkpoint failed...")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    net_g = DDP(net_g, device_ids=[rank])
    ema_model = ModelEmaV2(
        net_g, decay=0.9999
    )  # It's necessary that we put this after loading model.

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group["lr"] = hps.train.learning_rate / warmup_epoch * epoch
        # training
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                net_g,
                optim_g,
                ema_model,
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                net_g,
                optim_g,
                ema_model,
                scaler,
                [train_loader, None],
                None,
                None,
            )
        # update learning rate
        scheduler_g.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, ema_model, scaler, loaders, logger, writers
):
    net_g = nets
    optim_g = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    half_type = torch.bfloat16 if hps.train.half_type == "bf16" else torch.float16

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, lengths, uv, ppgs = items
        spec = spec.cuda(rank, non_blocking=True)
        y = y.cuda(rank, non_blocking=True)
        c = c.cuda(rank, non_blocking=True)
        f0 = f0.cuda(rank, non_blocking=True)
        uv = uv.cuda(rank, non_blocking=True)
        lengths = lengths.cuda(rank, non_blocking=True)
        ppgs = ppgs.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run, dtype=half_type):
            (prior_loss, diff_loss, f0_pred, lf0) = net_g(
                c,
                f0,
                uv,
                spec,
                ppgs=ppgs,
                c_lengths=lengths,
            )

        with autocast(enabled=False, dtype=half_type):
            f0_loss = F.smooth_l1_loss(f0_pred, lf0.detach())
            loss_gen_all = diff_loss + prior_loss + f0_loss  # + reversal_loss

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        ema_model.update(net_g)

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [diff_loss, prior_loss]
                reference_loss = 0
                for i in losses:
                    reference_loss += i
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info(
                    f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}"
                )

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "learning_rate": lr,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/diff": diff_loss,
                        # "loss/g/reversal": reversal_loss,
                        "loss/g/prior": prior_loss,
                        "loss/g/f0": f0_loss,
                        # "loss/g/energy": energy_loss,
                    }
                )

                image_dict = {
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        spec[0].data.cpu().numpy()
                    ),
                    "all/f0": utils.plot_data_to_numpy(
                        lf0[0, 0, :].cpu().numpy(),
                        f0_pred[0, 0, :].detach().cpu().numpy(),
                    ),
                    # "all/energy": utils.plot_data_to_numpy(
                    #     energy[0, 0, :].cpu().numpy(),
                    #     energy_pred[0, 0, :].detach().cpu().numpy(),
                    # ),
                }

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_default_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    ema_model,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_ema_{}.pth".format(global_step)),
                )
                keep_ckpts = getattr(hps.train, "keep_ckpts", 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(
                        path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts,
                        sort_by_time=True,
                    )

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, ".2f")
        logger.info(f"====> Epoch: {epoch}, cost {durtaion} s")
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, lengths, uv, ppgs = items
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            f0 = f0[:1].cuda(0)
            uv = uv[:1].cuda(0)
            ppgs = ppgs[:1].cuda(0)

            y_dec, _ = generator.module.infer(
                c, spec, f0, uv, ppgs=ppgs, n_timesteps=10
            )

        image_dict.update(
            {
                "gt/mel": utils.plot_spectrogram_to_numpy(spec[0].cpu().numpy()),
                "pred/y_dec": utils.plot_spectrogram_to_numpy(
                    y_dec[0][:, : lengths[0]].cpu().numpy()
                ),
            }
        )
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    main()
