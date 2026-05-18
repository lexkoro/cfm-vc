import logging
import multiprocessing
import os
from typing import Any, Optional

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import utils
import wandb
from data_utils import UnitMelLoader
from models.models import GameVC
from modules.ema_callback import EMA, EMAModelCheckpoint
from modules.schedulers import WarmupLR

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def _get(config, key, default=None):
    if config is None:
        return default
    value = getattr(config, key, default)
    return default if value is None else value


def resolve_checkpoint(hps) -> Optional[str]:
    ckpt_path = _get(hps.train, "checkpoint_restore_path", None)
    if ckpt_path is None:
        return None

    if not str(ckpt_path).endswith(".ckpt"):
        logging.warning(
            "Ignoring checkpoint_restore_path=%s because only Lightning .ckpt resume "
            "is supported in the rewritten trainer.",
            ckpt_path,
        )
        return None
    return ckpt_path


class VoiceConversionDataModule(pl.LightningDataModule):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.train_dataset = None
        self.eval_dataset = None

    def prepare_data(self):
        return

    def setup(self, stage=None):
        if stage in (None, "fit"):
            if self.train_dataset is None:
                self.train_dataset = UnitMelLoader("train", self.hps.data, verbose=True)
            if self.eval_dataset is None:
                self.eval_dataset = UnitMelLoader("val", self.hps.data, verbose=True)

    def train_dataloader(self):
        num_workers = min(
            _get(self.hps.train, "num_workers", 8), multiprocessing.cpu_count()
        )
        dataloader_kwargs = {}
        if num_workers > 0:
            dataloader_kwargs = {
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
            }

        return DataLoader(
            self.train_dataset,
            batch_size=self.hps.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
            **dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=2,
            collate_fn=self.eval_dataset.collate_fn,
        )


class VoiceConversionModule(pl.LightningModule):
    def __init__(self, hps):
        super().__init__()
        self.save_hyperparameters(ignore=["hps"])
        self.automatic_optimization = True
        self.hps = hps
        self.schedule_params = hps.train.schedule_params

        # Model
        self.model = GameVC(
            spec_channels=hps.data.n_mel_channels,
            unit_vocab_size=hps.data.unit_vocab_size,
            encoder=hps.model.encoder,
            decoder=hps.model.decoder,
        )
        self.validation_step_outputs = []
        self.train_epoch_preview = None

    def training_step(self, batch, batch_idx):
        units, unit_lengths, mel, mel_lengths = batch

        diff_loss, _, mel_generated = self.model(units, unit_lengths, mel, mel_lengths)

        # Log a single qualitative training preview per epoch.
        if batch_idx == 0 and self.trainer is not None and self.trainer.is_global_zero:
            mel_sample = mel[0].detach().float().cpu().numpy()
            generated_sample = mel_generated[0].detach().float().cpu().numpy()
            condition_sample = mel_sample - generated_sample

            self.train_epoch_preview = {
                "condition_img": utils.plot_spectrogram_to_numpy(condition_sample),
                "generated_img": utils.plot_spectrogram_to_numpy(generated_sample),
                "target_img": utils.plot_spectrogram_to_numpy(mel_sample),
            }

        self.log(
            "train/loss",
            diff_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=units.size(0),
        )

        return diff_loss

    def on_train_epoch_end(self):
        if self.global_rank == 0 and self.train_epoch_preview:
            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "train/condition_mel": wandb.Image(
                            self.train_epoch_preview["condition_img"]
                        ),
                        "train/generated_mel": wandb.Image(
                            self.train_epoch_preview["generated_img"]
                        ),
                        "train/target_mel": wandb.Image(
                            self.train_epoch_preview["target_img"]
                        ),
                    },
                    step=self.global_step,
                )

        self.train_epoch_preview = None

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Only generate for first few batches to save time
        if batch_idx >= 4:
            return

        units, unit_lengths, mel, _ = batch

        if units.size(0) < 2:
            return

        target_idx = 0
        source_idx = 1

        y_dec = self.model.infer(
            source_units=units[source_idx : source_idx + 1],
            target_units=units[target_idx : target_idx + 1],
            target_mel=mel[target_idx : target_idx + 1],
            source_lengths=unit_lengths[source_idx : source_idx + 1],
            target_lengths=unit_lengths[target_idx : target_idx + 1],
            n_timesteps=self.hps.inference.n_timesteps,
            temperature=self.hps.inference.temperature,
            guidance_scale=self.hps.inference.guidance_scale,
            solver=self.hps.inference.solver,
        )

        target_prompt = mel[target_idx].detach().float().cpu().numpy()
        source_content = mel[source_idx].detach().float().cpu().numpy()
        pred_source = y_dec[0].detach().float().cpu().numpy()

        self.validation_step_outputs.append(
            {
                "batch_idx": batch_idx,
                "target_img": utils.plot_spectrogram_to_numpy(target_prompt),
                "source_img": utils.plot_spectrogram_to_numpy(source_content),
                "pred_img": utils.plot_spectrogram_to_numpy(pred_source),
            }
        )

    def on_validation_epoch_end(self):
        if self.global_rank == 0 and self.validation_step_outputs:
            log_dict = {}
            for output in self.validation_step_outputs:
                batch_idx = output["batch_idx"]
                log_dict[f"eval/target_prompt_mel_{batch_idx}"] = wandb.Image(
                    output["target_img"]
                )
                log_dict[f"eval/source_content_mel_{batch_idx}"] = wandb.Image(
                    output["source_img"]
                )
                log_dict[f"eval/pred_source_mel_{batch_idx}"] = wandb.Image(
                    output["pred_img"]
                )

            if log_dict and isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(log_dict, step=self.global_step)

        self.validation_step_outputs.clear()

    def configure_optimizers(self) -> Any:
        trainable_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )

        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=self.schedule_params.max_lr
        )
        scheduler = WarmupLR(optimizer, **self.schedule_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."

    hps = utils.get_hparams()
    torch.manual_seed(hps.train.seed)

    datamodule = VoiceConversionDataModule(hps)

    logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT", "cfm-vc"),
        name=hps.model_dir,
    )

    model = VoiceConversionModule(hps)

    if hps.train.fp16_run:
        precision = "bf16-mixed" if hps.train.half_type == "bf16" else "16-mixed"
    else:
        precision = "32-true"

    trainer = pl.Trainer(
        default_root_dir=hps.model_dir,
        benchmark=False,
        accelerator="gpu",
        devices="auto",
        precision=precision,
        max_epochs=hps.train.epochs,
        gradient_clip_val=_get(hps.train, "gradient_clip_val", 1.0),
        accumulate_grad_batches=max(
            int(_get(hps.train, "accumulate_grad_batches", 1)), 1
        ),
        use_distributed_sampler=True,
        # num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=hps.train.log_interval,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            ModelSummary(max_depth=3),
            EMA(
                decay=0.9999,
                apply_ema_every_n_steps=1,
                start_step=0,
                save_ema_weights_in_callback_state=True,
                evaluate_ema_weights_instead=False,
            ),
            EMAModelCheckpoint(
                monitor="train/loss",
                save_top_k=2,
                save_last=True,
            ),
        ],
        logger=logger,
    )

    ckpt_path = resolve_checkpoint(hps)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
