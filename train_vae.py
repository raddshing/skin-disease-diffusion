from __future__ import annotations
from datetime import datetime
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import torch
from dataloader.datamodule import SimpleDataModule
from dataloader.dataset import HAM10000Dataset
from models import VAE, DiagonalGaussianDistribution
import subprocess
import shutil

torch.set_float32_matmul_precision('high' if IS_COLAB else 'medium')
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
path_run_dir = Path.cwd() / "runs" / current_time
path_run_dir.mkdir(parents=True, exist_ok=True)

DATA_PATH = "your/data/path/here"

model = VAE(
        in_channels=3,
        out_channels=3,
        emb_channels=8,  #4ch or 8ch
        spatial_dims=2,
        hid_chs=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        deep_supervision=1,
        use_attention="none",
        loss=torch.nn.MSELoss,          # reconstruction loss
        embedding_loss_weight=1e-6, # KL term weight
        sample_every_n_steps=500,

    )

logger = WandbLogger(
        project="skin-lesion-vae",
        name=f"vae_{current_time}",
        log_model=False,
    )

to_monitor = "val/L1"  
mode = "min"
save_and_sample_every = 100

early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.01,
        patience=30,
        mode=mode,
    )

checkpointing = ModelCheckpoint(
            dirpath=str(path_run_dir),
            monitor=to_monitor,
            # every_n_train_steps=280,
            save_last=True,
            save_top_k=2,
            mode=mode,
        )

train_ds = HAM10000Dataset(
        path_root=DATA_PATH,
        split="train",
        image_crop=450,  # 고정 크기로 center crop
        image_resize=(256, 256),  # 정사각형으로 리사이즈
        val_ratio=0.1,
        random_seed=42,
        augment_horizontal_flip=True,
        augment_vertical_flip=True,
    )

val_ds = HAM10000Dataset(
        path_root=DATA_PATH,
        split="val",
        image_crop=450,  # 고정 크기로 center crop
        image_resize=(256, 256),  # 정사각형으로 리사이즈
        val_ratio=0.1,
        random_seed=42,
    )

dm = SimpleDataModule(
    ds_train=train_ds,
    ds_val=val_ds,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,  # DDP spawn 최적화
    weights=train_ds.get_weights(),
)

trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        precision=32,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        # val_check_interval=1.0,
        log_every_n_steps=20,
        limit_val_batches=1.0,
        max_epochs=1001,
        min_epochs=50,
        # num_sanity_val_steps=2,
        logger=logger,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1
    )

trainer.fit(model, datamodule=dm)
