from __future__ import annotations

from datetime import datetime
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dataloader.datamodule import SimpleDataModule
from dataloader.dataset import HAM10000Dataset
from models.vae import VAE, DiagonalGaussianDistribution
from models.vis_token_extractor import VisTokenExtractor
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform
import cv2
import subprocess
import shutil
import wandb
import torch
import open_clip
import torch.nn as nn
import numpy as np

from models.diffusion.diffusion_pipeline import DiffusionPipeline
from models.diffusion.unet import UNet
from models.diffusion.gaussian_scheduler import GaussianNoiseScheduler
from models.diffusion.label_embedder import LabelEmbedder
from models.diffusion.time_embedder import TimeEmbedding


DATA_PATH = "your/data/path/here"


"""
OpenCLIP ViT-H/14
"""

model, _, preprocess = open_clip.create_model_and_transforms(
        model_name='ViT-H-14',
        pretrained='laion2b_s32b_b79k',
        device='cuda',          
        jit=False,
        precision='fp32',      
)

vis_backbone = model.visual

vis_backbone.eval().requires_grad_(False)
for p in vis_backbone.parameters():
    p.requires_grad = False
    
GLOBAL_VIS_BACKBONE = vis_backbone



vis_extractor = VisTokenExtractor(
        backbone=GLOBAL_VIS_BACKBONE,
        layer_ids=[5,11,17,23,31],
        k=32,
        proj_dim=1024,
        device='cuda'
).eval()

"""
Data augmentation
"""
class FromPIL(BasicTransform):
    """Convert PIL Image to numpy array"""
    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img, **params):
        return np.array(img)

    def get_transform_init_args_names(self):
        return []

strong_aug = A.Compose([
    FromPIL(),
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.Compose([
            A.ElasticTransform(alpha=600, border_mode = cv2.BORDER_REFLECT, sigma=10, p=0.3),
            A.Affine(scale=(0.9, 1.1), border_mode = cv2.BORDER_REFLECT, rotate=(-5, 5), p=0.3)
        ]),
        A.GridDistortion(num_steps=3, border_mode = cv2.BORDER_REFLECT, distort_limit=[-0.5, 0.5], p=0.7),
    ], p=0.3),

    A.Rotate(limit=13, border_mode = cv2.BORDER_REFLECT,p=0.5),
    # A.CenterCrop(300, 300),
    A.Resize(256, 256),

    A.OneOf([
        A.GaussianBlur(blur_limit=(1,4), p=0.2),
        A.GaussNoise(std_range=(0.005, 0.02),
                     mean_range=(0.0, 0.0),
                     per_channel=True,
                     noise_scale_factor=1.0,
                     p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 8),  # 1~8개의 구멍
            hole_height_range=(5, 15),
            hole_width_range=(5, 15),
            fill='inpaint_telea',
            p=0.2),
    ], p=0.3),
    A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ToTensorV2(),
])


train_ds = HAM10000Dataset(
        path_root=DATA_PATH,
        split="train",
        val_ratio=0.1,
        random_seed=42,
        transform= strong_aug
    )


val_ds = HAM10000Dataset(
        path_root=DATA_PATH,
        split="val",
        image_crop=450,  
        image_resize=(256, 256),
        val_ratio=0.1,
        random_seed=42,
    )

dm = SimpleDataModule(
    ds_train=train_ds,
    ds_val=val_ds,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    persistent_workers=False,
    weights=train_ds.get_weights(),
    balanced_epoch=True,
    sampler_num_samples=7000
)

current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
path_run_dir = Path.cwd() / 'runs' / str(current_time)
path_run_dir.mkdir(parents=True, exist_ok=True)
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

cond_embedder = LabelEmbedder
cond_embedder_kwargs = {
    'emb_dim': 1024,
    'num_classes': 7 # class number
}

time_embedder = TimeEmbedding
time_embedder_kwargs ={
    'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
}

noise_scheduler = GaussianNoiseScheduler
noise_scheduler_kwargs = {
    'timesteps': 1000,
    'beta_start': 0.002, # 0.0001, 0.0015
    'beta_end': 0.02, # 0.01, 0.0195
    'schedule_strategy': 'scaled_linear'
}

noise_estimator = UNet
noise_estimator_kwargs = {
    'in_ch':8, #4ch
    'out_ch':8, #4ch
    'spatial_dims':2,
    'hid_chs':  [256, 256, 512, 1024],
    'kernel_sizes':[3, 3, 3, 3],
    'strides':     [1, 2, 2, 2],
    'time_embedder':time_embedder,
    'time_embedder_kwargs': time_embedder_kwargs,
    'cond_embedder':cond_embedder,
    'cond_embedder_kwargs': cond_embedder_kwargs,
    'deep_supervision': False,
    'use_res_block':True,
    'use_attention':'none',
    'use_vis_adapter': True # vis_adapter
    }

latent_embedder = VAE
latent_embedder_checkpoint = 'path/vae_ckpt/path/here'

pipeline = DiffusionPipeline(
    noise_estimator=noise_estimator,
    noise_estimator_kwargs=noise_estimator_kwargs,
    noise_scheduler=noise_scheduler,
    noise_scheduler_kwargs = noise_scheduler_kwargs,
    latent_embedder=latent_embedder,
    latent_embedder_checkpoint = latent_embedder_checkpoint,
    vis_extractor = vis_extractor,    # vis_extractor
    beta          = 0.02,
    estimator_objective='x_T',
    estimate_variance=False,
    use_self_conditioning=False,
    use_ema=False,
    classifier_free_guidance_dropout=0.1, 
    do_input_centering=False,
    clip_x0=False,
    sample_every_n_steps=55,
)

to_monitor = "val/loss"  
min_max = "min"
save_and_sample_every = 100

logger = WandbLogger(
        project="skin-lesion-diffusion-vis",
        name=f"diffusion_{current_time}",
        log_model=False,
    )

early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, 
        patience=30, 
        mode='min'
    )
checkpointing = ModelCheckpoint(
    dirpath=str(path_run_dir), 
    monitor=to_monitor,
    # every_n_train_steps=20,
    save_last=True,
    save_top_k=2,
    mode=min_max,
)

trainer = Trainer(
        accelerator=accelerator,
        devices=[0],
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        limit_val_batches=1.0,
        logger=logger,
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )

trainer.fit(pipeline, datamodule=dm)