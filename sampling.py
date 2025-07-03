import torch
import open_clip
from models.vis_token_extractor import VisTokenExtractor
from pathlib import Path
from torchvision import utils
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

def rgb2gray(img):
    # img [B, C, H, W]
    return  ((0.3 * img[:,0]) + (0.59 * img[:,1]) + (0.11 * img[:,2]))[:, None]
    
    
def norm(x):
    return (x - x.min()) / (x.max() - x.min())

path_out = Path.cwd()/'results/ham10k_ml/samples'
path_out.mkdir(parents=True, exist_ok=True)

torch.manual_seed(0)
device = torch.device('cuda')



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


from models.diffusion.diffusion_pipeline import DiffusionPipeline
pipeline = DiffusionPipeline.load_from_checkpoint('path/diffusion_ckpt/path/here', vis_extractor=vis_extractor)
pipeline.to(device)

class_names = {
    0: 'Actinic Keratoses',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanocytic Nevi',
    5: 'Melanoma',
    6: 'Vascular Lesion'
}

experiment_config = {
    'classes': list(range(7)),
    'use_ddim': [True, False],
    'steps': {
        True: [50, 100, 150],   # DDIM
        False: [500, 750, 1000] # DDPM
    },
    'guidance_scales': [1,3,5,7]
}


def norm(x):
    return (x - x.min()) / (x.max() - x.min())

results = {}
timings = {}
quality_metrics = {}

for selected_class in range(7):  # 0~6

    
    total_rows_needed = 0
    for method in ['DDIM', 'DDPM']:
        total_rows_needed += len(experiment_config['steps'][method == 'DDIM'])

    print(f"\n=== {class_names[selected_class]} ===")
    print(f"필요한 행 수: {total_rows_needed}")

    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(total_rows_needed, 4, figure=fig, hspace=0.4, wspace=0.2)

    row = 0
    for method in ['DDIM', 'DDPM']:
        steps_list = experiment_config['steps'][method == 'DDIM']

        for steps in steps_list:
            for col, cfg in enumerate(experiment_config['guidance_scales']):
                ax = fig.add_subplot(gs[row, col])

                if (method in results[selected_class] and
                    steps in results[selected_class][method] and
                    cfg in results[selected_class][method][steps]):

                    samples = results[selected_class][method][steps][cfg]
                    img = norm(samples[0].cpu())

                    if img.shape[0] == 3:
                        img = img.permute(1, 2, 0)
                    else:
                        img = img.squeeze()

                    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                    time_val = timings[selected_class][method][steps][cfg]
                    ax.set_title(f'{method}\nSteps: {steps}\nCFG: {cfg}\n{time_val:.1f}s',
                                fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{method}\nSteps: {steps}\nCFG: {cfg}', fontsize=8)

                ax.axis('off')

            row += 1

    plt.suptitle(f'{class_names[selected_class]} - All Settings Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()



# save sample
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_samples = 1000  
batch_size = 200  

# ================================================
class_id = 1  # Class
# ================================================

# 저장 경로 설정
output_dir = Path(f'/path/output/dir/{class_id}')
output_dir.mkdir(parents=True, exist_ok=True)


use_ddim = True  # DDPM, DDIM   
guidance_scale = 3.0
steps = 750


print(f"Generating {num_samples} images for class {class_id}...")
print(f"Output directory: {output_dir}")

for batch_idx in range(0, num_samples, batch_size):
    current_batch_size = min(batch_size, num_samples - batch_idx)

    condition = torch.tensor([class_id] * current_batch_size, device=device)

    samples = pipeline.sample(
        num_samples=current_batch_size,
        img_size=(8, 32, 32),  
        steps=steps,
        use_ddim=use_ddim,
        guidance_scale=guidance_scale
    ).detach()

    for i, sample in enumerate(samples):
        img_idx = batch_idx + i
        img = (sample - sample.min()) / (sample.max() - sample.min())
        utils.save_image(img, output_dir / f'img_{img_idx:04d}.png')

    print(f"Progress: {batch_idx + current_batch_size}/{num_samples} images generated")

print(f"Completed! {num_samples} images saved in {output_dir}")