# Diffusion-based Skin Disease Data Augmentation

All experiments were conducted using:
- A single NVIDIA A100 GPU (40GB)
- PyTorch 2.0+
- Python 3.10


### 1. Prepare DATA
Download the [HAM10000 dataset](https://doi.org/10.1038/sdata.2018.161).
After extracting the dataset, organize the images into subdirectories according to their class labels as follows:
/path/to/HAM10000/
├── akiec/
├── bcc/
├── bkl/
├── df/
├── mel/
├── nv/
└── vasc/
Each folder should contain images corresponding to that specific class.
---