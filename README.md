# MRI Denoise

This repository provides code for training a 3D U-WGAN model to denoise fMRI volumes.

## Dataset

The code expects data from the [OpenNeuro ds005239 dataset](https://openneuro.org/datasets/ds005239/versions/1.0.1). Use the `openneuro` CLI to download it:

```bash
openneuro download --dataset ds005239 --version 1.0.1 --destination data
```

## Training

Install the dependencies and run training with a path to a NIfTI file from the dataset:

```bash
pip install -r requirements.txt
python -m mri_denoise.train data/sub-01/func/sub-01_task-something_bold.nii.gz
```

The trained generator will be saved as `fmri_denoiser_generator.h5`.

