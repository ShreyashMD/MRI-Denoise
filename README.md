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

## Google Colab

You can also train the model on Google Colab. After launching a notebook, install
the dependencies and download the dataset using the `openneuro` CLI:

```python
!pip install openneuro-cli tensorflow matplotlib nibabel
!openneuro download --dataset ds005239 --version 1.0.1 --destination data
```

Then run the cells in `notebooks/fmri_denoise_colab.ipynb` to start training. The
training pipeline uses `tf.data` caching and prefetching for faster throughput on
Colab GPUs without compromising model accuracy.

