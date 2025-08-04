# MRI Denoise

This repository provides code for training a 3D U-WGAN model to denoise fMRI volumes.

## Dataset

You can obtain training data in two ways:

1. **Full dataset** – Download the entire [OpenNeuro ds005239 dataset](https://openneuro.org/datasets/ds005239/versions/1.0.1) using the `openneuro` CLI:


## Training

Install the dependencies and run training with a path to a NIfTI file from the dataset:

```bash
pip install -r requirements.txt
python -m mri_denoise.train data/sub-01/func/sub-01_task-something_bold.nii.gz
```

The trained generator will be saved as `fmri_denoiser_generator.h5`.


Here’s a focused explanation of the core model architectures and their parameter counts.


## 1. **Generator (DenseNet_Generator)**
- **Purpose:** Converts noisy fMRI patches into denoised patches.
- **Architecture:**  
  - **Input shape:** e.g., (32, 32, 8, 1)
  - **Initial Conv3D + BatchNorm + ReLU**
  - **Dense Blocks:** Several “dense blocks” repeatedly concatenate their input with new convolutional outputs, capturing complex features and enabling efficient gradient flow.
  - **No explicit downsampling layers** for the smallest patch shape (since max depth=1 or 2).
  - **Bottleneck:** Stacked dense blocks.
  - **1×1×1 Output Conv3D** (final image, linear activation).

- **Parameters Examples:**  
    - **Small Patch Model (32×32×8×1):**
        - **Total params:** 272,929
        - **Trainable params:** 270,433
        - **Non-trainable params:** 2,496

    - **Large Patch Model (64×64×64×1):**
        - **Total params:** 3,132,489
        - **Trainable params:** 3,124,073
        - **Non-trainable params:** 8,416

## 2. **Critic (DenseNet_Critic)**
- **Purpose:** Distinguishes between real (clean) and denoised (generated) fMRI patches, used for WGAN loss.
- **Architecture:**  
  - **Input shape:** matches Generator output.
  - **Initial Conv3D + LeakyReLU**
  - **Dense Blocks:** (slightly fewer layers than generator).
  - **Transition Layers:** Downsample features with Conv3D+AveragePooling3D.
  - **Again, short for small patches (max depth=2); deeper for large patches.**
  - **GlobalAveragePooling3D → Dense(1) Output.**

- **Parameters Examples:**  
    - **Small Patch Model (32×32×8×1):**
        - **Total params:** 164,791
        - **Trainable params:** 163,743
        - **Non-trainable params:** 1,048

    - **Large Patch Model (64×64×64×1):**
        - **Total params:** 2,208,315
        - **Trainable params:** 2,203,691
        - **Non-trainable params:** 4,624

### **Notes on Parameter Counts**
- **Parameter count scales exponentially** with input patch size:  
    - Going from **(32,32,8)** to **(64,64,64)** increases both depth and width of the network, hence a much larger number of weights.

- **Dense blocks** quickly add parameters as more concatenation channels grow.

- **BatchNormalization** layers are present after nearly every convolution.

### **Why different numbers?**
- **Model Depth**: The allowed depth (and thus the number of dense/transition blocks) is automatically set based on the patch’s size.
- For **(32,32,8,1)**, only bottleneck dense blocks are used (max depth=1).  
- For **(64,64,64,1)**, multiple downsampling/up paths are possible (max depth=4~5).
- **The bigger the patch or volume, the deeper and wider the model.**

## **Summary Table**

| Input shape          | Generator Total Params | Critic Total Params |
|----------------------|-----------------------|---------------------|
| (32, 32, 8, 1)       | 272,929               | 164,791             |
| (64, 64, 64, 1)      | 3,132,489             | 2,208,315           |

### **Key Takeaways**

- **Param counts** are determined by input volume/patch size and dense block structure.
- **For bigger fMRI volumes, the number of trainable weights increases rapidly!**
- This architecture is scalable—the model adapts automatically based on your training patch size.




