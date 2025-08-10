import gc
import os
import time
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv3D, Conv3DTranspose, LeakyReLU, BatchNormalization,
    Concatenate, Layer, Flatten, Dense, AveragePooling3D,
    GlobalAveragePooling3D, ReLU
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)
np.random.seed(42)

def load_fmri_data(file_path):
    print(f"Loading fMRI data from {file_path}")
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine

    print(f"fMRI data shape: {data.shape}")
    return data, affine


def add_noise(data, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    noisy_data = data + noise
    return noisy_data


def preprocess_data(data, patch_size=(32, 32, 8)):
    data_min = data.min()
    data_max = data.max()
    data_norm = (data - data_min) / (data_max - data_min)

    x, y, z, t = data_norm.shape
    print(f"Preprocessing data of shape {data_norm.shape}")

    patch_size = (
        min(patch_size[0], x),
        min(patch_size[1], y),
        min(patch_size[2], z)
    )
    print(f"Using patch size: {patch_size}")

    patches = []
    noisy_patches = []

    time_points = np.linspace(0, t-1, min(t, 30), dtype=int)

    for time in time_points:
        step_x = max(1, patch_size[0] // 2)
        step_y = max(1, patch_size[1] // 2)
        step_z = max(1, patch_size[2] // 2)

        for i in range(0, x - patch_size[0] + 1, step_x):
            for j in range(0, y - patch_size[1] + 1, step_y):
                for k in range(0, z - patch_size[2] + 1, step_z):
                    patch = data_norm[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2], time]

                    if np.mean(patch) > 0.01:
                        patches.append(patch)
                        noisy_patch = add_noise(patch, noise_factor=0.15)
                        noisy_patches.append(noisy_patch)

                        if len(patches) >= 500:
                            break

            if len(patches) >= 500:
                break

        if len(patches) >= 500:
            break

    if len(patches) == 0:
        print("No patches met the criteria. Taking patches without filtering...")
        for time in time_points[:5]:
            for i in range(0, x - patch_size[0] + 1, patch_size[0]):
                for j in range(0, y - patch_size[1] + 1, patch_size[1]):
                    for k in range(0, z - patch_size[2] + 1, patch_size[2]):
                        patch = data_norm[i:i+patch_size[0], j:j+patch_size[1], k:k+patch_size[2], time]
                        patches.append(patch)
                        noisy_patch = add_noise(patch, noise_factor=0.15)
                        noisy_patches.append(noisy_patch)

                        if len(patches) >= 100:
                            break

    patches = np.array(patches,dtype=np.float32)[..., np.newaxis]
    noisy_patches = np.array(noisy_patches,dtype=np.float32)[..., np.newaxis]

    print(f"Created {len(patches)} patches of shape {patches[0].shape}")
    return patches, noisy_patches, (data_min, data_max)



def dense_block(x, growth_rate=32, num_layers=4, name_prefix="dense"):
    """DenseNet-style dense block with feature concatenation"""
    concat_feat = x
    
    for i in range(num_layers):
        # Bottleneck layer (1x1x1 conv)
        bn = BatchNormalization(name=f"{name_prefix}_bn_{i}")(concat_feat)
        relu = ReLU(name=f"{name_prefix}_relu_{i}")(bn)
        bottleneck = Conv3D(4 * growth_rate, (1, 1, 1), padding='same', 
                           name=f"{name_prefix}_bottleneck_{i}")(relu)
        
        # 3x3x3 conv
        bn2 = BatchNormalization(name=f"{name_prefix}_bn2_{i}")(bottleneck)
        relu2 = ReLU(name=f"{name_prefix}_relu2_{i}")(bn2)
        conv = Conv3D(growth_rate, (3, 3, 3), padding='same', 
                     name=f"{name_prefix}_conv_{i}")(relu2)
        
        # Concatenate new features
        concat_feat = Concatenate(name=f"{name_prefix}_concat_{i}")([concat_feat, conv])
    
    return concat_feat

def transition_layer(x, compression_factor=0.5, name_prefix="transition"):
    """Transition layer for downsampling"""
    num_filters = int(x.shape[-1] * compression_factor)
    
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = ReLU(name=f"{name_prefix}_relu")(x)
    x = Conv3D(num_filters, (1, 1, 1), padding='same', name=f"{name_prefix}_conv")(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), name=f"{name_prefix}_pool")(x)
    
    return x

def conv_block(x, filters, name_prefix="conv_block"):
    """Standard convolution block"""
    x = Conv3D(filters, (3, 3, 3), padding='same', name=f"{name_prefix}_conv")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = ReLU(name=f"{name_prefix}_relu")(x)
    return x

def build_generator(input_shape, growth_rate=16):
    """U-Net Generator with DenseBlocks"""
    inputs = Input(shape=input_shape, name='generator_input')
    
    # Calculate depth based on input size
    min_dim = min(input_shape[0], input_shape[1], input_shape[2])
    max_depth = 0
    temp_dim = min_dim
    while temp_dim > 4:
        temp_dim //= 2
        max_depth += 1
    
    print(f"Using max depth of {max_depth} for generator")
    
    # Encoder path with skip connections
    skips = []
    x = inputs
    
    # Initial convolution
    x = Conv3D(32, (3, 3, 3), padding='same', name='initial_conv')(x)
    x = BatchNormalization(name='initial_bn')(x)
    x = ReLU(name='initial_relu')(x)
    
    # Encoder blocks
    filter_counts = [32, 64, 128]
    for i in range(min(max_depth - 1, len(filter_counts))):
        # Dense block
        x = dense_block(x, growth_rate=growth_rate, num_layers=4, 
                       name_prefix=f"encoder_dense_{i}")
        skips.append(x)
        
        # Transition layer (downsample)
        if i < max_depth - 2:
            x = transition_layer(x, compression_factor=0.5, 
                               name_prefix=f"encoder_transition_{i}")
        else:
            # Last encoder layer - just downsample without compression
            x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), 
                               name=f"encoder_pool_{i}")(x)
    
    # Bottleneck with 2 dense blocks
    x = dense_block(x, growth_rate=growth_rate, num_layers=4, 
                   name_prefix="bottleneck_dense_1")
    x = dense_block(x, growth_rate=growth_rate, num_layers=4, 
                   name_prefix="bottleneck_dense_2")
    
    # Decoder path
    for i in range(min(max_depth - 1, len(filter_counts))):
        # Upsample
        target_filters = filter_counts[-(i+1)]
        x = Conv3DTranspose(target_filters, (3, 3, 3), strides=(2, 2, 2), 
                           padding='same', name=f"decoder_upsample_{i}")(x)
        x = BatchNormalization(name=f"decoder_bn_{i}")(x)
        x = ReLU(name=f"decoder_relu_{i}")(x)
        
        # Skip connection
        if i < len(skips):
            skip = skips[-(i+1)]
            x = Concatenate(name=f"decoder_concat_{i}")([x, skip])
        
        # Conv block
        x = conv_block(x, target_filters, name_prefix=f"decoder_block_{i}")
    
    # Output layer with linear activation
    output = Conv3D(1, (1, 1, 1), padding='same', activation='linear', 
                   name='output_conv')(x)
    
    return Model(inputs=inputs, outputs=output, name='DenseNet_Generator')

def build_critic(input_shape, growth_rate=16):
    """U-Net Style Discriminator with DenseBlocks"""
    inputs = Input(shape=input_shape, name='critic_input')
    
    # Calculate depth
    min_dim = min(input_shape[0], input_shape[1], input_shape[2])
    max_depth = 0
    temp_dim = min_dim
    while temp_dim > 2:
        temp_dim //= 2
        max_depth += 1
    
    print(f"Using max depth of {max_depth} for critic")
    
    # Encoder path
    skips = []
    x = inputs
    
    # Initial convolution
    x = Conv3D(32, (3, 3, 3), padding='same', name='critic_initial_conv')(x)
    # No batch norm on first layer of critic
    x = LeakyReLU(alpha=0.2, name='critic_initial_leaky')(x)
    
    # Encoder blocks
    filter_counts = [32, 64, 128]
    for i in range(min(max_depth - 1, len(filter_counts))):
        # Dense block
        x = dense_block(x, growth_rate=growth_rate, num_layers=3, 
                       name_prefix=f"critic_encoder_dense_{i}")
        skips.append(x)
        
        # Downsample
        x = transition_layer(x, compression_factor=0.5, 
                           name_prefix=f"critic_encoder_transition_{i}")
    
    # Bottleneck
    x = dense_block(x, growth_rate=growth_rate, num_layers=4, 
                   name_prefix="critic_bottleneck_dense")
    
    # Decoder path
    for i in range(min(max_depth - 1, len(filter_counts))):
        # Upsample
        target_filters = filter_counts[-(i+1)]
        x = Conv3DTranspose(target_filters, (3, 3, 3), strides=(2, 2, 2), 
                           padding='same', name=f"critic_decoder_upsample_{i}")(x)
        x = BatchNormalization(name=f"critic_decoder_bn_{i}")(x)
        x = LeakyReLU(alpha=0.2, name=f"critic_decoder_leaky_{i}")(x)
        
        # Skip connection
        if i < len(skips):
            skip = skips[-(i+1)]
            x = Concatenate(name=f"critic_decoder_concat_{i}")([x, skip])
        
        # Conv block
        x = Conv3D(target_filters, (3, 3, 3), padding='same', 
                  name=f"critic_decoder_conv_{i}")(x)
        x = BatchNormalization(name=f"critic_decoder_conv_bn_{i}")(x)
        x = LeakyReLU(alpha=0.2, name=f"critic_decoder_conv_leaky_{i}")(x)
    
    # Global Average Pooling instead of Flatten
    x = GlobalAveragePooling3D(name='critic_global_avg_pool')(x)
    
    # Output
    output = Dense(1, name='critic_output')(x)
    
    return Model(inputs=inputs, outputs=output, name='DenseNet_Critic')

class GradientPenaltyLayer(keras.layers.Layer):
    def __init__(self, critic, gp_weight=10.0, **kwargs):
        super(GradientPenaltyLayer, self).__init__(**kwargs)
        self.critic = critic
        self.gp_weight = tf.cast(gp_weight, tf.float32)
    
    def call(self, inputs):
        real_images, fake_images = inputs
        
        # Ensure consistent dtype - cast to float32 for gradient computation
        real_images = tf.cast(real_images, tf.float32)
        fake_images = tf.cast(fake_images, tf.float32)
        
        batch_size = tf.shape(real_images)[0]
        
        # Random interpolation - ensure alpha is same dtype
        alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
            # Ensure pred is float32
            pred = tf.cast(pred, tf.float32)
        
        gradients = tape.gradient(pred, interpolated)
        gradients_squared = tf.square(gradients)
        gradients_squared_sum = tf.reduce_sum(gradients_squared, axis=[1, 2, 3, 4])
        gradient_l2_norm = tf.sqrt(gradients_squared_sum + 1e-8)
        gradient_penalty = self.gp_weight * tf.square(gradient_l2_norm - 1.0)
        
        # Ensure output is float32
        return tf.cast(tf.reduce_mean(gradient_penalty), tf.float32)

# FIXED: Updated DenseNetWGAN with explicit float32 casting throughout
class DenseNetWGAN(keras.Model):
    def __init__(self, input_shape, critic_extra_steps=5, gp_weight=10.0, l1_weight=100.0):
        super(DenseNetWGAN, self).__init__()
        self.input_shape = input_shape
        self.generator = build_generator(input_shape)
        self.critic = build_critic(input_shape)
        self.critic_extra_steps = critic_extra_steps
        self.gp_weight = tf.cast(gp_weight, tf.float32)
        self.l1_weight = tf.cast(l1_weight, tf.float32)
        self.gp_layer = GradientPenaltyLayer(self.critic, gp_weight)
        
        # Optimizers with parameters from fMRI denoising research
        self.generator_optimizer = keras.optimizers.Adam(
            learning_rate=1e-4, beta_1=0.0, beta_2=0.9
        )
        self.critic_optimizer = keras.optimizers.Adam(
            learning_rate=1e-4, beta_1=0.0, beta_2=0.9
        )
        
        # Print model summaries
        print("Generator Architecture:")
        self.generator.summary()
        print("\nCritic Architecture:")
        self.critic.summary()
    
    def compile(self, **kwargs):
        super(DenseNetWGAN, self).compile(**kwargs)
    
    @tf.function
    def train_critic(self, noisy_images, real_images):
        # Ensure float32 for stable training
        noisy_images = tf.cast(noisy_images, tf.float32)
        real_images = tf.cast(real_images, tf.float32)
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noisy_images, training=True)
            fake_images = tf.cast(fake_images, tf.float32)  # Ensure consistency
            
            real_output = self.critic(real_images, training=True)
            fake_output = self.critic(fake_images, training=True)
            
            # Cast outputs to float32
            real_output = tf.cast(real_output, tf.float32)
            fake_output = tf.cast(fake_output, tf.float32)
            
            # Wasserstein loss - ensure float32
            critic_loss = tf.cast(tf.reduce_mean(fake_output) - tf.reduce_mean(real_output), tf.float32)
            
            # Gradient penalty - already returns float32
            gp = self.gp_layer([real_images, fake_images])
            
            # Ensure both are float32 before adding
            critic_loss = tf.cast(critic_loss, tf.float32)
            gp = tf.cast(gp, tf.float32)
            
            total_critic_loss = critic_loss + gp
        
        gradients = tape.gradient(total_critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
        
        return tf.cast(total_critic_loss, tf.float32)
    
    @tf.function
    def train_generator(self, noisy_images, real_images):
        # Ensure float32 for stable training
        noisy_images = tf.cast(noisy_images, tf.float32)
        real_images = tf.cast(real_images, tf.float32)
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(noisy_images, training=True)
            fake_images = tf.cast(fake_images, tf.float32)  # Ensure consistency
            
            fake_output = self.critic(fake_images, training=True)
            fake_output = tf.cast(fake_output, tf.float32)
            
            # Wasserstein loss - ensure float32
            wasserstein_loss = tf.cast(-tf.reduce_mean(fake_output), tf.float32)
            
            # L1 loss for fMRI denoising - ensure float32
            l1_loss = tf.cast(tf.reduce_mean(tf.abs(fake_images - real_images)) * self.l1_weight, tf.float32)
            
            # Total generator loss
            total_gen_loss = tf.cast(wasserstein_loss + l1_loss, tf.float32)
        
        gradients = tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return (tf.cast(total_gen_loss, tf.float32), 
                tf.cast(wasserstein_loss, tf.float32), 
                tf.cast(l1_loss, tf.float32))
    
    def train_step(self, data):
        noisy_images, real_images = data
        
        # Ensure input data is float32
        noisy_images = tf.cast(noisy_images, tf.float32)
        real_images = tf.cast(real_images, tf.float32)
        
        # Train critic multiple times
        for i in range(self.critic_extra_steps):
            c_loss = self.train_critic(noisy_images, real_images)
        
        # Train generator once
        g_loss, g_wasserstein, g_l1 = self.train_generator(noisy_images, real_images)
        
        return {
            'critic_loss': tf.cast(c_loss, tf.float32),
            'gen_loss': tf.cast(g_loss, tf.float32),
            'wasserstein_loss': tf.cast(g_wasserstein, tf.float32),
            'l1_loss': tf.cast(g_l1, tf.float32)
        }

# Training configuration
class TrainingConfig:
    def __init__(self):
        self.patch_size = (32, 32, 8)
        self.batch_size = 4  # Small batch size for memory efficiency
        self.epochs = 100
        self.save_interval = 10
        self.checkpoint_dir = "/kaggle/working/checkpoints"
        self.results_dir = "/kaggle/working/results"
        self.data_path = "/kaggle/working/fmri_dataset_chunk_1.npy"
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

def create_tf_dataset(noisy_patches, clean_patches, batch_size=4):
    """Create TensorFlow dataset for efficient training"""
    # Ensure data is float32
    noisy_patches = noisy_patches.astype(np.float32)
    clean_patches = clean_patches.astype(np.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices((noisy_patches, clean_patches))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def plot_training_progress(history, save_path):
    """Plot and save training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator loss
    axes[0, 0].plot(history['gen_loss'])
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # Critic loss
    axes[0, 1].plot(history['critic_loss'])
    axes[0, 1].set_title('Critic Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    
    # Wasserstein loss
    axes[1, 0].plot(history['wasserstein_loss'])
    axes[1, 0].set_title('Wasserstein Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    
    # L1 loss
    axes[1, 1].plot(history['l1_loss'])
    axes[1, 1].set_title('L1 Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_sample_results(model, test_data, epoch, save_dir):
    """Save sample denoising results"""
    noisy_sample, clean_sample = test_data
    
    # Take first sample from batch
    noisy_input = noisy_sample[:1]
    clean_target = clean_sample[:1]
    
    # Generate denoised output
    denoised_output = model.generator(noisy_input, training=False)
    
    # Convert to numpy
    noisy_np = noisy_input[0, ..., 0].numpy()
    clean_np = clean_target[0, ..., 0].numpy()
    denoised_np = denoised_output[0, ..., 0].numpy()
    
    # Save middle slice for visualization
    middle_slice = noisy_np.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noisy_np[:, :, middle_slice], cmap='gray')
    axes[0].set_title('Noisy Input')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_np[:, :, middle_slice], cmap='gray')
    axes[1].set_title('Denoised Output')
    axes[1].axis('off')
    
    axes[2].imshow(clean_np[:, :, middle_slice], cmap='gray')
    axes[2].set_title('Clean Target')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sample_epoch_{epoch:03d}.png")
    plt.close()

def train_densenet_wgan():
    """Main training function"""
    print("=== Starting DenseNet-WGAN Training ===")
    
    # Initialize configuration
    config = TrainingConfig()
    
    # Load and preprocess data
    print("Loading fMRI data...")
    try:
        fmri_data = np.load(config.data_path, allow_pickle=True)
        print(f"Loaded data shape: {fmri_data.shape}")
    except FileNotFoundError:
        print(f"Data file not found: {config.data_path}")
        print("Please ensure the data is downloaded and saved correctly.")
        return
    
    # Process all fMRI volumes
    all_patches = []
    all_noisy_patches = []
    
    for i, data in enumerate(fmri_data):
        print(f"\nProcessing volume {i+1}/{len(fmri_data)}")
        patches, noisy_patches, _ = preprocess_data(data, patch_size=config.patch_size)
        all_patches.extend(patches)
        all_noisy_patches.extend(noisy_patches)
        
        # Memory management
        if len(all_patches) > 2000:  # Limit total patches for memory
            break
    
    all_patches = np.array(all_patches, dtype=np.float32)
    all_noisy_patches = np.array(all_noisy_patches, dtype=np.float32)
    
    print(f"\nTotal patches created: {len(all_patches)}")
    print(f"Patch shape: {all_patches[0].shape}")
    
    # Split data
    train_noisy, test_noisy, train_clean, test_clean = train_test_split(
        all_noisy_patches, all_patches, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_noisy)}")
    print(f"Test samples: {len(test_noisy)}")
    
    # Create datasets
    train_dataset = create_tf_dataset(train_noisy, train_clean, config.batch_size)
    test_dataset = create_tf_dataset(test_noisy, test_clean, config.batch_size)
    
    # Get a test sample for visualization
    test_sample = next(iter(test_dataset))
    
    # Create model
    input_shape = all_patches[0].shape
    print(f"\nCreating model with input shape: {input_shape}")
    
    model = DenseNetWGAN(
        input_shape=input_shape,
        critic_extra_steps=5,
        gp_weight=10.0,
        l1_weight=100.0
    )
    
    model.compile()
    
    # Training history
    history = {
        'gen_loss': [],
        'critic_loss': [],
        'wasserstein_loss': [],
        'l1_loss': []
    }
    
    # Training loop
    print("\n=== Starting Training ===")
    start_time = time.time()
    
    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch + 1}/{config.epochs}')
        
        # Training metrics
        epoch_gen_loss = []
        epoch_critic_loss = []
        epoch_wasserstein_loss = []
        epoch_l1_loss = []
        
        # Train on batches
        for batch_idx, (noisy_batch, clean_batch) in enumerate(train_dataset):
            # Train step
            metrics = model.train_step((noisy_batch, clean_batch))
            
            # Collect metrics
            epoch_gen_loss.append(float(metrics['gen_loss']))
            epoch_critic_loss.append(float(metrics['critic_loss']))
            epoch_wasserstein_loss.append(float(metrics['wasserstein_loss']))
            epoch_l1_loss.append(float(metrics['l1_loss']))
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}: Gen Loss: {metrics["gen_loss"]:.4f}, '
                      f'Critic Loss: {metrics["critic_loss"]:.4f}, '
                      f'L1 Loss: {metrics["l1_loss"]:.4f}')
        
        # Average metrics for epoch
        avg_gen_loss = np.mean(epoch_gen_loss)
        avg_critic_loss = np.mean(epoch_critic_loss)
        avg_wasserstein_loss = np.mean(epoch_wasserstein_loss)
        avg_l1_loss = np.mean(epoch_l1_loss)
        
        # Store history
        history['gen_loss'].append(avg_gen_loss)
        history['critic_loss'].append(avg_critic_loss)
        history['wasserstein_loss'].append(avg_wasserstein_loss)
        history['l1_loss'].append(avg_l1_loss)
        
        print(f'Epoch {epoch + 1} Summary:')
        print(f'  Gen Loss: {avg_gen_loss:.4f}')
        print(f'  Critic Loss: {avg_critic_loss:.4f}')
        print(f'  Wasserstein Loss: {avg_wasserstein_loss:.4f}')
        print(f'  L1 Loss: {avg_l1_loss:.4f}')
        
        # Save checkpoints and samples
        if (epoch + 1) % config.save_interval == 0:
            print(f'Saving checkpoint at epoch {epoch + 1}...')
            
            # Save model weights
            generator_path = f"{config.checkpoint_dir}/generator_epoch_{epoch + 1:03d}.h5"
            critic_path = f"{config.checkpoint_dir}/critic_epoch_{epoch + 1:03d}.h5"
            
            model.generator.save_weights(generator_path)
            model.critic.save_weights(critic_path)
            
            # Save sample results
            save_sample_results(model, test_sample, epoch + 1, config.results_dir)
            
            # Plot training progress
            plot_path = f"{config.results_dir}/training_progress_epoch_{epoch + 1:03d}.png"
            plot_training_progress(history, plot_path)
            
            print(f'Checkpoint saved!')
        
        # Memory cleanup
        if epoch % 5 == 0:
            gc.collect()
    
    # Final save
    print("\n=== Training Complete ===")
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time/3600:.2f} hours")
    
    # Save final model
    final_generator_path = f"{config.checkpoint_dir}/final_generator.h5"
    final_critic_path = f"{config.checkpoint_dir}/final_critic.h5"
    
    model.generator.save_weights(final_generator_path)
    model.critic.save_weights(final_critic_path)
    
    # Save final training plot
    final_plot_path = f"{config.results_dir}/final_training_progress.png"
    plot_training_progress(history, final_plot_path)
    
    # Save training history
    np.save(f"{config.results_dir}/training_history.npy", history)
    
    print(f"Final model saved to: {final_generator_path}")
    print(f"Training history saved to: {config.results_dir}/training_history.npy")
    
    return model, history

# FIXED: Memory optimization with explicit float32 policy
def setup_memory_optimization():
    """Setup memory optimization for training"""
    # Explicitly set float32 policy
    tf.keras.mixed_precision.set_global_policy('float32')
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")
    
    print("Memory growth enabled - float32 policy set explicitly")

if __name__ == "__main__":
    # Setup memory optimization with explicit float32
    setup_memory_optimization()
    
    # Start training
    try:
        model, history = train_densenet_wgan()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
