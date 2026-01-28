import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from tensorflow.keras import layers

# ----------------------------
# Load and prepare the dataset
# ----------------------------
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

# ----------------------------
# Generator Model
# ----------------------------
def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                     padding="same", use_bias=False, activation="tanh"))
    assert model.output_shape == (None, 28, 28, 1)

    return model


generator = make_generator_model()

# Test generator output
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap="gray")
plt.axis("off")
plt.show()

# ----------------------------
# Discriminator Model
# ----------------------------
def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print("Discriminator output:", decision)

# ----------------------------
# Loss functions
# ----------------------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# ----------------------------
# Optimizers
# ----------------------------
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ----------------------------
# Checkpoints
# ----------------------------
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

# ----------------------------
# Training setup
# ----------------------------
EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, 0] + 1) / 2.0, cmap="gray")
        plt.axis("off")

    plt.savefig(f"image_at_epoch_{epoch:04d}.png")
    plt.close(fig)


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Time for epoch {epoch + 1}: {time.time() - start:.2f} sec")

    generate_and_save_images(generator, epochs, seed)


# ----------------------------
# Run training
# ----------------------------
train(train_dataset, EPOCHS)

# Restore latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# ----------------------------
# Create GIF
# ----------------------------
anim_file = "dcgan.gif"

with imageio.get_writer(anim_file, mode="I") as writer:
    filenames = sorted(glob.glob("image_at_epoch_*.png"))
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"Saved animation to {anim_file}")



#IF output it too slow....delete all above code and uncomment this code
# import tensorflow as tf
# import glob
# import imageio
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import time
# from tensorflow.keras import layers
#
# # ----------------------------
# # Speed optimizations
# # ----------------------------
# tf.config.optimizer.set_jit(True)  # Enable XLA
#
# # ----------------------------
# # Load and prepare the dataset
# # ----------------------------
# (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
#
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
# train_images = (train_images - 127.5) / 127.5
#
# BUFFER_SIZE = 10000
# BATCH_SIZE = 64   # Better for CPU
#
# train_dataset = (
#     tf.data.Dataset.from_tensor_slices(train_images)
#     .shuffle(BUFFER_SIZE)
#     .batch(BATCH_SIZE, drop_remainder=True)
#     .cache()
#     .prefetch(tf.data.AUTOTUNE)
# )
#
# # ----------------------------
# # Generator
# # ----------------------------
# def make_generator_model():
#     model = tf.keras.Sequential([
#         layers.Dense(7 * 7 * 192, use_bias=False, input_shape=(100,)),
#         layers.BatchNormalization(),
#         layers.LeakyReLU(),
#
#         layers.Reshape((7, 7, 192)),
#
#         layers.Conv2DTranspose(96, 5, strides=1, padding="same", use_bias=False),
#         layers.BatchNormalization(),
#         layers.LeakyReLU(),
#
#         layers.Conv2DTranspose(48, 5, strides=2, padding="same", use_bias=False),
#         layers.BatchNormalization(),
#         layers.LeakyReLU(),
#
#         layers.Conv2DTranspose(1, 5, strides=2, padding="same",
#                                use_bias=False, activation="tanh"),
#     ])
#     return model
#
# generator = make_generator_model()
#
# # ----------------------------
# # Discriminator
# # ----------------------------
# def make_discriminator_model():
#     model = tf.keras.Sequential([
#         layers.Conv2D(48, 5, strides=2, padding="same", input_shape=[28, 28, 1]),
#         layers.LeakyReLU(),
#         layers.Dropout(0.3),
#
#         layers.Conv2D(96, 5, strides=2, padding="same"),
#         layers.LeakyReLU(),
#         layers.Dropout(0.3),
#
#         layers.Flatten(),
#         layers.Dense(1),
#     ])
#     return model
#
# discriminator = make_discriminator_model()
#
# # ----------------------------
# # Losses & Optimizers
# # ----------------------------
# cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#
# def generator_loss(fake_output):
#     return cross_entropy(tf.ones_like(fake_output), fake_output)
#
# def discriminator_loss(real_output, fake_output):
#     return (
#         cross_entropy(tf.ones_like(real_output), real_output) +
#         cross_entropy(tf.zeros_like(fake_output), fake_output)
#     )
#
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
#
# # ----------------------------
# # Training setup
# # ----------------------------
# EPOCHS = 10
# noise_dim = 100
# num_examples_to_generate = 16
#
# seed = tf.random.normal([num_examples_to_generate, noise_dim])
#
# @tf.function(reduce_retracing=True)
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)
#
#         real_output = discriminator(images, training=True)
#         fake_output = discriminator(generated_images, training=True)
#
#         gen_loss = generator_loss(fake_output)
#         disc_loss = discriminator_loss(real_output, fake_output)
#
#     generator_optimizer.apply_gradients(
#         zip(gen_tape.gradient(gen_loss, generator.trainable_variables),
#             generator.trainable_variables)
#     )
#     discriminator_optimizer.apply_gradients(
#         zip(disc_tape.gradient(disc_loss, discriminator.trainable_variables),
#             discriminator.trainable_variables)
#     )
#
# # ----------------------------
# # Image saving
# # ----------------------------
# def generate_and_save_images(model, epoch, test_input):
#     predictions = model(test_input, training=False)
#
#     fig = plt.figure(figsize=(4, 4))
#     for i in range(predictions.shape[0]):
#         plt.subplot(4, 4, i + 1)
#         plt.imshow((predictions[i, :, :, 0] + 1) / 2.0, cmap="gray")
#         plt.axis("off")
#
#     plt.savefig(f"image_at_epoch_{epoch:04d}.png")
#     plt.close(fig)
#
# # ----------------------------
# # Training loop
# # ----------------------------
# def train(dataset, epochs):
#     for epoch in range(epochs):
#         start = time.time()
#
#         for image_batch in dataset:
#             train_step(image_batch)
#
#         # Only generate images every 5 epochs (faster)
#         if (epoch + 1) % 5 == 0:
#             generate_and_save_images(generator, epoch + 1, seed)
#
#         print(f"Epoch {epoch+1} time: {time.time() - start:.2f} sec")
#
# train(train_dataset, EPOCHS)
#
# # ----------------------------
# # Create GIF
# # ----------------------------
# anim_file = "dcgan.gif"
#
# with imageio.get_writer(anim_file, mode="I") as writer:
#     for filename in sorted(glob.glob("image_at_epoch_*.png")):
#         writer.append_data(imageio.imread(filename))
#
# print("Saved animation to dcgan.gif")
