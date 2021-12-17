import os
import sys
from numpy.core.shape_base import block

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
os.add_dll_directory(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/extras/CUPTI/lib64"
)
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/include")
os.add_dll_directory("C:/cudnn-windows-x86_64-8.3.1.22_cuda11.5-archive/bin")
os.add_dll_directory("C:/Users/Austin/Documents/nft/TensorRT-8.2.1.8/lib")
os.add_dll_directory("C:/Users/Austin/Documents/nft/dll_x64")
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)

# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#         # for gpu in gpus:
#         #     tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    LeakyReLU,
    Reshape,
    Conv2DTranspose,
    Conv2D,
    Dropout,
    Flatten,
)
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import numpy as np
import time
from IPython import display  # A command shell for interactive computing in Python.
import glob  # The glob module is used for Unix style pathname pattern expansion.
import imageio  # The library that provides an easy interface to read and write a wide range of image data


def make_generator_model():
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Reshape((-1,-1,3)))
    model.add(Dense(8 * 8 * 256, use_bias=False, input_shape=(1000,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False)
    )
    assert model.output_shape == (None, 8, 8, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False)
    )
    assert model.output_shape == (None, 16, 16, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(
        Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", use_bias=False)
    )
    assert model.output_shape == (None, 32, 32, 32)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(
        Conv2DTranspose(16, (5, 5), strides=(2, 2), padding="same", use_bias=False)
    )
    assert model.output_shape == (None, 64, 64, 16)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(
        Conv2DTranspose(
            3, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )
    assert model.output_shape == (None, 128, 128, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()

    model.add(
        Conv2D(64, (5, 5), strides=(2, 2), padding="same", input_shape=[128, 128, 3])
    )
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    # 1 - Generate images
    predictions = model(test_input, training=False)
    # 2 - Plot the generated images
    fig = plt.figure(figsize=(6, 6))
    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i + 1)
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5).numpy().astype(np.int32))
        plt.axis("off")
    # 3 - Save the generated images
    plt.savefig("x/image_at_epoch_{:04d}.png".format(epoch))
    plt.close(fig)
    plt.cla()
    plt.clf()


print("yo we goin")

with open("img_arr.npy", "rb") as f:
    img_arr = np.load(f)
# print(img_arr)
img_arr = img_arr.reshape(img_arr.shape[0], 128, 128, 3).astype(dtype=np.float32)

BUFFER_SIZE = img_arr.shape[0]
BATCH_SIZE = 256

print(img_arr.shape)

train_dataset = (
    tf.data.Dataset.from_tensor_slices(img_arr).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

generator = make_generator_model()
discriminator = make_discriminator_model()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

EPOCHS = 3000

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
num_examples_to_generate = 36
noise_dim = 1000
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# tf.function annotation causes the function
# to be "compiled" as part of the training
@tf.function
def train_step(images):

    # 1 - Create a random noise to feed it into the model
    # for the image generation
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # 2 - Generate images and calculate loss values
    # GradientTape method records operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # 3 - Calculate gradients using loss values and model variables
    # "gradient" method computes the gradient using
    # operations recorded in context of this tape (gen_tape and disc_tape).

    # It accepts a target (e.g., gen_loss) variable and
    # a source variable (e.g.,generator.trainable_variables)
    # target --> a list or nested structure of Tensors or Variables to be differentiated.
    # source --> a list or nested structure of Tensors or Variables.
    # target will be differentiated against elements in sources.

    # "gradient" method returns a list or nested structure of Tensors
    # (or IndexedSlices, or None), one for each element in sources.
    # Returned structure is the same as the structure of sources.
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    # 4 - Process  Gradients and Run the Optimizer
    # "apply_gradients" method processes aggregated gradients.
    # ex: optimizer.apply_gradients(zip(grads, vars))
    """
    Example use of apply_gradients:
    grads = tape.gradient(loss, vars)
    grads = tf.distribute.get_replica_context().all_reduce('sum', grads)
    # Processing aggregated gradients.
    optimizer.apply_gradients(zip(grads, vars), experimental_aggregate_gradients=False)
    """
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


def train(dataset, epochs):
    print("TRAININ")
    # A. For each epoch, do the following:
    for epoch in range(epochs):
        start = time.time()
        # 1 - For each batch of the epoch,
        for image_batch in dataset:
            # 1.a - run the custom "train_step" function
            # we just declared above
            train_step(image_batch)

        # 2 - Produce images for the GIF as we go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # 3 - Save the model every 5 epochs as
        # a checkpoint, which we will use later
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # 4 - Print out the completed epoch no. and the time spent
        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    # B. Generate a final image after the training is completed
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


train(train_dataset, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Display a single image using the epoch number
def display_image(epoch_no):
    return Image.open("x/image_at_epoch_{:04d}.png".format(epoch_no))


display_image(EPOCHS)

anim_file = "dcgan.gif"

with imageio.get_writer(anim_file, mode="I") as writer:
    filenames = glob.glob("x/image*.png")
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    # image = imageio.imread(filename)
    # writer.append_data(image)

display.Image(open("dcgan.gif", "rb").read())
