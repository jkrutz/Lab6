import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from matplotlib import pyplot
from sklearn.model_selection import KFold
from time import perf_counter


def main():
    time_start = perf_counter()
    # Lab 06 Transfer Learning Exercise

    # (1) Get cat & dog data
    train_ds, validation_ds, test_ds = fetch_cats_and_dogs_data()

    # Verify we got data (notice images are all different size).
    show_dogs_and_cats(data=train_ds)

    # Let's resize and normalize the data to make them all 150x150
    train_ds, validation_ds, test_ds = \
        resize_images(train_ds, validation_ds, test_ds)

    # Now, let's batch the data and use
    # caching & pre-fetching to optimize loading speed.
    train_ds, validation_ds, test_ds = \
        batch_cache_prefetch_data(train_ds, validation_ds, test_ds)

    # (2) Create a data augmentation layer to use during training
    data_augmentation = create_data_augmentation_layer()

    # View a sample of an augmented image.
    preview_augmentations(data=train_ds,
                          data_augmentation=data_augmentation,
                          num_imgs_preview=1)

    # (3) Build new deep model using Xception as the base model
    model, base_model = build_new_transfer_model(
        data_augmentation=data_augmentation)

    # (4) Train our new model
    history = train_model(model=model,
                          train_ds=train_ds,
                          validation_ds=validation_ds,
                          epochs=1)

    # evaluate model
    summarize_diagnostics(history)
    print("Evaluating model...")
    _, acc = model.evaluate(test_ds, verbose=1)
    print('> %.3f' % (acc * 100.0))
    evaluate_roc(model, test_ds, "New TF Model")

    # stores scores

    # (5) Fine-tune our model...
    fine_tune_model(model=model,
                    base_model=base_model,
                    train_ds=train_ds,
                    validation_ds=validation_ds,
                    epochs=2)

    evaluate_roc(model, test_ds, "Fine-Tuned Model")

    plt.legend(loc="lower right")
    plt.show()

    time_stop = perf_counter()

    elapsed_time = time_stop-time_start

    print("Elapsed time (in seconds):", elapsed_time)

    return


def fetch_cats_and_dogs_data():
    # First, let's fetch the cats vs. dogs dataset using TFDS.
    # If you have your own dataset, you'll probably want to use
    # the utility tf.keras.preprocessing.image_dataset_from_directory
    # to generate similar
    # labeled dataset objects from a set of images on disk filed into
    # class-specific folders.

    # Note that we're only using 40% of the 23,260 images in the dataset, since
    # we're looking to demonstrate transfer learning using a small dataset.
    # The testing and validation sets will hold only 2,326 elements each.

    train_ds, validation_ds, test_ds = tfds.load(
        "cats_vs_dogs",
        # Reserve 10% for validation and 10% for test
        split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
        as_supervised=True  # Include labels
    )

    # Note: a TF dataset is inherently lazily loaded.
    # So we might not know the size of the dataset up front.
    # Since we want to know how many elements are loaded,
    # we'll force the data to load into memory so elements can be counted
    # enumerating through the elements (normally, you won't want to do
    # this on very large sets).

    print(f"Number of training samples: {[i for i, _ in enumerate(train_ds)][-1] + 1}")
    print(f"Number of validation samples: {[i for i, _ in enumerate(validation_ds)][-1] + 1}")
    print(f"Number of test samples: {[i for i, _ in enumerate(test_ds)][-1] + 1}")

    return train_ds, validation_ds, test_ds


def show_dogs_and_cats(data, num_elements=9):
    # These are the first num_elements images in the training dataset.
    # Prior to executing the resize_images() function, they're all different sizes.
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(data.take(num_elements)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title(int(label))
        plt.axis("off")
    plt.show()


def resize_images(train_ds, validation_ds, test_ds):
    # Our raw images have a variety of sizes.
    # In addition, each pixel consists of 3 integer values between 0 and 255 (RGB level values).
    # This isn't a great fit for feeding a neural network. We need to do 2 things:
    #
    #   Standardize to a fixed image size. We pick 150x150.
    train_ds = train_ds.map(resize_image)
    validation_ds = validation_ds.map(resize_image)
    test_ds = test_ds.map(resize_image)

    return train_ds, validation_ds, test_ds


def resize_image(image, label):
    return tf.image.resize(image, (150, 150)), label


def batch_cache_prefetch_data(train_ds, validation_ds, test_ds):
    # Let's batch the data and use caching & prefetching to optimize loading speed.

    batch_size = 32

    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

    return train_ds, validation_ds, test_ds


def create_data_augmentation_layer():
    # When you don't have a large image dataset,
    # it's a good practice to artificially introduce sample diversity
    # by applying random yet realistic transformations to the training images.
    data_augmentation = keras.Sequential(
        [
            preprocessing.RandomFlip("horizontal"),
            preprocessing.RandomRotation(0.1),
        ]
    )
    return data_augmentation


def preview_augmentations(data, data_augmentation, num_imgs_preview=1):
    for images, labels in data.take(num_imgs_preview):
        plt.figure(figsize=(10, 10))
        first_image = images[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(
                tf.expand_dims(first_image, 0), training=True)
            plt.imshow(augmented_image[0].numpy().astype("int32"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.show()


def build_new_transfer_model(data_augmentation=None):
    if data_augmentation is None:
        data_augmentation = create_data_augmentation_layer()

    # We'll start with the Xception model as our base model.
    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False,
    )  # Do not include the ImageNet classifier at the "top".

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(150, 150, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation

    # Pre-trained Xception weights requires that input be normalized
    # from (0, 255) to a range (-1., +1.); normalize pixel values to between -1 and 1.
    # Outputs = (inputs - mean) / sqrt(var)
    #   We'll do this using a Normalization layer as part of the model itself.
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mn = np.array([127.5] * 3)
    var = mn ** 2

    # Scale inputs to [-1, +1]
    x = norm_layer(x)
    norm_layer.set_weights([mn, var])

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout

    # Add dense layer with 1 output for binary classification
    # #(the probability of a dog).
    # Note: to calculate the number of parameters (i.e. weights) in this new layer,
    #       param_number = output_params * (input_params + 1)
    #       So, in our case, the output from the Xception model is 2048,
    #       that becomes our input to the dense layer; therefore,
    #       param_number= 1 * (2048 +1) = 2049
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model, base_model


def train_model(model, train_ds, validation_ds, epochs=20):
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy()], )

    return model.fit(train_ds, epochs=epochs, validation_data=validation_ds)


def fine_tune_model(model, base_model, train_ds, validation_ds, epochs=10):
    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    base_model.trainable = True
    model.summary()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()], )

    return model.fit(train_ds, epochs=epochs, validation_data=validation_ds)


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    # plot loss
    pyplot.subplot(2, 1, 1)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(histories.history['loss'], color='blue', label='loss')
    pyplot.plot(histories.history['val_loss'], color='orange', label='val_loss')
    # plot accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.title('Classification Accuracy')
    pyplot.plot(histories.history['binary_accuracy'], color='blue', label='binary_accuracy')
    pyplot.plot(histories.history['val_binary_accuracy'], color='orange', label='val_binary_accuracy')
    pyplot.show()


def evaluate_roc(model, test_ds, label=None):
    # Convert labels into simple array
    labels = np.concatenate([y for x, y in test_ds], axis=0)
    # Calculate prediction scores
    y_scores = model.predict(test_ds).ravel()
    # Get AUC
    print("Area under ROC curve:", roc_auc_score(labels, y_scores), "\n")
    # Get ROC curve
    fpr, tpr, thresholds=roc_curve(labels, y_scores)
    # Now, plot ROC curve
    plot_roc_curve(fpr, tpr, label)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid()
    plt.ylabel("True Positive Rate (Recall)")
    plt.xlabel("False Positive Rate")


main()
