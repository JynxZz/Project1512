# Imports
from google.cloud import storage
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Variables
CUR_DIR = os.getcwd()
DATA_DIR = os.path.join(CUR_DIR, "data")

RAW_DATA = os.path.join(DATA_DIR, "raw_data")
DATA_CLEAN = os.path.join(DATA_DIR, "data_clean")

BUCKET_NAME = "test_boobies"
METADATA = pd.read_excel(os.path.join(DATA_DIR, "metadata.xlsx"))

BATCH_SIZE = 64
EPOCHS = 5


# Upload files to GCP bucket storage
def upload_files_to_gcp(bucket_name: str, source_directory: str):
    """
    Uploads all files from a local directory to a GCP bucket.

    Parameters:
    - bucket_name: Name of the GCP bucket.
    - source_directory: Local directory from which to upload files.

    Return:
    - None
    """

    # Initialize GCP Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure the destination blob folder path ends with '/'
    if destination_blob_folder and not destination_blob_folder.endswith("/"):
        destination_blob_folder += "/"

    # Walk through the source directory
    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            # Construct the local file path
            local_path = os.path.join(root, filename)
            # print(local_path)

            # Construct the destination path in the bucket
            if destination_blob_folder:
                relative_path = os.path.relpath(local_path, source_directory)
                blob_path = destination_blob_folder + relative_path
            else:
                blob_path = filename

            # Upload the file
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f'Uploaded {local_path} to "gs://{bucket_name}/{blob_path}"')


def create_and_upload_merged_csv(
    bucket_name: str,
    metadata_csv,
    output_csv_name: str = "ready_to_train.csv",
    file_extension: str = ".jpg",
):
    """
    Fetches files with a specific extension from a GCP bucket, merges their paths with another DataFrame,
    and uploads the merged DataFrame as a CSV to the bucket.

    Parameters:
    - bucket_name: The name of the GCP bucket.
    - metadata_csv: The DataFrame to merge with. It should have columns 'id' and 'label'.
    - output_csv_name: The name of the output CSV file to be stored in the bucket.
    - file_extension: The file extension to filter by. Default is '.jpg'.

    Return:
    - None
    """

    # Initialize a GCP Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create a list to hold file information
    files_info = []

    # Iterate over the files in the bucket, filtering by the specified extension
    for blob in bucket.list_blobs():
        if blob.name.lower().endswith(file_extension):
            file_id = blob.name.rsplit(".", 1)[0]  # Extract the file ID
            files_info.append(
                {
                    "image_id": np.int64(int(file_id)),
                    "path": f"gs://{bucket_name}/{blob.name}",
                }
            )

    # Create a DataFrame from the file information
    df_files = pd.DataFrame(files_info)

    # Select only the columnes we need
    metadata_csv = metadata_csv[["image_id", "cancer"]]

    # Merge the DataFrames on the 'id' column
    merged_df = pd.merge(
        df_files, metadata_csv, on="image_id", how="inner"
    )  # Use for the final CSV
    merged_df = pd.merge(
        df_files, metadata_csv, on="image_id", how="left"
    )  # Use for the tests

    # Convert the DataFrame to a CSV string
    csv_string = merged_df.to_csv(index=False)

    # Save the CSV string to a file in the bucket
    blob = bucket.blob(output_csv_name)
    blob.upload_from_string(csv_string, "text/csv")
    print(f'CSV file "{output_csv_name}" uploaded to bucket "{bucket_name}".')


#####
# Dataset Creation
#####
# Load and process images
def load_and_process_image(file_path: str, label):
    """
    Loads and processes an image file for model training.

    Parameters:
    - file_path: The path to the image file.
    - label: The label associated with the image file.

    Returns:
    - Tuple containing the processed image and its label.
    """

    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [128, 128])  # Resize images
    img = img / 255.0  # Normalize to [0,1]
    return img, label


def create_dataset(input: str = "local"):
    """
    Creates a dataset for model training.

    Parameters:
    - input: Specifies the source of the dataset, 'local' or 'cloud'.

    Returns:
    - TensorFlow dataset object.
    """
    # local or cloud
    # Load the dataset
    if input == "local":
        df = pd.read_csv("ready_to_train.csv")
    if input == "cloud":
        df = pd.read_csv("gs://test_boobies/ready_to_train.csv")

    # Create a TensorFlow dataset
    paths = df["path"].values
    labels = df["cancer"].values

    labels = tf.cast(labels, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_process_image)

    return dataset


#####
# Model
#####
def initialize_model():
    """
    Initializes a sequential model for binary classification.

    Returns:
    - TensorFlow Sequential model.
    """
    model = Sequential()
    model.add(Conv2D(16, (4, 4), input_shape=(128, 128, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=10, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    return model


#####
# Callback
#####
class custom_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs["accuracy"] >= 0.97:
            self.model.stop_training = True


custom_callback = custom_callback()

#####
# Optimizer
#####
optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam"
)

#####
# Loss Fn
#####
lossfn = tf.keras.losses.BinaryCrossentropy(
    from_logits=False, label_smoothing=0.0, axis=-1, name="binary_crossentropy"
)


#####
# Workflow
#####
def initialize_and_compile_model(optimizer, lossfn):
    """
    Initializes the model, compiles it with the specified optimizer and loss function,
    and prints the model summary.

    Parameters:
    - optimizer: The optimizer to use for training the model.
    - lossfn: The loss function to use for training.

    Returns:
    - Compiled TensorFlow model.
    """
    print("\nInit the model :")
    model = initialize_model()
    model.compile(optimizer=optimizer, loss=lossfn, metrics=["accuracy"])
    return model


def batch_dataset(dataset, batch_size: int):
    """
    Batches the dataset with the specified batch size.

    Parameters:
    - dataset: The dataset to batch.
    - batch_size: The size of each batch.

    Returns:
    - Batched dataset.
    """
    return dataset.batch(batch_size)


def split_dataset(batched_dataset, ratio: float = 0.8):
    """
    Splits the batched dataset into training and testing datasets.

    Parameters:
    - batched_dataset: The batched dataset to split.

    Returns:
    - Tuple containing the training and testing datasets.
    """
    size = int(len(batched_dataset) * ratio)

    train = batched_dataset.take(size)
    test = batched_dataset.skip(size)

    return train, test


def train_model(model, train, test, epochs: int, callbacks: list):
    """
    Trains the model on the training dataset and validates it on the testing dataset.

    Parameters:
    - model: The model to train.
    - train: The training dataset.
    - test: The testing dataset.
    - epochs: The number of epochs to train for.
    - callbacks: A list of callbacks to use during training.

    Returns:
    - History object resulting from model training.
    """
    history = model.fit(
        train,
        validation_data=test,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


if __name__ == "__main__":
    # Upload alls jpeg to create the test
    # upload_files_to_gcp(BUCKET_NAME, RAW_DATA)

    # Walk inside the bucket & create the metadata
    # create_and_upload_merged_csv(BUCKET_NAME, METADATA)

    print("==== Starting Workflow ====")

    # Step 1: Create the Dataset
    print("\n=== Step 1: Creating the Dataset ===")
    dataset = create_dataset()
    print("Dataset created successfully.")
    print(dataset)

    # Step 2: Initialize and Compile the Model
    print("\n=== Step 2: Initializing and Compiling the Model ===")
    model = initialize_and_compile_model(optimizer, lossfn)
    print("Model initialized and compiled successfully.")

    # Step 3: Batch the Dataset
    print("\n=== Step 3: Batching the Dataset ===")
    batched_dataset = batch_dataset(dataset, BATCH_SIZE)
    print(f"Dataset batched with batch size {BATCH_SIZE}.")

    # Step 4: Create Train/Test Split
    print("\n=== Step 4: Creating Train/Test Split ===")
    train, test = split_dataset(batched_dataset, 0.8)
    print(
        f"Train/Test split created. Train size: {len(train)}, Test size: {len(test)}."
    )

    # Step 5: Train the Model
    print("\n=== Step 5: Training the Model ===")
    history = train_model(model, train, test, epochs=5, callbacks=[custom_callback])
    print("Model training complete.")

    # Conclusion
    print("\n==== Workflow Completed Successfully ====")
