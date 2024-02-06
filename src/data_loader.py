import sys
import logging
import argparse
import os
import cv2
import numpy as np
import joblib as pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

sys.path.append("src/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    filemode="w",
    filename="./logs/data_loader.log",
)


class Loader:
    """
    Loader class for creating a dataloader for the MNIST dataset.

    Parameters
    ----------
    batch_size : int, optional
        The batch size for the dataloader. Default is 64.
    num_samples : int, optional
        The number of samples to use from the MNIST dataset. Default is 1000.
    image_height : int, optional
        The height of the resized images. Default is 64.
    image_width : int, optional
        The width of the resized images. Default is 64.

    Methods
    -------
    download_mnist()
        Downloads the MNIST dataset, applies transformations, and returns a subset.

    create_dataloader(subset_dataset)
        Creates and saves a dataloader for the provided subset using joblib.

    Examples
    --------
    # Instantiate the Loader class
    loader = Loader(batch_size=64, num_samples=1000, image_height=64, image_width=64)

    # Download the MNIST dataset
    subset_dataset = loader.download_mnist()

    # Create and save the dataloader
    loader.create_dataloader(subset_dataset=subset_dataset)
    """

    def __init__(
        self, batch_size=64, num_samples=1000, image_height=64, image_width=64
    ):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.image_height = 64
        self.image_width = 64
        self.in_channels = 1
        self.lr_images = list()

    def download_mnist(self):
        """
        Downloads the MNIST dataset, applies transformations, and returns a subset.

        Returns
        -------
        subset_dataset : Subset
            Subset of the MNIST dataset.
        """
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_height, self.image_width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        full_dataset = MNIST(
            root="./data/raw/",
            train=True,
            download=True,
            transform=transform,
        )
        subset_indices = range(self.num_samples)
        subset_dataset = Subset(full_dataset, subset_indices)

        return subset_dataset

    def download_div2k(self):
        """
        Loads and processes low-resolution images from the DIV2K dataset.

        This method reads low-resolution images from a specified directory,
        resizes them to the dimensions defined in the instance (self.image_height, self.image_width),
        and appends them to the instance's lr_images list. The images are then converted to a NumPy array,
        normalized (pixel values are divided by 255 to bring them into the range [0,1]),
        and reshaped to the format expected by the model (batch_size, channels, height, width).

        The method assumes that the low-resolution images are stored in a subdirectory './data/low'
        relative to the current working directory.

        Returns:
            numpy.ndarray: A 4D NumPy array containing the processed low-resolution images,
                        with shape (num_images, 1, self.image_height, self.image_width).

        Notes:
            - The method updates self.lr_images with the processed images.
            - The images are read in grayscale mode.
            - This method does not perform error handling for file reading.
            Ensure that the specified directory contains only valid image files.
        """
        lr_image_path = os.path.join("./data/low")
        for image in os.listdir(lr_image_path):
            image = cv2.imread(os.path.join(lr_image_path, image), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (self.image_height, self.image_width))
            self.lr_images.append(image)

        # Convert list into NumPy
        self.lr_images = np.array(self.lr_images).astype(np.float32) / 255.0
        self.lr_images = self.lr_images.reshape(
            -1, self.in_channels, self.image_height, self.image_width
        )

        return self.lr_images

    def create_dataloader(self, subset_dataset=None):
        """
        Creates and saves a dataloader for the provided subset using joblib.

        Parameters
        ----------
        subset_dataset : Subset, optional
            Subset of the MNIST dataset.

        Raises
        ------
        Exception
            If subset_dataset is not provided.

        Notes
        -----
        The dataloader is saved as a pickle file in the "./data/processed/" directory.
        """
        if subset_dataset is not None:
            dataloader = DataLoader(
                dataset=subset_dataset, batch_size=self.batch_size, shuffle=True
            )

            try:
                pickle.dump(
                    value=dataloader, filename="./data/processed/dataloader.pkl"
                )
            except Exception as e:
                logging.exception("Pickle file is not saved...".capitalize())

        else:
            raise Exception("Dataloader needs to have the subset dataset".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the dataloader".capitalize())

    parser.add_argument(
        "--download_mnist",
        action="store_true",
        help="Download the mnist dataset".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--subset_samples",
        type=int,
        default=1000,
        help="Number of samples to use".capitalize(),
    )
    parser.add_argument(
        "--download_div2k",
        action="store_true",
        help="Download the mnist dataset".capitalize(),
    )

    args = parser.parse_args()

    if args.download_mnist:
        if args.batch_size and args.subset_samples:
            logging.info("Downloading the mnist dataset...".capitalize())

            loader = Loader(
                batch_size=args.batch_size,
                num_samples=args.subset_samples,
                image_height=64,
                image_width=64,
            )
            subset_dataset = loader.download_mnist()
            loader.create_dataloader(subset_dataset=subset_dataset)

            logging.info("Dataloader created successfully".capitalize())
        else:
            logging.info(
                "Please provide the batch size and the number of samples".capitalize()
            )

    if args.download_div2k:
        if args.batch_size and args.subset_samples:
            logging.info("Downloading the mnist dataset...".capitalize())

            loader = Loader(
                batch_size=args.batch_size,
                num_samples=args.subset_samples,
                image_height=64,
                image_width=64,
            )
            subset_dataset = loader.download_div2k()
            loader.create_dataloader(subset_dataset=subset_dataset)

            logging.info("Dataloader created successfully".capitalize())
        else:
            logging.info(
                "Please provide the batch size and the number of samples".capitalize()
            )
