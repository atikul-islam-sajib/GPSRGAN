import sys
import logging
import argparse
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/generator.log",
)

sys.path.append("src/")

from utils import total_trainable_params as compute_params


class ResidualBlock(nn.Module):
    """
    A residual block for the Generator network.

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    Methods
    -------
    forward(x)
        Performs a forward pass through the residual block.

    Examples
    --------
    # Create a residual block
    residual_block = ResidualBlock(in_channels=64)

    # Perform forward pass
    output = residual_block(input_tensor)
    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        """
        Perform a forward pass through the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after the residual block.
        """
        return x + self.block(x)


class UpSample(nn.Module):
    """
    An up-sampling module for the Generator network.

    Methods
    -------
    forward(x)
        Performs a forward pass through the up-sampling module.

    Examples
    --------
    # Create an up-sampling module
    upsample_module = UpSample()

    # Perform forward pass
    output = upsample_module(input_tensor)
    """

    def __init__(self):
        super(UpSample, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
        )

    def forward(self, x):
        """
        Perform a forward pass through the up-sampling module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after up-sampling.
        """
        return self.upsample(x)


class Generator(nn.Module):
    """
    The Generator network for image super-resolution.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. Default is 1.
    num_residual_blocks : int, optional
        Number of residual blocks in the network. Default is 16.

    Methods
    -------
    forward(x)
        Performs a forward pass through the generator network.

    Examples
    --------
    # Create a generator
    generator = Generator(in_channels=1, num_residual_blocks=16)

    # Perform forward pass
    output_image = generator(input_image)
    """

    def __init__(self, in_channels=1, num_residual_blocks=16):
        super(Generator, self).__init__()

        # Initial Convolutional Layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU()
        )

        # Residual Blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Up-Sampling Module
        self.upsample = nn.Sequential(*[UpSample() for _ in range(2)])

        # Final Convolutional Layer
        self.final = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4), nn.Tanh()
        )

    def forward(self, x):
        """
        Perform a forward pass through the generator network.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after super-resolution.
        """
        initial = self.initial(x)
        residuals = self.residuals(initial)
        x = initial + residuals
        x = self.upsample(x)
        x = self.final(x)
        return x


if __name__ == "__main__":
    """
    Command-line script to define and log information about the Generator model.
    """
    parser = argparse.ArgumentParser(
        description="Define and log information about the Generator model."
    )
    parser.add_argument(
        "--generator", action="store_true", help="Define the Generator model."
    )

    args = parser.parse_args()

    if args.generator:
        logging.info("Defining Generator Model...")
        generator = Generator()
        logging.info(
            "Total Trainable Parameters: {}".format(compute_params(model=generator))
        )
        logging.info("Generator Model Defined.")
    else:
        logging.info("Generator Model Not Defined.")
