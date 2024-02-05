import sys
import logging
import argparse
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/discriminator.log",
)

sys.path.append("src/")

from utils import total_trainable_params as compute_params


class Discriminator(nn.Module):
    """
    Discriminator module for adversarial training in a Generative Adversarial Network (GAN).

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. Default is 1.

    Attributes
    ----------
    nf : int
        Number of filters in the first convolutional layer.

    main : torch.nn.Sequential
        Main sequential architecture of the discriminator.

    Methods
    -------
    forward(x)
        Performs a forward pass through the discriminator network.

    Examples
    --------
    # Create a discriminator
    discriminator = Discriminator(in_channels=1)

    # Perform forward pass
    output = discriminator(input_tensor)

    """

    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.nf = 64
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.nf,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.nf,
                out_channels=self.nf,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.nf),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.nf,
                out_channels=self.nf * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.nf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.nf * 2,
                out_channels=self.nf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.nf * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.nf * 2,
                out_channels=self.nf * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.nf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.nf * 4,
                out_channels=self.nf * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.nf * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.nf * 4,
                out_channels=self.nf * 8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.nf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.nf * 8,
                out_channels=self.nf * 8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.nf * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.nf * 8, out_channels=1024, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        """
        Perform a forward pass through the discriminator network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the discriminator.
        """
        if x is not None:
            x = self.main(x)
        else:
            logging.exception("Features is not defined in discriminator".capitalize())
            raise Exception("Features is not defined in discriminator".capitalize())

        return torch.sigmoid(x.view(x.size(0), -1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discriminator model defined".capitalize()
    )
    parser.add_argument(
        "--discriminator", action="store_true", help="Discriminator model".capitalize()
    )

    args = parser.parse_args()

    if args.discriminator:
        logging.info("Discriminator model defined".capitalize())
        discriminator = Discriminator()

        try:
            logging.info(
                "Trainable params # {}".format(compute_params(model=discriminator))
            )
        except Exception as e:
            logging.exception("Model is not passed to compute parameters".capitalize())

    else:
        logging.exception("Discriminator model is not defined".capitalize())
        raise Exception("Discriminator model is not defined".capitalize())
