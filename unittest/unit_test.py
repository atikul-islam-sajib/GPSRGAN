import sys
import unittest
import joblib as pickle

sys.path.append("src/")

from utils import total_trainable_params as compute_params
from src.data_loader import Loader
from src.generator import Generator
from src.discriminator import Discriminator


class UnitTest(unittest.TestCase):
    """
    This test class contains unit tests for validating the data loading process, generator, and discriminator models within the GAN framework.

    Attributes:
        dataloader (DataLoader): DataLoader object that provides batches of data.
        generator (Generator): Instance of the Generator model.
        discriminator (Discriminator): Instance of the Discriminator model.
        total_download_data (int): Counter for the total number of data samples downloaded.
        total_params_generator (int): Counter for the total number of trainable parameters in the generator.
        total_params_discriminator (int): Counter for the total number of trainable parameters in the discriminator.
    """

    def setUp(self):
        """
        Set up method called before each test method.
        Initializes the data loader, generator, and discriminator, and sets the counter attributes to zero.
        """
        self.dataloader = pickle.load(open("./data/processed/dataloader.pkl", "rb"))
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.total_download_data = 0
        self.total_params_generator = 0
        self.total_params_discriminator = 0

    def tearDown(self):
        """
        Tear down method called after each test method.
        Resets the counters for downloaded data samples and the parameters in generator and discriminator.
        """
        self.total_download_data = 0
        self.total_params_generator = 0
        self.total_params_discriminator = 0

    def test_num_download_samples(self):
        """
        Test to verify the total number of downloaded samples.
        Iterates through the dataloader and sums up the samples in each batch,
        then checks if the total number of samples matches the expected count.
        """
        for data, _ in self.dataloader:
            self.total_download_data += data.shape[0]

        self.assertEqual(self.total_download_data, 1000)

    def test_params_on_generator(self):
        """
        Test to verify the number of trainable parameters in the generator model.
        Computes the total number of parameters using a utility function and
        checks if it matches the expected number.
        """
        self.total_params_generator = compute_params(model=self.generator)
        self.assertEqual(self.total_params_generator, 1486352)

    def test_params_on_discriminator(self):
        """
        Test to verify the number of trainable parameters in the discriminator model.
        Computes the total number of parameters using a utility function and
        checks if it matches the expected number.
        """
        self.total_params_discriminator = compute_params(model=self.discriminator)
        self.assertEqual(self.total_params_discriminator, 533505)

    def test_num_download_samples_div2k(self):
        """
        Test to verify the number of trainable parameters in the discriminator model.
        Computes the total number of parameters using a utility function and
        checks if it matches the expected number.
        """
        for data, _ in self.dataloader:
            self.total_download_data += data.shape[0]

        self.assertEqual(self.total_download_data, 196)


if __name__ == "__main__":
    unittest.main()
