import sys
import unittest
import joblib as pickle

sys.path.append("src/")

from utils import total_trainable_params as compute_params
from src.data_loader import DataLoader
from src.generator import Generator


class UnitTest(unittest.TestCase):

    def setUp(self):
        self.dataloader = pickle.load(open("./data/processed/dataloader.pkl", "rb"))
        self.generator = Generator()
        self.total_download_data = 0
        self.total_params_generator = 0

    def tearDown(self):
        self.total_download_data = 0
        self.total_params_generator = 0

    def test_num_download_samples(self):
        for data, _ in self.dataloader:
            self.total_download_data += data.shape[0]

        self.assertEqual(self.total_download_data, 1000)

    def test_params_on_generator(self):
        self.total_params_generator = compute_params(model=self.generator)
        self.assertEqual(self.total_params_generator, 1486352)


if __name__ == "__main__":
    unittest.main()
