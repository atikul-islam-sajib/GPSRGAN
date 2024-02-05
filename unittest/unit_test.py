import sys
import unittest
import joblib as pickle

sys.path.append("src/")

from src.data_loader import DataLoader


class UnitTest(unittest.TestCase):

    def setUp(self):
        self.dataloader = pickle.load(open("./data/processed/dataloader.pkl", "rb"))
        self.total_download_data = 0

    def tearDown(self):
        self.total_download_data = 0

    def test_num_download_samples(self):
        for data, _ in self.dataloader:
            self.total_download_data += data.shape[0]

        self.assertEqual(self.total_download_data, 1000)


if __name__ == "__main__":
    unittest.main()
