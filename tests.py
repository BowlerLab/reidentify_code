import unittest
from dataloader import *

TEST_DATA_PATH = "test_data"

class TestLoadData(unittest.TestCase):
    # Load a dataset from CSV, and check that the result is equivalent to the binary file we have on hand.
    # If the dataset changes and the binary file is not regenerated, this test will fail.
    def test_load_datasets(self):
        from os.path import join
        import pickle
        for name in ALL_DATASET_NAMES:
            with self.subTest(msg="Test loading '{name}'".format(name=name)):
                # Load binary dataset.
                filename = "{name}_binary_dataset.pkl".format(name=name.lower())
                binary_dataset_path = join(TEST_DATA_PATH,filename)
                with open(binary_dataset_path,"rb") as f:
                    binary_dataset = pickle.load(f)

                # Load CSV dataset.
                prots,snps,clin,other_snps = load_dataset(name)

                # Compare each member
                self.assertTrue(name == binary_dataset["name"],msg="Checking dataset name.")
                self.assertTrue(prots.equals(binary_dataset["proteins"]),msg="Checking protein data equality.")
                self.assertTrue(snps.equals(binary_dataset["snps"]),msg="Checking SNP data equality.")
                self.assertTrue(clin.equals(binary_dataset["clinical"]),msg="Checking clinical data equality.")
                self.assertTrue(other_snps.equals(binary_dataset["other_snps"]),msg="Checking background SNPs.")



class TestTrainData(unittest.TestCase):
    def test_something(self):
        pass

class TestPredictData(unittest.TestCase):
    def test_something(self):
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
