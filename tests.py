import unittest
import numpy as np
from dataloader import *
from metrics import get_top_k_accuracy

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

class TestTopKAccuracy(unittest.TestCase):
    # Top-K accuracy function works based on distance. (i.e. smallest value is the top match).

    # Tests that Top-1 accuracy is 1.0 when all diagonal elements have the lowest distance.
    # Test is run for multiple array sizes.
    def test_all_correct(self):
        for dim in [10,100,1000]:
            with self.subTest(msg="Testing with array dimension: %d" % dim):
                dist_mat = np.ones(shape=(dim,5*dim))
                nrows,ncols = dist_mat.shape
                d_idcs = np.diag_indices(dim)
                dist_mat[d_idcs] = 0.01
                top_k = get_top_k_accuracy(dist_mat=dist_mat,k=1,count_ties=False)
                n_correct = top_k.sum(axis=0)
                acc = n_correct[0] / nrows
                self.assertTrue(acc == 1.0)

    def test_all_same(self):
        for dim in [10,100,1000]:
            for ties in [False,True]:
                with self.subTest(msg="Testing with array dimension: %d and ties = %s" % (dim,ties)):
                    dist_mat = np.ones(shape=(dim,5*dim))
                    nrows,ncols = dist_mat.shape
                    top_k = get_top_k_accuracy(dist_mat=dist_mat,k=1,count_ties=ties)
                    n_correct = top_k.sum(axis=0)
                    acc = n_correct[0] / nrows
                    # If ties==True, then accuracy should be 100% (since all values are the same).
                    # If ties==False, then accuracy should be 0% for the same reason.
                    self.assertTrue(acc == (1.0 if ties else 0.0))

    def test_top_k(self):
        for dim in [10,100]:
            with self.subTest(msg="Testing with array dimension: %d and ties = %s"):
                # Distances will be [0,dim-1]
                dist_mat = np.tile(np.arange(dim),(dim,1))
                nrows,ncols = dist_mat.shape
                top_k = get_top_k_accuracy(dist_mat=dist_mat,k=dim,count_ties=False)
                n_correct = top_k.sum(axis=0)
                self.assertTrue(np.all(n_correct == np.arange(dim)+1))




class TestTrainData(unittest.TestCase):
    def test_something(self):
        pass

class TestPredictData(unittest.TestCase):
    def test_something(self):
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
