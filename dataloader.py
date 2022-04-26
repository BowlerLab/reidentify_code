import os
import pandas as pd
from paths import *


def load_proteins_snps_clinical(protein_path, snp_path, clinical_path, protein_sep="\t", snp_sep=",", clin_sep="\t",
                                clin_age_col="age_baseline", clin_race_col="race", clin_gender_col="gender"):
    protein_df = pd.read_csv(protein_path, sep=protein_sep, index_col=0)
    snp_df = pd.read_csv(snp_path, sep=snp_sep, index_col=0, keep_default_na=False)
    clinical_df = pd.read_csv(clinical_path, sep=clin_sep, index_col=0)
    # Get intersection of SIDs
    use_sids = set(protein_df.index).intersection(snp_df.columns)
    # We can use the genotypes for SIDs which do not have protein measurements as extras when testing.
    other_sids = set(snp_df.columns) - use_sids
    protein_df = protein_df.loc[use_sids, :]
    snp_df_reduced = snp_df.loc[:, use_sids].transpose()
    snp_df_other = snp_df.loc[:, other_sids].transpose()
    clinical_df = clinical_df.loc[use_sids, [clin_race_col, clin_gender_col, clin_age_col]]
    clinical_df.columns = ["race", "gender", "age"]
    return protein_df.sort_index(), snp_df_reduced.sort_index(), clinical_df.sort_index(), snp_df_other.sort_index()


def load_copdgene_p1():
    return load_proteins_snps_clinical(COPDGene_P1_PROTEINS, COPDGene_P1_SNPS, COPDGene_P1_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P1_AGE_COL,
                                       clin_race_col=COPDGene_P1_RACE_COL, clin_gender_col=COPDGene_P1_GENDER_COL)


def load_copdgene_p1_jhs():
    return load_proteins_snps_clinical(COPDGene_P1_JHS_PROTEINS, COPDGene_P1_JHS_SNPS, COPDGene_P1_JHS_CLINICAL,
                                       protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P1_JHS_AGE_COL,
                                       clin_race_col=COPDGene_P1_JHS_RACE_COL,
                                       clin_gender_col=COPDGene_P1_JHS_GENDER_COL)


def load_copdgene_p1_jhs_only():
    return load_proteins_snps_clinical(COPDGene_P1_JHS_ONLY_PROTEINS, COPDGene_P1_JHS_ONLY_SNPS,
                                       COPDGene_P1_JHS_ONLY_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P1_JHS_ONLY_AGE_COL,
                                       clin_race_col=COPDGene_P1_JHS_ONLY_RACE_COL,
                                       clin_gender_col=COPDGene_P1_JHS_ONLY_GENDER_COL)


def load_copdgene_p2():
    return load_proteins_snps_clinical(COPDGene_P2_PROTEINS, COPDGene_P2_SNPS, COPDGene_P2_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P2_AGE_COL,
                                       clin_race_col=COPDGene_P2_RACE_COL, clin_gender_col=COPDGene_P2_GENDER_COL)


def load_copdgene_p2_jhs():
    return load_proteins_snps_clinical(COPDGene_P2_JHS_PROTEINS, COPDGene_P2_JHS_SNPS, COPDGene_P2_JHS_CLINICAL,
                                       protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P2_JHS_AGE_COL,
                                       clin_race_col=COPDGene_P2_JHS_RACE_COL,
                                       clin_gender_col=COPDGene_P2_JHS_GENDER_COL)


def load_copdgene_p1_p2():
    p1_protein, p1_snp, p1_clin, p1_other_snp = load_copdgene_p1()
    p2_protein, p2_snp, p2_clin, p2_other_snp = load_copdgene_p2()

    comb_protein = pd.concat([p1_protein, p2_protein], axis=0)
    comb_snp = pd.concat([p1_snp, p2_snp], axis=0)
    comb_clin = pd.concat([p1_clin, p2_clin], axis=0)
    # Get only SNPs which appear in both p1 and p2 'other' SNPs (i.e. don't appear in either of the original sets.
    other_snp_index = set(p1_other_snp.index).intersection(p2_other_snp.index)
    comb_other_snp = p1_other_snp.loc[other_snp_index]

    assert (len(other_snp_index.intersection(comb_snp.index)) == 0)
    return comb_protein, comb_snp, comb_clin, comb_other_snp


def load_spiromics():
    return load_proteins_snps_clinical(SPIROMICS_PROTEINS, SPIROMICS_SNPS, SPIROMICS_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_sep=",", clin_age_col=SPIROMICS_AGE_COL,
                                       clin_race_col=SPIROMICS_RACE_COL, clin_gender_col=SPIROMICS_GENDER_COL)


def load_spiromics_jhs():
    return load_proteins_snps_clinical(SPIROMICS_JHS_PROTEINS, SPIROMICS_JHS_SNPS, SPIROMICS_JHS_CLINICAL,
                                       protein_sep="\t",
                                       snp_sep=",", clin_sep=",", clin_age_col=SPIROMICS_JHS_AGE_COL,
                                       clin_race_col=SPIROMICS_JHS_RACE_COL, clin_gender_col=SPIROMICS_JHS_GENDER_COL)


def load_spiromics_jhs_only():
    return load_proteins_snps_clinical(SPIROMICS_JHS_ONLY_PROTEINS, SPIROMICS_JHS_ONLY_SNPS,
                                       SPIROMICS_JHS_ONLY_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_sep=",", clin_age_col=SPIROMICS_JHS_ONLY_AGE_COL,
                                       clin_race_col=SPIROMICS_JHS_ONLY_RACE_COL,
                                       clin_gender_col=SPIROMICS_JHS_ONLY_GENDER_COL)


def load_copdgene_p2_5k():
    return load_proteins_snps_clinical(COPDGene_P2_5K_PROTEINS, COPDGene_P2_5K_SNPS, COPDGene_P2_5K_CLINICAL,
                                       protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P2_5K_AGE_COL,
                                       clin_race_col=COPDGene_P2_5K_RACE_COL, clin_gender_col=COPDGene_P2_5K_GENDER_COL)


def load_copdgene_p2_5k_all():
    return load_proteins_snps_clinical(COPDGene_P2_5K_ALL_PROTEINS, COPDGene_P2_5K_ALL_SNPS,
                                       COPDGene_P2_5K_ALL_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P2_5K_ALL_AGE_COL,
                                       clin_race_col=COPDGene_P2_5K_ALL_RACE_COL,
                                       clin_gender_col=COPDGene_P2_5K_ALL_GENDER_COL)


def load_copdgene_p3_5k_all():
    return load_proteins_snps_clinical(COPDGene_P3_5K_ALL_PROTEINS, COPDGene_P3_5K_ALL_SNPS,
                                       COPDGene_P3_5K_ALL_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P3_5K_ALL_AGE_COL,
                                       clin_race_col=COPDGene_P3_5K_ALL_RACE_COL,
                                       clin_gender_col=COPDGene_P3_5K_ALL_GENDER_COL)

def load_copdgene_5k_train():
    return load_proteins_snps_clinical(COPDGene_5k_TRAIN_PROTEINS,COPDGene_5k_TRAIN_GENO,
                                       COPDGene_5k_TRAIN_CLINICAL, protein_sep=",",snp_sep="\t",
                                       clin_age_col=COPDGene_5k_TRAIN_AGE_COL,clin_race_col=COPDGene_5k_TRAIN_RACE_COL,
                                       clin_gender_col=COPDGene_5k_TRAIN_GENDER_COL)

def load_copdgene_5k_test():
    return load_proteins_snps_clinical(COPDGene_5k_TEST_PROTEINS,COPDGene_5k_TEST_GENO,
                                       COPDGene_5k_TEST_CLINICAL, protein_sep=",",snp_sep="\t",
                                       clin_age_col=COPDGene_5k_TEST_AGE_COL,clin_race_col=COPDGene_5k_TEST_RACE_COL,
                                       clin_gender_col=COPDGene_5k_TEST_GENDER_COL)

def load_copdgene_5k_qc_train():
    return load_proteins_snps_clinical(COPDGene_5k_QC_TRAIN_PROTEINS,COPDGene_5k_QC_TRAIN_GENO,
                                       COPDGene_5k_QC_TRAIN_CLINICAL, protein_sep=",",snp_sep=",",
                                       clin_age_col=COPDGene_5k_QC_TRAIN_AGE_COL,clin_race_col=COPDGene_5k_QC_TRAIN_RACE_COL,
                                       clin_gender_col=COPDGene_5k_QC_TRAIN_GENDER_COL)

def load_copdgene_5k_qc_test():
    return load_proteins_snps_clinical(COPDGene_5k_QC_TEST_PROTEINS,COPDGene_5k_QC_TEST_GENO,
                                       COPDGene_5k_QC_TEST_CLINICAL, protein_sep=",",snp_sep=",",
                                       clin_age_col=COPDGene_5k_QC_TEST_AGE_COL,clin_race_col=COPDGene_5k_QC_TEST_RACE_COL,
                                       clin_gender_col=COPDGene_5k_QC_TEST_GENDER_COL)

def load_dataset(name):
    # Select the training dataset.
    if name == COPDGene_P1_NAME:
        prots, snps, clin, other_snps = load_copdgene_p1()
    elif name == COPDGene_P2_NAME:
        prots, snps, clin, other_snps = load_copdgene_p2()
    elif name == COPDGene_P2_JHS_NAME:
        prots, snps, clin, other_snps = load_copdgene_p2_jhs()
    elif name == COPDGene_P1_P2_NAME:
        prots, snps, clin, other_snps = load_copdgene_p1_p2()
    elif name == COPDGene_P2_5K_NAME:
        prots, snps, clin, other_snps = load_copdgene_p2_5k()
    elif name == COPDGene_P2_5K_ALL_NAME:
        prots, snps, clin, other_snps = load_copdgene_p2_5k_all()
    elif name == COPDGene_P3_5K_ALL_NAME:
        prots, snps, clin, other_snps = load_copdgene_p3_5k_all()
    elif name == SPIROMICS_NAME:
        prots, snps, clin, other_snps = load_spiromics()
    elif name == COPDGene_P1_JHS_NAME:
        prots, snps, clin, other_snps = load_copdgene_p1_jhs()
    elif name == SPIROMICS_JHS_NAME:
        prots, snps, clin, other_snps = load_spiromics_jhs()
    elif name == COPDGene_P1_JHS_ONLY_NAME:
        prots, snps, clin, other_snps = load_copdgene_p1_jhs_only()
    elif name == SPIROMICS_JHS_ONLY_NAME:
        prots, snps, clin, other_snps = load_spiromics_jhs_only()
    elif name == COPDGene_5k_TRAIN_NAME:
        prots, snps, clin, other_snps = load_copdgene_5k_train()
    elif name == COPDGene_5k_TEST_NAME:
        prots, snps, clin, other_snps = load_copdgene_5k_test()
    elif name == COPDGene_5k_QC_TRAIN_NAME:
        prots, snps, clin, other_snps = load_copdgene_5k_qc_train()
    elif name == COPDGene_5k_QC_TEST_NAME:
        prots, snps, clin, other_snps = load_copdgene_5k_qc_test()
    else:
        raise ValueError("bad dataset name.")
    return prots, snps, clin, other_snps


# Make a binary representation of the dataset and dump it to a file.
def dump_dataset_to_file(name, prots, snps, clin, other_snps, filename):
    import pickle
    dump_data = {"name": name,
                 "proteins": prots,
                 "snps": snps,
                 "clinical": clin,
                 "other_snps": other_snps}
    with open(filename, "wb") as f:
        pickle.dump(dump_data, f)
        print("Wrote {filename} to file.".format(filename=filename))


def load_data(args):
    train_data_arg = args.train_data
    test_data_arg = args.test_data

    train_p, train_snp, train_clin, train_o_snp = load_dataset(train_data_arg)
    print("Loaded %s as training set." % train_data_arg)

    # If training and testing data are the same, we don't need to read the files again.
    if train_data_arg == test_data_arg:
        test_p, test_snp, test_clin, test_o_snp = (train_p, train_snp, train_clin, train_o_snp)
    else:
        test_p, test_snp, test_clin, test_o_snp = load_dataset(test_data_arg)
    print("Loaded %s as testing set." % test_data_arg)

    # Check for overlap between train and test and correct (Only for COPDGene)
    if train_p is not None and test_p is not None:
        intersect_sids = set(train_p.index).intersection(test_p.index)
        print("Found %d intersecting SIDs." % len(intersect_sids))

    return train_data_arg, train_p, train_snp, train_clin, train_o_snp, test_data_arg, test_p, test_snp, test_clin, test_o_snp


# If run as the main function, we generate binary files for all datasets.
if __name__ == "__main__":
    from datetime import datetime
    from os.path import join
    from os import makedirs

    date_string = datetime.now().strftime("%d_%m_%y")
    dir_name = "binary_data_{date}".format(date=date_string)

    os.makedirs(dir_name)

    print("Generating binary files for all datasets. Results will be saved in '{}'.".format(dir_name))

    for name in ALL_DATASET_NAMES:
        filename = join(dir_name, "{}_binary_dataset.pkl".format(name.lower()))

        prots, snps, clin, other_snps = load_dataset(name)

        dump_dataset_to_file(name=name,
                             prots=prots,
                             snps=snps,
                             clin=clin,
                             other_snps=other_snps,
                             filename=filename)
