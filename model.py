#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
from paths import *
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB

np.seterr(divide="raise", under="warn", over="warn")

tag_dict = {}


def make_train_test(pqtls, train_data, train_labels, test_data, test_labels, log_transform=True,
                    align_to_reference=False, dump_charts=False, train_other_snps=None, test_other_snps=None):
    """ Formats the training and testing data for input into the model.
    Parameters
    ----------
    pqtls
        A list of tuples, each containing the name of a SNP and protein, respectively in the format (SNP, protein)
    train_data
        Formatted training data (protein data)
    train_labels
        Formatted training labels (genotype/SNP data)
    test_data
        Formatted testing data (protein data)
    test_labels
        Formatted testing labels (genotype/SNP data)
    log_transform : bool
        If True, protein/X values will be transformed with log(x+1) before model training. If False, values are not
        transformed.
    align_to_reference : bool
        Before training, adjust the proteins levels for pQTL using the mean-shift method.
    dump_charts : bool
        Write out some figures in addition to the text output.
    train_other_snps
        Other genotype profiles (presumably without protein data) to be used as part of the genotype pool.
    test_other_snps
        Other genotype profiles to be used as part of the genotype pool.

    Returns
    -------
    Formatted training and testing data.
    """
    snp_list = [p[0] for p in pqtls]
    prot_list = [p[1] for p in pqtls]
    ref_snps = pd.read_csv(REFERENCE_SNPS, index_col=0)
    ref_snps = ref_snps.loc[snp_list].values[:, 0]
    if len(ref_snps[0]) == 1:
        ref_snps = ref_snps*2
    xtrain_tmp = train_data.loc[:, prot_list].values
    xtest_tmp = test_data.loc[:, prot_list].values



    ytrain_tmp = train_labels.loc[:, snp_list].values
    ytest_tmp = test_labels.loc[:, snp_list].values

    # Log transform if required
    if log_transform:
        xtrain_tmp = np.log(1 + xtrain_tmp)
        xtest_tmp = np.log(1 + xtest_tmp)

    unadj_tmp_df_1 = pd.DataFrame(xtrain_tmp,index=train_data.index,columns=prot_list)
    unadj_tmp_df_2 = pd.DataFrame(xtest_tmp,index=test_data.index,columns=prot_list)
    unadj_tmp_df_all = pd.concat([unadj_tmp_df_1,unadj_tmp_df_2]).sort_index()
    unadj_tmp_df_all.to_csv("COPDGene_P2_5K_Unadjusted.csv")

    if dump_charts:
        for j, (snp, prot) in enumerate(pqtls):
            tmp_df = pd.DataFrame({prot: xtrain_tmp[:, j], snp: ytrain_tmp[:, j]})
            tmp_df = tmp_df.loc[tmp_df[snp] != "nan"]
            order_snps = sorted(tmp_df[snp].unique())
            if ref_snps[j] != order_snps[0]:
                order_snps = order_snps[::-1]
            plt.figure()
            sns.histplot(data=tmp_df, x=prot, hue=snp, palette="tab10", hue_order=order_snps)
            # PRINTS Figure 3a
            prot_norm = prot.replace(" ","_").replace(",","_").replace("/","_").lower()
            plt.savefig("figs/hists/%s_%s_unadj.png" % (snp, prot_norm))
            plt.close()

    if align_to_reference:

        # Iterate over train SNPs
        train_mean_diffs = []
        for i in range(xtrain_tmp.shape[1]):
            tmp_y = ytrain_tmp[:, i]
            ref_mean = xtrain_tmp[tmp_y == ref_snps[i], i].mean()
            non_na_y = tmp_y[~(tmp_y == "nan")]
            for snp in np.unique(non_na_y):
                if snp == ref_snps[i]:
                    continue
                mean_diff = xtrain_tmp[tmp_y == snp, i].mean() - ref_mean
                train_mean_diffs.append(mean_diff)
                # Adjust
                xtrain_tmp[tmp_y == snp, i] -= mean_diff
        # Iterate over test SNPs
        for i in range(xtest_tmp.shape[1]):
            tmp_y = ytest_tmp[:, i]
            ref_mean = xtest_tmp[tmp_y == ref_snps[i], i].mean()
            non_na_y = tmp_y[~(tmp_y == "nan")]
            for snp in np.unique(non_na_y):
                if snp == ref_snps[i]:
                    continue
                mean_diff = xtest_tmp[tmp_y == snp, i].mean() - ref_mean
                xtest_tmp[tmp_y == snp, i] -= mean_diff

        if dump_charts:
            for j, (snp, prot) in enumerate(pqtls):
                tmp_df = pd.DataFrame({prot: xtrain_tmp[:, j], snp: ytrain_tmp[:, j]})
                tmp_df = tmp_df.loc[tmp_df[snp] != "nan"]

                order_snps = sorted(tmp_df[snp].unique())
                if ref_snps[j] != order_snps[0]:
                    order_snps = order_snps[::-1]

                plt.figure()
                sns.histplot(data=tmp_df, x=prot, hue=snp, hue_order=order_snps, palette="tab10")
                # Prints Figure 3b
                prot_norm = prot.replace(" ", "_").replace(",", "_").replace("/", "_").lower()
                plt.savefig("figs/hists/%s_%s_adj.png" % (snp, prot_norm))
                plt.close()
    assert np.all(
        train_data.index.values == train_labels.index.values), "Indices for train_data and train_labels do not match!"
    assert np.all(
        test_data.index.values == test_labels.index.values), "Indices for test_data and test_labels do not match!"

    train_sids = train_data.loc[:, prot_list].index.values
    test_sids = test_data.loc[:, prot_list].index.values

    if type(train_other_snps) == pd.DataFrame or type(test_other_snps) == pd.DataFrame:
        train_other_snps_tmp = train_other_snps.loc[:, snp_list]
        test_other_snps_tmp = test_other_snps.loc[:, snp_list]

        tmp_concat = np.concatenate([ytrain_tmp, ytest_tmp, train_other_snps_tmp, test_other_snps_tmp]).T
    else:
        tmp_concat = np.concatenate([ytrain_tmp, ytest_tmp]).T

    all_classes = [np.setdiff1d(np.unique(x), ["nan"]) for x in tmp_concat]

    # data_idcs = [prot_list.index("DERM"),prot_list.index("sICAM-5")]
    # prefix = "adj_" if align_to_reference else "unadj_"
    # tmp_train_out = pd.DataFrame(xtrain_tmp[:,data_idcs],index=train_data.index,columns=["DERM","sICAM-5"])
    # tmp_test_out = pd.DataFrame(xtest_tmp[:,data_idcs],index=test_data.index,columns=["DERM","sICAM-5"])
    #
    # tmp_train_snps_out = pd.DataFrame(ytrain_tmp[:,data_idcs],index=train_labels.index,columns=["rs610403","rs281440"])
    # tmp_test_snps_out = pd.DataFrame(ytest_tmp[:,data_idcs],index=test_labels.index,columns=["rs610403","rs281440"])
    #
    # tmp_train_out.to_csv(prefix+"derm_sicam5_train.csv")
    # tmp_train_snps_out.to_csv(prefix+"derm_sicam5_train_snps.csv")
    #
    # tmp_test_out.to_csv(prefix+"derm_sicam5_test.csv")
    # tmp_test_snps_out.to_csv(prefix+"derm_sicam5_test_snps.csv")

    tmp_df_1 = pd.DataFrame(xtrain_tmp,index=train_data.index,columns=prot_list)
    tmp_df_2 = pd.DataFrame(xtest_tmp,index=test_data.index,columns=prot_list)
    tmp_df_all = pd.concat([tmp_df_1,tmp_df_2]).sort_index()
    tmp_df_all.to_csv("COPDGene_P2_5K_GenotypeAdjusted.csv")

    return np.transpose(xtrain_tmp), np.transpose(ytrain_tmp), train_sids, np.transpose(xtest_tmp), np.transpose(
        ytest_tmp), test_sids, all_classes


def train_model(train_proteins, train_snps, all_classes, skip_train=False):
    """ Trains a GaussianNB model for each pQTL.
    Parameters
    ----------
    train_proteins
        Protein data used for model training
    train_snps
        Genotype data used for model training
    all_classes
        A list of all possible genotypes for each pQTL
    skip_train
        Boolean flag to skip training (useful for just dumping an untrained model file).

    Returns
    -------
    A list of trained models (one per pQTL).
    """
    assert (len(train_proteins) == len(train_snps))
    models = []
    class_orders = []
    class_priors = []
    for i in range(len(train_proteins)):
        # Because of the way we parse the data, missing values are coded as the string
        # 'nan'.
        non_nan_labels = ~(train_snps[i] == "nan")
        use_x = train_proteins[i, non_nan_labels, np.newaxis]
        use_y = train_snps[i, non_nan_labels]

        num_unique = len(all_classes[i])
        class_prior_i = np.ones(num_unique) / num_unique
        nb = GaussianNB(priors=class_prior_i)
        if not skip_train:
            nb.partial_fit(use_x, use_y, classes=all_classes[i])

        # # Check for sICAM-1 manually
        # if i == 2:
        #     from matplotlib.cm import tab10
        #     # FIGURE 1.a
        #     tmp_df = pd.DataFrame({"sICAM-1": use_x.squeeze(), "Genotype": use_y.squeeze()})
        #     plot_pts = np.linspace(use_x.min(), use_x.max(), 200)
        #     preds = nb.predict_proba(plot_pts.reshape(-1, 1))
        #     fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
        #     sns.violinplot(data=tmp_df, x="Genotype", y="sICAM-1", ax=ax[0], order=reversed(nb.classes_),
        #                 palette={"AA": tab10(0),
        #                          "GA": tab10(1),
        #                          "GG": tab10(2)})
        #     ax[1].stackplot(plot_pts, preds.T, labels=nb.classes_)
        #     ax[1].set_xlim(plot_pts.min(), plot_pts.max())
        #     ax[1].set_ylim(0, 1)
        #     ax[1].legend(title="Genotype", fancybox=False, edgecolor="black", facecolor="white", framealpha=1.0)
        #     ax[1].set_xlabel("Log-transformed protein level for sICAM-1")
        #     ax[1].set_ylabel("Genotype Probability")
        #     plt.tight_layout()
        #     plt.savefig("figs/sICAM_prob_cuml.png")
        #     plt.close()

        models.append(nb)
        # assert np.all(nb.classes_ == all_classes[i])
        class_orders.append({v: k for k, v in enumerate(all_classes[i])})
        class_priors.append(class_prior_i)
    return models, class_orders, class_priors


def predict_model(models, test_proteins, log_odds=False):
    """
    Parameters
    ----------
    models
        List of models output from train_model.
    test_proteins
        List of proteins to predict the genotype for.
    log_odds : bool
        If true, use the log-odds method for prediction.
    Returns
    -------
    A set of genotype predictions for test_proteins.
    """
    assert (len(test_proteins) == len(models))

    total_preds = []
    for i in range(len(models)):
        nb = models[i]
        plc = np.full(shape=(test_proteins[0].shape[0], 3), fill_value=np.finfo(np.float32).min, dtype=np.float32)
        if log_odds:
            preds = nb.predict_proba(test_proteins[i, :, np.newaxis])
        else:
            preds = nb.predict_log_proba(test_proteins[i, :, np.newaxis])

        # Enumerate classes used if all three classes aren't here.
        for j, arg_idx in enumerate(np.argsort(nb.classes_)):
            plc[:, arg_idx] = preds[:, j]

        total_preds.append(plc)
    return np.stack(total_preds, axis=0)


def make_progress_bar(progress, width=50):
    """Generates a progress bar for monitoring output progress.
    """
    bar = (("=" * (int(progress * width) - 1)) + ">").ljust(width, " ")
    return "[" + bar + "]"


def eval_model(y_pred, y_true, class_orders, class_priors, num_proteins=100, memo_tag="default", log_odds=False):
    """
    Parameters
    ----------
    y_pred
        A set of predicted genotype profiles output from predict_model
    y_true
        A pool of possible genotype profiles which we will match each proteome profile against.
    class_orders
        The order that classes are arranged in each model.
    class_priors
        The prior probabilities for each class.
    num_proteins
        The number of proteins to use for model evaluation
    memo_tag
        A memoization tag. If we are calculating output results for multiple #s of pQTLs, we can store previous results
        and reuse them to reduce the amount of re-computed results.
    log_odds
        Use log odds for prediction.

    Returns
    -------

    """
    # For each row of y_true
    y_true_copy = y_true.copy()
    # Make copy of y_true and keep track of where NaNs are
    nan_idcs = (y_true == "nan")

    # List for all preds

    use_prev_result = False
    # Check if there is a memoization result we can use.
    if memo_tag in tag_dict:
        for key in sorted(tag_dict[memo_tag].keys(), reverse=True):
            if key <= num_proteins:
                print("Memoized result available!")
                start_i = key
                # Restore the memoized array
                all_preds = tag_dict[memo_tag][key]
                use_prev_result = True
                break
    # If not, start from 0.
    if not use_prev_result:
        tag_dict[memo_tag] = {}
        start_i = 0
        all_preds = np.zeros((y_pred.shape[1], y_true.shape[1]))

    for i in np.arange(start_i, num_proteins):
        print(make_progress_bar(progress=i / num_proteins), end="\r")
        # Get class orders
        class_order = class_orders[i]
        # Set nan values to be a dummy.
        y_true_copy[i, nan_idcs[i]] = sorted(list(class_order.keys()))[0]
        # Get indices
        ind_list = np.array([class_order[x] for x in y_true_copy[i]])
        # One-hot encoded
        onehot = np.eye(3)[ind_list]

        if log_odds:
            onehot[nan_idcs[i]] = np.nan
            # Get the probability that each subject has this genotype
            tmp_preds = np.matmul(y_pred[i], onehot.T)
            tmp_preds = np.maximum(tmp_preds, np.finfo(np.float64).eps)
            tmp_preds = np.log(np.minimum(5, tmp_preds / (1 - tmp_preds + np.finfo(np.float64).eps)))
            tmp_preds[:, nan_idcs[i]] = 0
        else:
            # onehot[nan_idcs[i]] = 1 / (3 * np.ones(3))
            onehot[nan_idcs[i]] = np.nan
            # Get the probability that each subject has this genotype
            tmp_preds = np.matmul(y_pred[i], onehot.T)
            tmp_preds[:, nan_idcs[i]] = np.log(1 / 3)

        all_preds += tmp_preds
        # all_preds += tmp_preds
        del tmp_preds
    print()

    tag_dict[memo_tag][num_proteins] = all_preds
    # full_arr = np.stack(all_preds).transpose(1, 2, 0).sum(axis=-1)

    return all_preds
