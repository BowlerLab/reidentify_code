#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
np.seterr(divide="raise",under="warn",over="warn")

import argparse
import pickle
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.naive_bayes import GaussianNB
from collections import Counter
import seaborn as sns
from matplotlib.cm import tab10
from matplotlib.ticker import AutoMinorLocator
from paths import *

#PAIR_LIST = "output/use_pqtls_p1_only.csv"
#REFERENCE_SNPS = "output/reference_snps_p1.csv"


def get_train_test_2(pqtls, train_data, train_labels, test_data, test_labels, log_transform=True,
                     align_to_reference=False, dump_charts=False, train_other_snps = None, test_other_snps = None):
    snp_list = [p[0] for p in pqtls]
    prot_list = [p[1] for p in pqtls]
    ref_snps = pd.read_csv(REFERENCE_SNPS, index_col=0)
    ref_snps = ref_snps.loc[snp_list].values[:, 0]

    xtrain_tmp = train_data.loc[:, prot_list].values
    xtest_tmp = test_data.loc[:, prot_list].values

    ytrain_tmp = train_labels.loc[:, snp_list].values
    ytest_tmp = test_labels.loc[:, snp_list].values

    # Log transform if required
    if log_transform:
        xtrain_tmp = np.log(1 + xtrain_tmp)
        xtest_tmp = np.log(1 + xtest_tmp)

        # REMOVE THIS
        #scaler = StandardScaler()
        #xtrain_tmp = scaler.fit_transform(xtrain_tmp)
        #xtest_tmp = scaler.fit_transform(xtest_tmp)

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
            plt.savefig("figs/hists/%s_%s_unadj.png" % (snp, prot))
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
                plt.savefig("figs/hists/%s_%s_adj.png" % (snp, prot))
                plt.close()
    assert np.all(
        train_data.index.values == train_labels.index.values), "Indices for train_data and train_labels do not match!"
    assert np.all(
        test_data.index.values == test_labels.index.values), "Indices for test_data and test_labels do not match!"

    train_sids = train_data.loc[:, prot_list].index.values
    test_sids = test_data.loc[:, prot_list].index.values

    if type(train_other_snps) == pd.DataFrame or type(test_other_snps) == pd.DataFrame:
        train_other_snps_tmp = train_other_snps.loc[:,snp_list]
        test_other_snps_tmp = test_other_snps.loc[:,snp_list]

        tmp_concat = np.concatenate([ytrain_tmp,ytest_tmp,train_other_snps_tmp,test_other_snps_tmp]).T
    else:
        tmp_concat = np.concatenate([ytrain_tmp, ytest_tmp]).T

    all_classes = [np.setdiff1d(np.unique(x), ["nan"]) for x in tmp_concat]

    return np.transpose(xtrain_tmp), np.transpose(ytrain_tmp), train_sids, np.transpose(xtest_tmp), np.transpose(
        ytest_tmp), test_sids, all_classes


def train_model(train_proteins, train_snps, all_classes,skip_train=False):
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

        # Check for sICAM-1 manually
        # if i == 2:
        #     # FIGURE 1.a
        #     plot_pts = np.linspace(use_x.min(), use_x.max(), 200)
        #     preds = nb.predict_proba(plot_pts.reshape(-1, 1))
        #     plt.figure()
        #     plt.stackplot(plot_pts, preds.T, labels=nb.classes_)
        #     plt.xlim(plot_pts.min(), plot_pts.max())
        #     plt.ylim(0, 1)
        #     plt.legend(title="Genotype", fancybox=False, edgecolor="black", facecolor="white", framealpha=1.0)
        #     plt.xlabel("Log-transformed protein level for sICAM-1")
        #     plt.ylabel("Genotype Probability")
        #     plt.savefig("figs/sICAM_prob_cuml.png")
        #     plt.close()

        models.append(nb)
        #assert np.all(nb.classes_ == all_classes[i])
        class_orders.append({v: k for k, v in enumerate(all_classes[i])})
        class_priors.append(class_prior_i)
    return models, class_orders, class_priors


def predict_model(models, test_proteins,log_odds=False):
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


def eval_model(y_pred, y_true, class_orders, class_priors, num_proteins=100, memo_tag="default", log_odds=False):
    # For each row of y_true
    y_true_copy = y_true.copy()
    # Make copy of y_true and keep track of where NaNs are
    nan_idcs = (y_true == "nan")

    # List for all preds

    use_prev_result = False
    # Check if there is a memoization result we can use.
    if memo_tag in tag_dict:
        for key in sorted(tag_dict[memo_tag].keys(),reverse=True):
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

    for i in np.arange(start_i,num_proteins):
        print(i)
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
            tmp_preds = np.maximum(tmp_preds,np.finfo(np.float64).eps)
            tmp_preds = np.log(np.minimum(5, tmp_preds / (1 - tmp_preds + np.finfo(np.float64).eps)))
            tmp_preds[:, nan_idcs[i]] = 0
        else:
            #onehot[nan_idcs[i]] = 1 / (3 * np.ones(3))
            onehot[nan_idcs[i]] = np.nan
            # Get the probability that each subject has this genotype
            tmp_preds = np.matmul(y_pred[i], onehot.T)
            tmp_preds[:,nan_idcs[i]] = np.log(1/3)

        all_preds += tmp_preds
        #all_preds += tmp_preds
        del tmp_preds

    tag_dict[memo_tag][num_proteins] = all_preds
    #full_arr = np.stack(all_preds).transpose(1, 2, 0).sum(axis=-1)

    return all_preds


def get_top_k_accuracy(dist_mat, k=1, count_ties=False):
    # Output matrix will be
    out_mat = np.zeros((dist_mat.shape[0], k))
    # Iterate over all rows of matrix
    for i in range(dist_mat.shape[0]):
        # Get unique values for this row (This also sorts smallest to largest)
        unq_vals = np.unique(dist_mat[i, :])
        # Iterate over the smallest k values.
        for kv, unq in zip(range(k), unq_vals[:k]):
            k_idcs = np.squeeze(np.argwhere(dist_mat[i, :] == unq), axis=-1)
            # Check if the correct value occurs in the set at all
            if i in k_idcs:
                # In case where value is uniquely smallest at this k, we are done.
                if (k_idcs.shape[0] == 1 or count_ties):
                    out_mat[i, kv:] = 1
                # At this point, we can end the loop. If the value is uniquely the smallest (or we are counting ties),
                # then we have discovered the smallest k.
                # Otherwise, we discovered it, but it was not unique so we count a miss.
                break
    return out_mat


def logsumexp(v, axis=None):
    max_v = np.max(v, axis=axis, keepdims=True)
    centered = v - max_v
    return max_v + np.log(np.sum(np.exp(centered), axis=axis, keepdims=True))


def softmax_group(df):
    dft = df.copy()
    dft["prob"] = np.exp(dft["log_prob"].values - logsumexp(dft["log_prob"].values))
    return dft


def make_comparison_df(prob_matrix, sids, clinical, title=None, other_sids=None):
    fig, ax = plt.subplots(ncols=1,figsize=(12,120))
    count = 0
    ax = [ax, count]
    if other_sids is None:
        other_sid_ids = ["NP%d" % d for d in np.arange(len(sids), prob_matrix.shape[1])]
    else:
        other_sid_ids = other_sids
    sids_plus = np.concatenate([sids, other_sid_ids])

    disp_df = pd.DataFrame(prob_matrix, index=sids, columns=sids_plus)
    disp_df.index.name = "True Subject"

    top_match_df = pd.DataFrame(disp_df.values.argsort(axis=1)[:, ::-1], index=disp_df.index,
                 columns=["Guess_%d" % i for i in range(disp_df.shape[1])]).applymap(lambda val: disp_df.columns[val])

    top_match_df.to_csv("{}_actual_vs_predicted.csv".format(title.replace("_prob_dist","")))
    return

    disp_df.reset_index(inplace=True)
    disp_df = disp_df.melt(id_vars=["True Subject"], value_name="log_prob", var_name="Predicted Subject").join(clinical,
                                                                                                               on="Predicted Subject")
    #disp_df = disp_df.loc[disp_df["True Subject"].isin(["CU101800", "CU100195", "LA192182"])]
    disp_df["match"] = disp_df["True Subject"] == disp_df["Predicted Subject"]

    disp_df = disp_df.groupby("True Subject").apply(softmax_group)
    disp_df["marker_size"] = disp_df["match"].apply(lambda x: 5 if x else 2)
    # Order sids by probability of correct match.
    y_pad = 0.5
    y_jitter = 0.25
    np.random.seed(1)
    y_jitter_vals = np.random.uniform(-y_jitter, y_jitter, size=(disp_df.shape[0]))
    disp_df["y_jitter_val"] = y_jitter_vals
    sids_order = disp_df.loc[disp_df.match].sort_values("prob").loc[:, "True Subject"].values
    sids_range = np.arange(len(sids_order)) * (y_pad + (2 * y_jitter))
    sids_y_map = {sid: y_coord for sid, y_coord in zip(sids_order, sids_range)}

    disp_df["y_coords"] = disp_df.loc[:, ["True Subject", "y_jitter_val"]].apply(
        lambda x: sids_y_map[x["True Subject"]] + x["y_jitter_val"], axis=1)

    # Get actual vs predicted SIDs for the mismatches:
    pred_vs_actual = disp_df.groupby(level=0).apply(lambda df: {"Actual": df.index.get_level_values("True Subject")[0],
                                                                "Predicted": df.loc[df.prob == df.prob.max(), "Predicted Subject"].values[0]})
    pred_vs_actual = pd.DataFrame.from_records(pred_vs_actual)
    pred_vs_actual.to_csv("{}_actual_vs_predicted.csv".format(title.replace("_prob_dist","")))

    nonmatch_df = disp_df.loc[~disp_df["match"]]
    match_df = disp_df.loc[disp_df["match"]]
    ax[0].clear()
    #ax[0].scatter(np.log10(nonmatch_df["prob"]), nonmatch_df["y_coords"], color=tab10(0), s=2, label="Non-match")
    #ax[0].scatter(np.log10(match_df["prob"]), match_df["y_coords"], color=tab10(1), s=5, label="Match")
    #ax[0].set_xlim([-160, 1.1])
    xticks = ax[0].get_xticks()
    #ax[0].set_xticklabels(labels=['$10^{%d}$' % tick if tick != 0 else '$1$' for tick in xticks])
    #ax[0].set_yticks(ticks=sids_range)
    #ax[0].set_yticklabels(labels=["Subject %d" % (d + 1) for d in reversed(range(len(sids_order)))])
    #ax[0].set_yticklabels(labels=sids_order,fontsize=8)
    #ax[0].set_ylim([sids_range.min() - 0.5, sids_range.max() + 0.5])
    #ax[0].set_xlabel("Probability of genotype given a subject's proteome profile")
    #ax[0].legend(fancybox=False, edgecolor="black", loc="upper left", framealpha=1.0)
    plt.tight_layout()
    # PRINTS FIGURE 2
    plt.savefig("figs/{}.png".format(title))


def train_test_accuracy(train_prob_matrices, train_sids, train_clinical, test_prob_matrices, test_sids, test_clinical,
                        train_title, test_title, fname, count_ties=False, draw_probs=False, train_other_sids = None,
                        test_other_sids = None):
    # Prints Figure
    if draw_probs:
        make_comparison_df(test_prob_matrices, test_sids, test_clinical, test_title + "_prob_dist",
                           other_sids=test_other_sids)

    train_prob_matrix = train_prob_matrices
    test_prob_matrix = test_prob_matrices

    #two_pct_of_pool = int(np.ceil(0.02*train_prob_matrix.shape[1]))
    #test_two_pct_of_pool = int(np.ceil(0.02*test_prob_matrix.shape[1]))
    one_pct_cutoff = np.floor(train_prob_matrix.shape[1] * 0.01).astype(np.int32)

    train_top_k = get_top_k_accuracy(-train_prob_matrix, k=one_pct_cutoff, count_ties=count_ties)
    test_top_k = get_top_k_accuracy(-test_prob_matrix, k=one_pct_cutoff, count_ties=count_ties)

    np.save("train_%s_top_k%s.npy" % (train_title,adj_suffix), train_top_k)
    np.save("test_%s_top_k%s.npy" % (test_title,adj_suffix), test_top_k)
    train_all_top_k = train_top_k.sum(axis=0) / train_prob_matrix.shape[0]

    one_pct_cutoff = np.floor(train_prob_matrix.shape[1] * 0.01).astype(np.int32)
    tr_top_1 = train_all_top_k[0]
    tr_top_3 = train_all_top_k[2]
    tr_top_1pc = train_all_top_k[one_pct_cutoff - 1]

    print("Train Top 1 Accuracy: %f" % tr_top_1)
    print("Train Top 3 Accuracy %f" % tr_top_3)
    print("Train Top 1%% Accuracy %f (k=%d)" % (tr_top_1pc, one_pct_cutoff))

    train_nhw = train_top_k[train_clinical["race"] == 1]
    train_nhw_top_k = train_nhw.sum(axis=0) / train_nhw.shape[0]

    train_aa = train_top_k[train_clinical["race"] == 2]
    train_aa_top_k = train_aa.sum(axis=0) / train_aa.shape[0]
    test_all_top_k = test_top_k.sum(axis=0) / test_prob_matrix.shape[0]

    test_one_pct_cutoff = np.floor(test_prob_matrix.shape[1] * 0.01).astype(np.int32)
    ts_top_1 = test_all_top_k[0]
    ts_top_3 = test_all_top_k[2]
    ts_top_1pc = test_all_top_k[test_one_pct_cutoff - 1]

    print("Test Top 1 Accuracy: %f" % ts_top_1)
    print("Test Top 3 Accuracy %f" % ts_top_3)
    print("Test Top 1%% Accuracy %f (k=%d)" % (ts_top_1pc, test_one_pct_cutoff))

    test_nhw = test_top_k[test_clinical["race"] == 1]
    test_nhw_top_k = test_nhw.sum(axis=0) / test_nhw.shape[0]

    test_aa = test_top_k[test_clinical["race"] == 2]
    test_aa_top_k = test_aa.sum(axis=0) / test_aa.shape[0]

    test_other = test_top_k[np.logical_and(test_clinical["race"] != 1, test_clinical["race"] != 2)]
    test_other_top_k = test_other.sum(axis=0) / test_other.shape[0]

    fig, ax = plt.subplots(ncols=2, figsize=(20, 7))
    ax = ax.reshape(-1)

    train_ties = Counter((train_prob_matrix == train_prob_matrix[
        np.arange(train_prob_matrix.shape[0]), np.arange(train_prob_matrix.shape[0])].reshape(-1, 1)).sum(axis=-1))
    test_ties = Counter((test_prob_matrix == test_prob_matrix[
        np.arange(test_prob_matrix.shape[0]), np.arange(test_prob_matrix.shape[0])].reshape(-1, 1)).sum(axis=-1))

    print("Train ties: %s" % train_ties)
    print("Test ties: %s" % test_ties)

    # Train Data
    #ax[0].set_title("Top-K accuracy for '%s' dataset (Train Set)" % train_title)
    irange = np.concatenate([np.arange(0,one_pct_cutoff,4)])
    prange = irange+1
    #prange = np.arange(1, two_pct_of_pool+1,4)
    ax[0].plot(prange, train_all_top_k[irange], "o-",
               label="All Subjects (n={:d} of {:d} genotyped)".format(train_prob_matrix.shape[0],
                                                                      train_prob_matrix.shape[1]))
    ax[0].plot(prange, train_nhw_top_k[irange], "o-", label="NHW Subjects (n={:d})".format(train_nhw.shape[0]))
    ax[0].plot(prange, train_aa_top_k[irange], "o-", label="AA Subjects (n={:d})".format(train_aa.shape[0]))
    ax[0].plot(prange, prange / train_prob_matrix.shape[1], "o-", label="Random Guess", color="gray")
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("Accuracy")
    tick_range = np.concatenate([np.arange(1, one_pct_cutoff+1, 8)])
    ax[0].set_xticks(tick_range)
    ax[0].set_yticks(np.linspace(0, 1.0, 11))
    ax[0].yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax[0].legend(title="Race", fancybox=False, edgecolor="black")
    train_ax_pct = ax[0].twiny()
    train_norm_k = ["%0.2f%%" % (x * 100) for x in tick_range / train_prob_matrix.shape[1]]
    train_ax_pct.set_xticks(tick_range)
    train_ax_pct.xaxis.set_ticks_position("bottom")
    train_ax_pct.xaxis.set_label_position("bottom")
    train_ax_pct.spines["bottom"].set_position(("outward", 36))
    train_ax_pct.set_xlabel("% of Genotyped Subject Pool Size")
    train_ax_pct.set_xlim(ax[0].get_xlim())
    train_ax_pct.set_xticklabels(train_norm_k, rotation=45, ha="right")
    # Test Data
    #ax[1].set_title("Top-K accuracy for '%s' dataset (Test Set)" % test_title)
    ax[1].plot(prange, test_all_top_k[irange], "o-",
               label="All Subjects (n={:d} of {:d} genotyped)".format(test_prob_matrix.shape[0],
                                                                      test_prob_matrix.shape[1]))
    ax[1].plot(prange, test_nhw_top_k[irange], "o-", label="NHW Subjects (n={:d})".format(test_nhw.shape[0]))
    ax[1].plot(prange, test_aa_top_k[irange], "o-", label="AA Subjects (n={:d})".format(test_aa.shape[0]))
    ax[1].plot(prange, test_other_top_k[irange], "o-", label="Other Race (n={:d})".format(test_other.shape[0]))
    ax[1].plot(prange, prange / test_prob_matrix.shape[1], "o-", label="Random Guess", color="gray")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xticks(tick_range)
    ax[1].set_yticks(np.linspace(0, 1.0, 11))
    ax[1].yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax[1].legend(title="Race", fancybox=False, edgecolor="Black")
    test_ax_pct = ax[1].twiny()
    test_norm_k = ["%0.2f%%" % (x * 100) for x in tick_range / test_prob_matrix.shape[1]]
    test_ax_pct.set_xticks(tick_range)
    test_ax_pct.xaxis.set_ticks_position("bottom")
    test_ax_pct.xaxis.set_label_position("bottom")
    test_ax_pct.spines["bottom"].set_position(("outward", 36))
    test_ax_pct.set_xlabel("% of Genotyped Subject Pool Size")
    test_ax_pct.set_xlim(ax[1].get_xlim())
    test_ax_pct.set_xticklabels(test_norm_k, rotation=45, ha="right")
    plt.tight_layout()
    # PRINTS FIGURE 3
    plt.savefig("figs/" + fname)
    plt.close()

    return tr_top_1, tr_top_3, tr_top_1pc, ts_top_1, ts_top_3, ts_top_1pc


def prediction_matrix(train_prob_matrix, test_prob_matrix, train_title, test_title, fname):
    fig, ax = plt.subplots(ncols=2, figsize=(26, 12))

    # Train Data
    ax[0].set_title("Predicted vs Actual Subject '%s' dataset (Train Set)" % train_title)
    ax[0].imshow(train_prob_matrix[:300, :300], cmap="magma")
    ax[0].set_xlabel("Predicted Subject")
    ax[0].set_ylabel("Actual Subject")

    # Test Data
    ax[1].set_title("Predicted vs Actual Subject '%s' dataset (Test Set)" % test_title)
    ax[1].imshow(test_prob_matrix[:300, :300], cmap="magma")
    ax[1].set_xlabel("Predicted Subject")
    ax[1].set_ylabel("Actual Subject")

    plt.savefig("figs/" + fname)

    plt.close()


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
    return load_proteins_snps_clinical(COPDGene_P1_JHS_PROTEINS, COPDGene_P1_JHS_SNPS, COPDGene_P1_JHS_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P1_JHS_AGE_COL,
                                       clin_race_col=COPDGene_P1_JHS_RACE_COL, clin_gender_col=COPDGene_P1_JHS_GENDER_COL)

def load_copdgene_p1_jhs_only():
    return load_proteins_snps_clinical(COPDGene_P1_JHS_ONLY_PROTEINS, COPDGene_P1_JHS_ONLY_SNPS, COPDGene_P1_JHS_ONLY_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P1_JHS_ONLY_AGE_COL,
                                       clin_race_col=COPDGene_P1_JHS_ONLY_RACE_COL, clin_gender_col=COPDGene_P1_JHS_ONLY_GENDER_COL)

def load_copdgene_p2():
    return load_proteins_snps_clinical(COPDGene_P2_PROTEINS, COPDGene_P2_SNPS, COPDGene_P2_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P2_AGE_COL,
                                       clin_race_col=COPDGene_P2_RACE_COL, clin_gender_col=COPDGene_P2_GENDER_COL)

def load_copdgene_p2_jhs():
    return load_proteins_snps_clinical(COPDGene_P2_JHS_PROTEINS, COPDGene_P2_JHS_SNPS, COPDGene_P2_JHS_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P2_JHS_AGE_COL,
                                       clin_race_col=COPDGene_P2_JHS_RACE_COL, clin_gender_col=COPDGene_P2_JHS_GENDER_COL)

def load_copdgene_p1_p2():
    p1_protein, p1_snp, p1_clin, p1_other_snp = load_copdgene_p1()
    p2_protein, p2_snp, p2_clin, p2_other_snp = load_copdgene_p2()

    comb_protein = pd.concat([p1_protein,p2_protein],axis=0)
    comb_snp = pd.concat([p1_snp,p2_snp],axis=0)
    comb_clin = pd.concat([p1_clin,p2_clin],axis=0)
    # Get only SNPs which appear in both p1 and p2 'other' SNPs (i.e. don't appear in either of the original sets.
    other_snp_index = set(p1_other_snp.index).intersection(p2_other_snp.index)
    comb_other_snp = p1_other_snp.loc[other_snp_index]

    assert(len(other_snp_index.intersection(comb_snp.index)) == 0)
    return comb_protein,comb_snp,comb_clin,comb_other_snp

def load_spiromics():
    return load_proteins_snps_clinical(SPIROMICS_PROTEINS, SPIROMICS_SNPS, SPIROMICS_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_sep=",", clin_age_col=SPIROMICS_AGE_COL,
                                       clin_race_col=SPIROMICS_RACE_COL, clin_gender_col=SPIROMICS_GENDER_COL)

def load_spiromics_jhs():
    return load_proteins_snps_clinical(SPIROMICS_JHS_PROTEINS, SPIROMICS_JHS_SNPS, SPIROMICS_JHS_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_sep=",", clin_age_col=SPIROMICS_JHS_AGE_COL,
                                       clin_race_col=SPIROMICS_JHS_RACE_COL, clin_gender_col=SPIROMICS_JHS_GENDER_COL)

def load_spiromics_jhs_only():
    return load_proteins_snps_clinical(SPIROMICS_JHS_ONLY_PROTEINS, SPIROMICS_JHS_ONLY_SNPS, SPIROMICS_JHS_ONLY_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_sep=",", clin_age_col=SPIROMICS_JHS_ONLY_AGE_COL,
                                       clin_race_col=SPIROMICS_JHS_ONLY_RACE_COL, clin_gender_col=SPIROMICS_JHS_ONLY_GENDER_COL)

def load_copdgene_p2_5k():
    return load_proteins_snps_clinical(COPDGene_P2_5K_PROTEINS, COPDGene_P2_5K_SNPS, COPDGene_P2_5K_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P2_5K_AGE_COL,
                                       clin_race_col=COPDGene_P2_5K_RACE_COL, clin_gender_col=COPDGene_P2_5K_GENDER_COL)

def load_copdgene_p2_5k_all():
    return load_proteins_snps_clinical(COPDGene_P2_5K_ALL_PROTEINS, COPDGene_P2_5K_ALL_SNPS, COPDGene_P2_5K_ALL_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P2_5K_ALL_AGE_COL,
                                       clin_race_col=COPDGene_P2_5K_ALL_RACE_COL, clin_gender_col=COPDGene_P2_5K_ALL_GENDER_COL)

def load_copdgene_p3_5k_all():
    return load_proteins_snps_clinical(COPDGene_P3_5K_ALL_PROTEINS, COPDGene_P3_5K_ALL_SNPS, COPDGene_P3_5K_ALL_CLINICAL, protein_sep="\t",
                                       snp_sep=",", clin_age_col=COPDGene_P3_5K_ALL_AGE_COL,
                                       clin_race_col=COPDGene_P3_5K_ALL_RACE_COL, clin_gender_col=COPDGene_P3_5K_ALL_GENDER_COL)

# Loads the training and testing datasets based on the flags from args.
def load_data(args):
    train_data_arg = args.train_data
    test_data_arg = args.test_data

    # Select the training dataset.
    if train_data_arg == COPDGene_P1_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p1()
    elif train_data_arg == COPDGene_P2_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p2()
    elif train_data_arg == COPDGene_P2_JHS_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p2_jhs()
    elif train_data_arg == COPDGene_P1_P2_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p1_p2()
    elif train_data_arg == COPDGene_P2_5K_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p2_5k()
    elif train_data_arg == COPDGene_P2_5K_ALL_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p2_5k_all()
    elif train_data_arg == COPDGene_P3_5K_ALL_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p3_5k_all()
    elif train_data_arg == SPIROMICS_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_spiromics()
    elif train_data_arg == COPDGene_P1_JHS_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p1_jhs()
    elif train_data_arg == SPIROMICS_JHS_PROTEINS:
        train_p ,train_snp, train_clin, train_o_snp = load_spiromics_jhs()
    elif train_data_arg == COPDGene_P1_JHS_ONLY_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_copdgene_p1_jhs_only()
    elif train_data_arg == SPIROMICS_JHS_ONLY_NAME:
        train_p, train_snp, train_clin, train_o_snp = load_spiromics_jhs_only()
    else:
        raise ValueError("bad dataset name.")

    print("Loaded %s as training set." % train_data_arg)

    if train_data_arg == test_data_arg:
        test_p, test_snp, test_clin, test_o_snp = (train_p, train_snp, train_clin, train_o_snp)
    else:
        # Select the testing dataset.
        if test_data_arg == COPDGene_P1_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_copdgene_p1()
        elif test_data_arg == COPDGene_P2_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_copdgene_p2()
        elif test_data_arg == COPDGene_P2_JHS_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_copdgene_p2_jhs()
        elif test_data_arg == COPDGene_P1_P2_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_copdgene_p1_p2()
        elif test_data_arg == COPDGene_P2_5K_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_copdgene_p2_5k()
        elif test_data_arg == SPIROMICS_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_spiromics()
        elif test_data_arg == SPIROMICS_JHS_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_spiromics_jhs()
        elif test_data_arg == COPDGene_P1_JHS_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_copdgene_p1_jhs()
        elif test_data_arg == COPDGene_P1_JHS_ONLY_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_copdgene_p1_jhs_only()
        elif test_data_arg == SPIROMICS_JHS_ONLY_NAME:
            test_p, test_snp, test_clin, test_o_snp = load_spiromics_jhs_only()
        else:
            raise ValueError("bad dataset name.")
    print("Loaded %s as testing set." % test_data_arg)

    # Check for overlap between train and test and correct (Only for COPDGene)
    intersect_sids = set(train_p.index).intersection(test_p.index)
    print("Found %d intersecting SIDs." % len(intersect_sids))

    return train_data_arg, train_p, train_snp, train_clin, train_o_snp, test_data_arg, test_p, test_snp, test_clin, test_o_snp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, choices=["COPDGene_P1",
                                                           "COPDGene_P1_JHS",
                                                           "COPDGene_P1_JHS_ONLY",
                                                           "COPDGene_P2",
                                                           "COPDGene_P2_JHS",
                                                           "COPDGene_P1_P2",
                                                           "COPDGene_P2_5K",
                                                           "COPDGene_P2_5K_ALL",
                                                           "COPDGene_P3_5K_ALL",
                                                           "SPIROMICS",
                                                           "SPIROMICS_JHS",
                                                           "SPIROMICS_JHS_ONLY"])
    parser.add_argument("--test_data", type=str, choices=["COPDGene_P1",
                                                          "COPDGene_P1_JHS",
                                                          "COPDGene_P1_JHS_ONLY",
                                                          "COPDGene_P2",
                                                          "COPDGene_P2_JHS",
                                                          "COPDGene_P1_P2",
                                                          "COPDGene_P2_5K",
                                                          "COPDGene_P2_5K_ALL",
                                                          "COPDGene_P3_5K_ALL",
                                                          "SPIROMICS",
                                                          "SPIROMICS_JHS",
                                                          "SPIROMICS_JHS_ONLY"])
    parser.add_argument("--use_pqtls", nargs="+", type=int, help="Use only the top N pQTLs (sorted by FDR). Pass a single value or a set of values to test.",default=[100])
    parser.add_argument("--mean_adjust",action="store_true")
    parser.add_argument("--log_odds",action="store_true")
    parser.add_argument("--skip_train",action="store_true")
    args = parser.parse_args()

    adj_suffix = "_adj" if args.mean_adjust else ""
    log_odds_suffix = "_lg" if args.log_odds else ""
    # Step 1: Load train/test data.
    train_data, train_proteins, train_snps, train_clinical, train_other_snps, test_data, test_proteins, test_snps, test_clinical, test_other_snps = load_data(
        args)
    # Step 2: Read pQTL list.
    pqtls = pd.read_csv(PAIR_LIST)
    # Step 2.5: Sort pQTL list by abs. val of effect size (for using less that 144 SNPs).
    # Absolute value of effect size beta
    #pqtls["abs_beta_p1"] = np.abs(pqtls["FDR"])
    # Sort by this value.
    sort_pqtls = pqtls.sort_values("p-value", ascending=True)
    # top_n = sort_pqtls.iloc[:args.use_pqtls[0]]
    # n_copdgene_pqtls = len(top_n.loc[sort_pqtls.set == "COPDGene_P1"])
    # n_jhs_pqtls = len(top_n.loc[sort_pqtls.set == "JHS"])
    # assert (n_copdgene_pqtls + n_jhs_pqtls) == len(top_n)
    # print("There are %d pQTLs from COPDGene P1, and %d pQTLs from JHS." % (n_copdgene_pqtls,n_jhs_pqtls))
    #sort_pqtls = sort_pqtls.loc[sort_pqtls.set == "COPDGene_P1"]
    # Drop c-Jun, which is problematic
    sort_pqtls = sort_pqtls.drop(sort_pqtls.index[sort_pqtls.gene == "c-Jun"])
    # TEMPORARY, DELETE ME
    #reduced_list = pd.read_csv("reduced_set_selection_jhs.csv",header=0,squeeze=True)
    #sort_pqtls = sort_pqtls.loc[sort_pqtls.gene.isin(reduced_list)]
    #use_pqtls = sort_pqtls.iloc[:i]
    pqtl_pairs = [(x, y) for x, y in zip(sort_pqtls.SNP, sort_pqtls.gene)]

    # Step 3: Convert raw data into training and test datasets with x as protein measurements and y as genotype.
    x_train, y_train, train_sids, x_test, y_test, test_sids, all_classes = get_train_test_2(pqtl_pairs, train_proteins,
                                                                                            train_snps, test_proteins,
                                                                                            test_snps,
                                                                                            align_to_reference=args.mean_adjust,
                                                                                            train_other_snps = train_other_snps,
                                                                                            test_other_snps = test_other_snps)
    #x_train = np.random.random(x_train.shape)
    #x_test = np.random.random(x_test.shape)
    # Step 4: Train model using training dataset.
    trained_model, class_order, class_prior = train_model(x_train, y_train, all_classes,skip_train=args.skip_train)


    # Step 4.5: Dump out trained model in pickle (binary) format so that we can predict without using the training data.
    ref_snps = pd.read_csv(REFERENCE_SNPS, index_col=0)
    #sort_pqtls["gene"] = sort_pqtls["gene_JHS"]
    dump_obj = {"model":trained_model,
                "class_order":class_order,
                "class_prior":class_prior,
                "sort_pqtls":sort_pqtls,
                "ref_snps":ref_snps,
                "all_classes":all_classes,
                "is_trained_model":not args.skip_train}

    with open("frozen_model.pkl","wb") as write_file:
        pickle.dump(dump_obj,write_file)
    # Step 5: Generate prediction probabilities for all 3 genotype classes at each pQTL.
    # Do this for both training and testing datasets.
    train_preds = predict_model(trained_model, x_train,log_odds=args.log_odds)
    # Don't predict twice if using the same dataset
    if test_data != train_data:
        test_preds = predict_model(trained_model, x_test,log_odds=args.log_odds)
    else:
        test_preds = train_preds

    # Step 5.5: Add in the genotypes of subjects without protein measurements to increase the pool size.
    #y_train = np.concatenate([y_train, train_other_snps.loc[:, [p[0] for p in pqtl_pairs]].values.T], axis=-1)
    #y_test = np.concatenate([y_test, test_other_snps.loc[:, [p[0] for p in pqtl_pairs]].values.T], axis=-1)

    train_other_sids = np.array(train_other_snps.index)
    test_other_sids = np.array(test_other_snps.index)

    # Dictionary for memoizing results
    tag_dict = {}

    # Since pQTLs are already sorted, we can just use the first N pQTLs to get the performance for using smaller numbers
    # of pQTLs
    results_index = args.use_pqtls
    results_list = []

    # TEMP OUTPUTS FOR FEATURE SELECTION
    np.save("COPDGene_P1_train_preds.npy",train_preds)
    np.save("COPDGene_P1_train_y.npy",y_train)
    with open("COPDGene_P1_train_corder.pkl","wb") as f:
        pickle.dump(class_order,f)

    for res_i in results_index:
        print("Running with i=%d" % res_i)
        # Step 6: For each known protein palette, generate a probability score of that protein palette arising from each
        # genotype vector in the pool.
        train_prob_m= eval_model(train_preds, y_train, class_order, class_prior, num_proteins=res_i, memo_tag=train_data, log_odds=args.log_odds)
        if test_data != train_data:
            test_prob_m = eval_model(test_preds, y_test, class_order, class_prior, num_proteins=res_i, memo_tag=test_data,log_odds=args.log_odds)
        else:
            test_prob_m = train_prob_m

        # Compute the train/test accuracy and charts for performance of the model.
        row = train_test_accuracy(train_prob_m, train_sids, train_clinical, test_prob_m, test_sids, test_clinical,
                                  train_title=train_data, test_title=test_data,
                                  fname="train_%s_test_%s_%d%s%s_proteins_accuracy.png" % (train_data, test_data, res_i, adj_suffix,log_odds_suffix),
                                  draw_probs=False, test_other_sids = test_other_sids, train_other_sids = train_other_sids)
        results_list.append(row)

    out_df = pd.DataFrame.from_records(results_list,index=results_index,columns=["Train Top-1 Accuracy",
                                                                                 "Train Top 3 Accuracy",
                                                                                 "Train Top 1%% Accuracy",
                                                                                 "Test Top 1 Accuracy",
                                                                                 "Test Top 3 Accuracy",
                                                                                 "Test Top 1%% Accuracy"])
    out_df.to_csv("test_results_{}_{}{}{}.csv".format(train_data,test_data,adj_suffix,log_odds_suffix))