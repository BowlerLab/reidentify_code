import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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

def train_test_accuracy(train_prob_matrices, train_sids, train_clinical, test_prob_matrices, test_sids, test_clinical,
                        train_title, test_title, fname, count_ties=False, draw_probs=False, train_other_sids=None,
                        test_other_sids=None,suffix=""):
    # Prints Figure
    if draw_probs:
        make_comparison_df(test_prob_matrices, test_sids, test_clinical, test_title + "_prob_dist",
                           other_sids=test_other_sids)

    train_prob_matrix = train_prob_matrices
    test_prob_matrix = test_prob_matrices

    # two_pct_of_pool = int(np.ceil(0.02*train_prob_matrix.shape[1]))
    # test_two_pct_of_pool = int(np.ceil(0.02*test_prob_matrix.shape[1]))
    one_pct_cutoff = np.floor(train_prob_matrix.shape[1] * 0.01).astype(np.int32)

    train_top_k = get_top_k_accuracy(-train_prob_matrix, k=one_pct_cutoff, count_ties=count_ties)
    test_top_k = get_top_k_accuracy(-test_prob_matrix, k=one_pct_cutoff, count_ties=count_ties)

    np.save("train_%s_top_k%s.npy" % (train_title, suffix), train_top_k)
    np.save("test_%s_top_k%s.npy" % (test_title, suffix), test_top_k)
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
    # ax[0].set_title("Top-K accuracy for '%s' dataset (Train Set)" % train_title)
    irange = np.concatenate([np.arange(0, one_pct_cutoff, 4)])
    prange = irange + 1
    # prange = np.arange(1, two_pct_of_pool+1,4)
    ax[0].plot(prange, train_all_top_k[irange], "o-",
               label="All Subjects (n={:d} of {:d} genotyped)".format(train_prob_matrix.shape[0],
                                                                      train_prob_matrix.shape[1]))
    ax[0].plot(prange, train_nhw_top_k[irange], "o-", label="NHW Subjects (n={:d})".format(train_nhw.shape[0]))
    ax[0].plot(prange, train_aa_top_k[irange], "o-", label="AA Subjects (n={:d})".format(train_aa.shape[0]))
    ax[0].plot(prange, prange / train_prob_matrix.shape[1], "o-", label="Random Guess", color="gray")
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("Accuracy")
    tick_range = np.concatenate([np.arange(1, one_pct_cutoff + 1, 8)])
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
    # ax[1].set_title("Top-K accuracy for '%s' dataset (Test Set)" % test_title)
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


def make_comparison_df(prob_matrix, sids, clinical, title=None, other_sids=None):
    from matplotlib.cm import tab10
    #fig, ax = plt.subplots(ncols=1, figsize=(12, 120))
    fig,ax = plt.subplots(ncols=1,figsize=(7,5))
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
                                columns=["Guess_%d" % i for i in range(disp_df.shape[1])]).applymap(
        lambda val: disp_df.columns[val])

    #top_match_df.to_csv("{}_actual_vs_predicted.csv".format(title.replace("_prob_dist", "")))
    #return

    disp_df.reset_index(inplace=True)
    disp_df = disp_df.melt(id_vars=["True Subject"], value_name="log_prob", var_name="Predicted Subject").join(clinical,
                                                                                                               on="Predicted Subject")
    disp_df = disp_df.loc[disp_df["True Subject"].isin(["CU101800", "CU100195", "LA192182"])]
    disp_df["match"] = disp_df["True Subject"] == disp_df["Predicted Subject"]

    disp_df = disp_df.groupby("True Subject").apply(identity)
    disp_df["prob"] = np.exp(disp_df["log_prob"])
    #disp_df.set_index("True Subject",inplace=True)
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
                                                                "Predicted": df.loc[
                                                                    df.prob == df.prob.max(), "Predicted Subject"].values[
                                                                    0]})
    pred_vs_actual = pd.DataFrame.from_records(pred_vs_actual)
    pred_vs_actual.to_csv("{}_actual_vs_predicted.csv".format(title.replace("_prob_dist", "")))

    nonmatch_df = disp_df.loc[~disp_df["match"]]
    match_df = disp_df.loc[disp_df["match"]]
    ax[0].clear()
    ax[0].scatter(np.log10(nonmatch_df["prob"]), nonmatch_df["y_coords"], color=tab10(0), s=2, label="Non-match")
    ax[0].scatter(np.log10(match_df["prob"]), match_df["y_coords"], color=tab10(1), s=5, label="Match")
    ax[0].set_xlim([-120, 1.1])
    xticks = ax[0].get_xticks()
    ax[0].set_xticklabels(labels=['$10^{%d}$' % tick if tick != 0 else '$1$' for tick in xticks])
    ax[0].set_yticks(ticks=sids_range)
    ax[0].set_yticklabels(labels=["Subject %d" % (d + 1) for d in reversed(range(len(sids_order)))])
    #ax[0].set_yticklabels(labels=sids_order,fontsize=8)
    ax[0].set_ylim([sids_range.min() - 0.5, sids_range.max() + 0.5])
    ax[0].set_xlabel("Log odds of genotype matching a subject's proteome profile")
    ax[0].legend(fancybox=False, edgecolor="black", loc="upper left", framealpha=1.0)
    plt.tight_layout()
    # PRINTS FIGURE 2
    plt.savefig("figs/{}.png".format(title))


def logsumexp(v, axis=None):
    max_v = np.max(v, axis=axis, keepdims=True)
    centered = v - max_v
    return max_v + np.log(np.sum(np.exp(centered), axis=axis, keepdims=True))


def softmax_group(df):
    dft = df.copy()
    dft["prob"] = np.exp(dft["log_prob"].values - logsumexp(dft["log_prob"].values))
    return dft

def identity(df):
    dft = df.copy()
    return dft

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