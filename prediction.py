import pandas as pd
import numpy as np
import pickle
import argparse
import datetime
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm
from matplotlib import pyplot as plt
from os.path import isdir, join, split
from os import makedirs, getcwd
import seaborn as sns


def load_files(snp_path, protein_path, use_pqtls, snp_sep, protein_sep):
    print("Reading SNP file '%s'..." % snp_path)
    snps = pd.read_csv(snp_path, sep=snp_sep, index_col=0, keep_default_na=False)
    print("Finished reading SNP file.")
    print("Reading protein file '%s'..." % protein_path)
    prots = pd.read_csv(protein_path, sep=protein_sep, index_col=0)
    print("Finished reading protein file.")

    intersect_sids = set(snps.columns) & set(prots.index)
    int_sid_list = list(intersect_sids)
    print("Found %d intersecting SIDs between protein and SNP files." % len(intersect_sids))
    background_sids = set(snps.columns) - intersect_sids
    back_sid_list = list(background_sids)
    print("The remaining %d SIDs without protein data will be used as background." % len(background_sids))

    # use_snps = snps.loc[]
    # Check how many SNPs overlap between the use list and the given dataset.
    avail_snps = use_pqtls.SNP.isin(snps.index)
    avail_snps_num = avail_snps.sum()
    print("%d out of %d desired SNPs are available in the dataset." % (avail_snps_num, len(use_pqtls)))
    # We need to grab this so that we can return it just in case not all SNPs are available.
    avail_snps_lg = np.argwhere(use_pqtls.SNP.isin(snps.index).values).reshape(-1)
    use_pqtls_reduced = use_pqtls.loc[use_pqtls.SNP.isin(snps.index)]

    # Align the data
    snps_out = snps.loc[use_pqtls_reduced.SNP, int_sid_list]
    snps_out_background = snps.loc[use_pqtls_reduced.SNP, back_sid_list]
    snps_out = pd.concat([snps_out, snps_out_background], axis=1)

    # Log transform the proteins
    proteins_out = np.log(1 + prots.loc[int_sid_list, use_pqtls_reduced.gene])

    return snps_out, proteins_out, avail_snps_lg


def predict_genotype(model, prots):
    total_preds = []
    for i in range(len(model)):
        nb = model[i]
        prots_np = prots.values.T
        plc = np.full(shape=(prots_np[0].shape[0], 3), fill_value=np.finfo(np.float32).min, dtype=np.float32)
        preds = nb._joint_log_likelihood(prots_np[i, :, np.newaxis])

        # Enumerate classes used if all three classes aren't here.
        for i, arg_idx in enumerate(np.argsort(nb.classes_)):
            plc[:, arg_idx] = preds[:, i]

        total_preds.append(plc)
    return np.stack(total_preds, axis=0)


def eval_model(y_pred, y_true, class_orders):
    y_true = y_true.values
    # For each row of y_true
    y_true_copy = y_true.copy()
    # Make copy of y_true and keep track of where NaNs are
    nan_idcs = (y_true == "nan")

    # List for all preds
    all_preds = []
    for i in range(y_pred.shape[0]):
        # Get class orders
        class_order = class_orders[i]
        # Set nan values to be a dummy.
        y_true_copy[i, nan_idcs[i]] = sorted(list(class_order.keys()))[0]
        # Get indices
        ind_list = np.array([class_order[x] for x in y_true_copy[i]])
        # One-hot encoded
        onehot = np.eye(3)[ind_list]
        onehot[nan_idcs[i]] = 1 / (3 * np.ones(3))
        # Get the probability that each subject has this genotype
        tmp_preds = np.matmul(y_pred[i], onehot.T)
        all_preds.append(tmp_preds)
    full_arr = np.stack(all_preds).transpose(1, 2, 0)
    return full_arr


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


def get_outputs(prob_matrix, sids):
    ideal_list = [20, 40, 60, 100, 144, 223]
    # Assemble list of #s of proteins to test
    test_prot_max = prob_matrix.shape[-1]

    actual_list = [v for v in ideal_list if test_prot_max > v] + [test_prot_max]
    one_pct = np.minimum(int(np.floor(prob_matrix.shape[1] * 0.01)), prob_matrix.shape[0])

    rows = []
    for test_v in actual_list:
        temp_prob_matrix = prob_matrix[:, :, :test_v].sum(axis=-1)
        tmp_top_k = get_top_k_accuracy(-temp_prob_matrix, k=np.minimum(temp_prob_matrix.shape[0], 100))
        tmp_top_k_pct = tmp_top_k.sum(axis=0) / tmp_top_k.shape[0]

        top_1 = tmp_top_k_pct[0]
        top_3 = tmp_top_k_pct[2]
        top_1pct = tmp_top_k_pct[one_pct - 1]

        rows.append({"K": test_v,
                     "Top 1 Acc.": top_1,
                     "Top 3 Acc.": top_3,
                     "Top 1% ({}) Acc.".format(one_pct): top_1pct})

    out_df = pd.DataFrame.from_records(rows)
    return out_df


def run_qc(model, prots, snps, sort_pqtls, used_pqtls):
    qc_df_list = []
    if not isdir(join(getcwd(), "qc")):
        print("Making QC directory...")
        makedirs("qc")
    used_pqtls_new = list(used_pqtls)
    for i in used_pqtls:
        md = model[i]
        snp = sort_pqtls["SNP"].iloc[i]
        prot = sort_pqtls["gene"].iloc[i]

        snp_series = snps.loc[snp, :]
        prot_series = prots.loc[:, [prot]]

        join_df = prot_series.join(snp_series, how="inner")
        join_df = join_df.loc[join_df[snp] != 'nan']

        test_classes = set(join_df[snp].unique())
        test_classes.discard("nan")

        train_classes = set(md.classes_)

        if test_classes != train_classes:
            if len(test_classes - train_classes) == 0:
                print("Warning: SNP '%s' is missing an expected SNP value. This is likely normal." % snp)
            else:
                print("ERROR: SNP '%s' string values are different than expected.\nExpected: %s\nReceived: %s" % (snp,
                                                                                                                  train_classes,
                                                                                                                  test_classes))
                print("Removing this SNP from the usable pQTLs list since this will cause prediction to fail.")
                used_pqtls_new.remove(i)

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.histplot(data=join_df, x=prot, hue=snp, stat="density", hue_order=md.classes_, ax=ax)

        lg = ax.get_legend()
        hist_handles = lg.legendHandles
        hist_labels = [t._text for t in lg.texts]
        model_rvs = dict()
        trained_qc_rows = []
        for j, cls in enumerate(md.classes_):
            ### Get mean/stddev from trained model.
            mean = md.theta_[j, 0]
            var = md.sigma_[j, 0]
            std = np.sqrt(var)

            ### Plot QC Histograms
            rv = norm(loc=mean, scale=std)
            xrange = np.linspace(rv.ppf(0.001), rv.ppf(0.999), num=100)
            ax.plot(xrange, rv.pdf(xrange), label="{} NB Estimate".format(cls))

            model_rvs[cls] = rv

            trained_qc_rows.append({"Class": cls,
                                    "Train Data Mean": mean,
                                    "Train Data Var": var})

        plot_handles, plot_labels = ax.get_legend_handles_labels()
        ax.legend(handles=hist_handles + plot_handles, labels=hist_labels + plot_labels)
        fig.savefig("qc/plot_%s_%s.png" % (prot, snp))
        plt.close(fig)

        test_qc_df = join_df.groupby(snp) \
            .agg(["mean", "var"]) \
            .droplevel(level=0, axis="columns") \
            .rename(columns={"mean": "Test Data Mean", "var": "Test Data Var"})

        train_qc_df = pd.DataFrame.from_records(trained_qc_rows, index="Class")
        comb_qc_df = pd.concat([train_qc_df, test_qc_df], axis=1).reset_index().rename(columns={"index": "Class"})
        comb_qc_df["SNP"] = snp
        comb_qc_df["Protein"] = prot
        comb_qc_df["Train Mean Order"] = comb_qc_df["Train Data Mean"].argsort()
        comb_qc_df["Test Mean Order"] = comb_qc_df["Test Data Mean"].argsort()
        qc_df_list.append(comb_qc_df)

    all_qc_df = pd.concat(qc_df_list).set_index(["SNP", "Protein", "Class"])

    return np.array(used_pqtls_new), all_qc_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snp_file", required=True, type=str,
                        help="Path to the file containing SNPs for each subject. Each row should correspond to a SNP, with subjects in columns.")
    parser.add_argument("--snp_sep", type=str, default=",")
    parser.add_argument("--protein_file", required=True, type=str,
                        help="Path to the file containing protein levels. Each row should correspond to a subject, with proteins in columns.")
    parser.add_argument("--protein_sep", type=str, default="\t")
    parser.add_argument("--use_pqtls", default=100, type=int,
                        help="Use the top N pQTLs from the complete set. Default is 100.")
    parser.add_argument("--output_file",
                        default="geno_pred_results_{}.csv".format(datetime.datetime.now().strftime("%h%d_%Y")),
                        type=str, help="Name of the output file to write.")
    parser.add_argument("--skip_qc", action="store_true", help="Include this flag to skip the QC step.")

    args = parser.parse_args()
    print("Loading model from file...")
    with open("frozen_model.pkl", "rb") as model_file:
        model_all = pickle.load(model_file)
    print("Model loaded successfully.")
    model = model_all["model"]
    class_order = model_all["class_order"]
    class_prior = model_all["class_prior"]
    sort_pqtls = model_all["sort_pqtls"]
    pqtl_pairs = [(x, y) for x, y in zip(sort_pqtls.SNP, sort_pqtls.gene)]

    snps, prots, used_pqtls = load_files(snp_path=args.snp_file,
                                         protein_path=args.protein_file,
                                         use_pqtls=sort_pqtls,
                                         snp_sep=args.snp_sep,
                                         protein_sep=args.protein_sep)

    if not args.skip_qc:
        used_pqtls, qc_df = run_qc(model=model,
                                   prots=prots,
                                   snps=snps,
                                   sort_pqtls=sort_pqtls,
                                   used_pqtls=used_pqtls)
        qc_df.to_excel("qc/qc_results.xlsx")

    snps = snps.iloc[used_pqtls]
    prots = prots.iloc[:,used_pqtls]

    model = [model[x] for x in used_pqtls]
    class_order = [class_order[x] for x in used_pqtls]

    prediction_matrix = predict_genotype(model=model,
                                         prots=prots)

    prob_matrix = eval_model(y_pred=prediction_matrix,
                             y_true=snps,
                             class_orders=class_order)

    output_df = get_outputs(prob_matrix=prob_matrix,
                            sids=np.array(prots.index))

    output_df.to_csv(args.output_file, index=False)
    print("Wrote results to '%s'." % args.output_file)
