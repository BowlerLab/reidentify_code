import pandas as pd
import numpy as np
import pickle
import argparse
import datetime
from sklearn.naive_bayes import GaussianNB
from matplotlib.ticker import AutoMinorLocator,MultipleLocator
from scipy.stats import norm
from matplotlib import pyplot as plt
from os.path import isdir, join, split
from os import makedirs, getcwd
from metrics import train_test_accuracy,make_comparison_df
VERSION = 0.1


def load_files(snp_path, protein_path, use_pqtls, snp_sep, protein_sep, clin_file):
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
    print(
        "The remaining %d SIDs without protein data will be used as background for prediction." % len(background_sids))

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

    # Clinical data
    if clin_file is not None:
        clin_out = pd.read_csv(clin_file, index_col=0)
        clin_out.columns = [x.lower() for x in clin_out.columns]
        clin_out = clin_out.loc[proteins_out.index, ["race"]]
        clin_out["race"] = clin_out["race"].replace(
            {1: "NHW", 2: "AA", 3: "Other", 4: "Other", 5: "Other", 6: "Other", 7: "Other"})
    else:
        clin_out = None

    return snps_out, proteins_out, avail_snps_lg, clin_out


# Function to predict probabilities for each genotype class.
def predict_genotype(model, prots):
    total_preds = []
    prots_np = prots.values.T
    for i in range(len(model)):
        nb = model[i]
        plc = np.full(shape=(prots_np[0].shape[0], 3), fill_value=np.finfo(np.float32).min, dtype=np.float32)
        preds = nb.predict_proba(prots_np[i, :, np.newaxis])
        # Enumerate classes used if all three classes aren't here.
        for i, arg_idx in enumerate(np.argsort(nb.classes_)):
            plc[:, arg_idx] = preds[:, i]

        total_preds.append(plc)
    return np.stack(total_preds, axis=0)


def train_model(model, snps, prots, class_orders):
    assert len(model) == len(prots.columns)
    assert len(model) == len(snps)

    for i in range(len(model)):
        nb = model[i]

        # Temporary dataframe to hold protein values.
        tmp_prot = prots.iloc[:, i].to_frame()
        tmp_snp = snps.iloc[i, :]

        # Make a temporary joined dataframe holding the snp/protein pairs.
        # Since we need to have both a SNP and a protein level, we use an
        # inner join to ensure that only SIDs which have both measurements are retained.
        join_df = tmp_prot.join(tmp_snp, how="inner")

        # Drop any valus which have 'nan' in the SNP value
        non_nan_labels = ~(join_df.iloc[:, 1] == "nan")
        join_df = join_df.loc[non_nan_labels]

        prot_vals = join_df.iloc[:, 0].values.reshape(-1, 1)
        snp_vals = join_df.iloc[:, 1].values
        # Fit the model using the new values.
        nb.partial_fit(prot_vals, snp_vals, classes=class_orders[i])


# Function to compute probability of observing a genotype given a protein profile.
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

        onehot[nan_idcs[i]] = np.nan
        # Get the probability that each subject has this genotype
        tmp_preds = np.matmul(y_pred[i], onehot.T)
        tmp_preds = np.maximum(tmp_preds, np.finfo(np.float64).eps)
        tmp_preds = np.log(np.minimum(5, tmp_preds / (1 - tmp_preds + np.finfo(np.float64).eps)))
        tmp_preds[:, nan_idcs[i]] = 0

        # onehot[nan_idcs[i]] = 1 / (3 * np.ones(3))
        # # Get the probability that each subject has this genotype
        # tmp_preds = np.matmul(y_pred[i], onehot.T)
        all_preds.append(tmp_preds)
    full_arr = np.stack(all_preds).transpose(1, 2, 0)
    return full_arr


# Function to compute top-k accuracy for a distance matrix.
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


# Evaluates the model at each multiple #s of pQTLs used.
def get_outputs(prob_matrix, sids, clin_data,draw_probs = False):
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

    if 100 in actual_list:
        fig, ax = plt.subplots(figsize=(9, 7))

        prob_matrix = prob_matrix[:, :, :100].sum(axis=-1)
        top_k = get_top_k_accuracy(-prob_matrix, k=100)
        top_k_pct = top_k.sum(axis=0) / top_k.shape[0]

        x_idcs = np.arange(1,51,2)
        y_data = top_k_pct[x_idcs - 1]

        ax.plot(x_idcs,y_data,"o-",label="All Subjects (n=%d of %d genotyped)" % (prob_matrix.shape[0],prob_matrix.shape[1]))
        ax.set_ylim([-0.05,1.05])
        ax.set_xlim([0,51])
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        ticks_arr = np.arange(1,50,4)
        ax.set_xticks(ticks_arr)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        # Top K for each unique ancestry
        for anc in np.unique(clin_data.race):
            use_data = clin_data.race == anc
            anc_top_k = top_k[use_data]
            anc_top_k_pct = anc_top_k.sum(axis=0) / anc_top_k.shape[0]

            anc_y_data = anc_top_k_pct[x_idcs - 1]
            ax.plot(x_idcs,anc_y_data,"o-",label="%s (n=%d)" % (anc,anc_top_k.shape[0]))

        # Random Guess Line
        ax.plot(x_idcs,x_idcs/prob_matrix.shape[1],"o-",color="gray",label="Random Guess")
        # Legend and Labels
        ax.legend(fancybox=False, title="Ancestry", edgecolor="black")
        ax.set_xlabel("K")
        ax.set_ylabel("Accuracy")

        pct_ax = ax.twiny()
        pct_ax.set_xlim(ax.get_xlim())
        pct_ax.set_xticks(ticks_arr)
        pct_ax.set_xticklabels(["{:.2f}%".format((val/prob_matrix.shape[1])*100) for val in ticks_arr])
        pct_ax.spines["bottom"].set_position(("axes",-0.1))
        pct_ax.xaxis.set_ticks_position("bottom")
        pct_ax.xaxis.set_label_position("bottom")
        pct_ax.set_xlabel("% of Genotype Pool Size (n={})".format(prob_matrix.shape[1]))
        plt.tight_layout()
        plt.savefig(args.output_file + "_acc.png")
        plt.close()

        if draw_probs:
            make_comparison_df(prob_matrix=prob_matrix,sids=sids,clinical=clin_data,title=args.output_file + "_probs")



    out_df = pd.DataFrame.from_records(rows)
    return out_df


def _is_increasing(x):
    return np.all(x.values[1:] > x.values[0:-1])


def _both_same_direction(df):
    # Find indices where the test data order is not -1 (i.e. missing)
    rd_df = df.loc[df["Test Mean Order"] != -1, ["Train Mean Order", "Test Mean Order"]]

    train_increasing = _is_increasing(rd_df["Train Mean Order"])
    test_increasing = _is_increasing(rd_df["Test Mean Order"])

    return train_increasing == test_increasing


def run_qc(model, prots, snps, sort_pqtls, used_pqtls, generate_hists=False):
    qc_df_list = []
    if not isdir(join(getcwd(), "qc")):
        print("Making QC directory...")
        makedirs("qc")
    if not isdir(join(getcwd(),"qc",args.output_file)):
        makedirs(join("qc",args.output_file))
    out_path = join("qc",args.output_file)
    used_pqtls_new = list(used_pqtls)
    # Conditionally import this since it may not be available.
    if generate_hists:
        import seaborn as sns
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
        class_set = md.classes_

        removed_class = False
        if test_classes != train_classes:
            if len(test_classes - train_classes) == 0:
                print(
                    "WARNING: SNP '%s' is missing an expected SNP class. This is likely normal.\n\tExpected: %s\n\tReceived: %s" % (
                        snp, train_classes, test_classes))
            else:
                print(
                    "ERROR: SNP '%s' classes are different than expected.\n\tExpected: %s\n\tReceived: %s" % (snp,
                                                                                                              train_classes,
                                                                                                              test_classes))
                print("\tRemoving this SNP from the usable pQTLs list since this will cause prediction to fail.")
                removed_class = True
                used_pqtls_new.remove(i)
                class_set = sorted(list(test_classes))

        if generate_hists:
            fig, ax = plt.subplots(figsize=(9, 7))
            sns.histplot(data=join_df, x=prot, hue=snp, stat="density", hue_order=class_set, ax=ax, common_norm=False)

            lg = ax.get_legend()
            hist_handles = lg.legendHandles
            hist_labels = [t._text for t in lg.texts]
        model_rvs = dict()
        trained_qc_rows = []
        flag_pqtl = False
        for j, cls in enumerate(md.classes_):
            ## Get mean/stddev from trained model.
            mean = md.theta_[j, 0]
            var = md.sigma_[j, 0]
            std = np.sqrt(var)

            ## Plot QC Histograms
            rv = norm(loc=mean, scale=std)
            xrange = np.linspace(rv.ppf(0.001), rv.ppf(0.999), num=100)
            if generate_hists:
                ax.plot(xrange, rv.pdf(xrange), label="{} NB Estimate".format(cls))

            model_rvs[cls] = rv

            # Check how many points fall outside of 3*stddev range
            df_vals = join_df.loc[join_df[snp] == cls, prot]
            flag_class = False
            n_std_devs = 2
            if len(df_vals) != 0:
                pts_outside = (np.abs((df_vals - mean) / std) > n_std_devs).sum()
                pct_outside = (pts_outside / len(df_vals) * 100)

                if pct_outside > 50:
                    print(
                        "WARNING: For genotype '%s' in SNP '%s', %d (%0.2f%%) of points fall outside %s*stddev of estimated mean." % (
                            cls,
                            snp,
                            pts_outside,
                            pct_outside,
                            n_std_devs))
                    flag_class = True
                    flag_pqtl = True
            trained_qc_rows.append({"Class": cls,
                                    "Train Data Mean": mean,
                                    "Train Data Var": var,
                                    "Flag_Class": flag_class})
        if flag_pqtl and args.remove_qc_fails:
            if not removed_class:
                print("\tBecause --remove_qc_fails is set, pQTL (%s,%s) will be removed for prediction." % (prot,snp))
                used_pqtls_new.remove(i)
            else:
                print("\t--remove_qc_fails dicates that pQTL (%s,%s) should be removed, but class was already previously removed" % (prot,snp))
        if generate_hists:
            plot_handles, plot_labels = ax.get_legend_handles_labels()
            ax.legend(handles=hist_handles + plot_handles, labels=hist_labels + plot_labels)
            fig.savefig(join(out_path,"plot_%s_%s.png" % (prot.lower().replace("/","").replace(" ","").replace("-","_"), snp)))
            plt.close(fig)

        test_qc_df = join_df.groupby(snp) \
            .agg(["mean", "var"]) \
            .droplevel(level=0, axis="columns") \
            .rename(columns={"mean": "Test Data Mean", "var": "Test Data Var"})

        train_qc_df = pd.DataFrame.from_records(trained_qc_rows, index="Class")
        # Make QC Dataframe
        comb_qc_df = pd.concat([train_qc_df, test_qc_df], axis=1).reset_index().rename(columns={"index": "Class"})
        comb_qc_df["SNP"] = snp
        comb_qc_df["Protein"] = prot
        comb_qc_df["Train Mean Order"] = comb_qc_df["Train Data Mean"].argsort()
        comb_qc_df["Test Mean Order"] = comb_qc_df["Test Data Mean"].argsort()

        # same_dir = _both_same_direction(comb_qc_df)
        #
        # if not same_dir:
        #     str_df = str(comb_qc_df.loc[:, ["Class", "Train Mean Order", "Test Mean Order"]])
        #     # Prepend tab after each newline
        #     str_df = str_df.replace("\n", "\n\t\t")
        #     cprint("WARNING: For SNP '%s', The ordering of per-class means is different than expected.\n\t\t%s" % (
        #     snp, str_df), color="yellow")
        #     comb_qc_df["Flag"] = True

        qc_df_list.append(comb_qc_df)

    all_qc_df = pd.concat(qc_df_list).set_index(["SNP", "Protein", "Class"])
    # Style all_qc_df
    style_qc_df = all_qc_df.style.apply(
        lambda row: ["background-color: yellow" if row.Flag_Class is True else "" for v in row], axis=1)
    return np.array(used_pqtls_new), style_qc_df


if __name__ == "__main__":
    current_time = datetime.datetime.now().strftime("%h%d_%Y_%I%M%p")
    # Command line argument setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        required=True,
                        type=str,
                        choices=["predict", "train"])
    parser.add_argument("--train_test_split",
                        type=float,
                        default=0.5,
                        help="Percentage of data to use for training. 1-<val> is used for testing.")
    parser.add_argument("--read_split_file",
                        type=str,
                        help="Instead of randomly splitting into training and test sets, "
                             "read the train/test split from this file. This argument "
                             "overrides --train_test_split (i.e. If both are specified, "
                             "then the split_file will be used.")
    parser.add_argument("--split_file_out",
                        type=str,
                        default="split.csv",
                        help="When using --train_test_split, this argument sets the filename to write the train/test "
                             "split to. Default is 'split.csv'")
    parser.add_argument("--random_seed",
                        type=int,
                        default=1,
                        help="Seed for train/test split. Defaults to 1.")
    parser.add_argument("--snp_file",
                        required=True,
                        type=str,
                        help="Path to the file containing SNPs for each subject. Each row should correspond to a SNP, "
                             "with subjects in columns.")
    parser.add_argument("--snp_sep",
                        type=str,
                        default=",")
    parser.add_argument("--protein_file",
                        required=True,
                        type=str,
                        help="Path to the file containing protein levels. Each row should correspond to a subject, "
                             "with proteins in columns.")
    parser.add_argument("--protein_sep",
                        type=str,
                        default="\t")
    parser.add_argument("--use_pqtls",
                        default=100,
                        type=int,
                        help="Use the top N pQTLs from the complete set. Default is 100.")
    parser.add_argument("--output_file",
                        default="geno_pred_results_{}".format(current_time),
                        type=str, help="Name of the output file to write.")
    parser.add_argument("--model_output_file",
                        default="model_{}.pkl".format(current_time),
                        type=str,
                        help="The output filename for the modified model after training with new data.")
    parser.add_argument("--model_file",
                        default="frozen_model.pkl",
                        type=str,
                        help="The input model file to use.")
    parser.add_argument("--clinical_file",
                        required=False,
                        type=str,
                        help="Path to clinical data file.")
    parser.add_argument("--skip_qc",
                        action="store_true",
                        help="Include this flag to skip the QC step.")
    parser.add_argument("--generate_hists",
                        action="store_true",
                        help="Generates histograms during QC process.")
    parser.add_argument("--remove_qc_fails",
                        action="store_true",
                        help="Remove pQTLs that fail the QC process.")
    parser.add_argument("--draw_probs",
                        action="store_true",
                        help="Draw a probability plot for a few sids. Right now it only works for the SPIROMICS dataset.")
    args = parser.parse_args()

    print("Genotype Training and Prediction Code\n")
    print("Running with following argument configuration:")
    all_args = args.__dict__
    # Align the keys pretty
    longest_key = np.max([len(x) for x in all_args.keys()])
    for arg, value in all_args.items():
        disp_val = value
        if type(value) == str:
            disp_val = "'" + value + "'"
        print("\t{}: {}".format(arg.rjust(longest_key, " "), disp_val))
    print()
    print("Step 1: Data Loading")
    print("Loading model from file...")
    with open(args.model_file, "rb") as model_file:
        model_all = pickle.load(model_file)
    jhs_mapping = pd.read_csv("jhs_mapping_file.csv")
    jhs_mapping_dict = {k: v for k, v in zip(jhs_mapping.jhs_name, jhs_mapping.Target)}
    model_all["sort_pqtls"]["gene"] = model_all["sort_pqtls"]["gene"].apply(lambda x: jhs_mapping_dict[x])
    print("Model loaded successfully.")
    model = model_all["model"]
    class_order = model_all["class_order"]
    class_prior = model_all["class_prior"]
    sort_pqtls = model_all["sort_pqtls"]
    all_classes = model_all["all_classes"]
    is_trained_model = model_all["is_trained_model"]
    pqtl_pairs = [(x, y) for x, y in zip(sort_pqtls.SNP, sort_pqtls.gene)]

    # Load the files used for prediction
    snps, prots, used_pqtls, clin = load_files(snp_path=args.snp_file,
                                                    protein_path=args.protein_file,
                                                    use_pqtls=sort_pqtls,
                                                    snp_sep=args.snp_sep,
                                                    protein_sep=args.protein_sep,
                                                    clin_file=args.clinical_file)

    #possibly_bad = np.argwhere(sort_pqtls.SNP.isin(["rs3219175","rs3211938","rs7613239","rs11964923","rs12074019","rs12290398","rs696765","rs16825588","rs7846128"]).values).squeeze()
    # possibly_bad = np.argwhere(sort_pqtls.SNP.isin(["rs3219175",
    #                                                 "rs11964923",
    #                                                 "rs12290398",
    #                                                 "rs3211938",
    #                                                 "rs12074019",
    #                                                 "rs16825588",
    #                                                 "rs11698467",]).values).squeeze()
    #
    # used_pqtls = np.setdiff1d(used_pqtls,possibly_bad)
    # Run QC if selected
    if not args.skip_qc and is_trained_model:
        used_pqtls, qc_df = run_qc(model=model,
                                   prots=prots,
                                   snps=snps,
                                   sort_pqtls=sort_pqtls,
                                   used_pqtls=used_pqtls,
                                   generate_hists=args.generate_hists)
        qc_df.to_excel(join("qc",args.output_file,"qc_results.xlsx"))

    ### MODIFIED CODE SECTION
    # Add a step to select only the available pQTLs
    sort_pqtls_new = sort_pqtls.iloc[used_pqtls]
    all_classes_new = [all_classes[i] for i in used_pqtls]
    # Use all pQTLs that were not flagged during earlier QC
    snps = snps.loc[sort_pqtls_new.SNP]
    prots = prots.loc[:, sort_pqtls_new.gene]
    ### END MODIFIED CODE SECTION

    model = [model[x] for x in used_pqtls]
    class_order = [class_order[x] for x in used_pqtls]

    if args.mode == "train":
        print("\nStep 2: Model Training")
        # First find how many SIDs have both SNP and protein data.
        # Then use the train_test_split_parameter to select which SIDs are used for training and testing.

        if args.read_split_file is None:
            print("No Train/Test Split file was specified, will randomly generate the split.")
            # Split the SIDs with protein data randomly
            np.random.seed(args.random_seed)
            # If the percentage does not divide equally, round up for training.
            num_training = int(np.ceil(args.train_test_split * len(prots)))
            # Testing data is simply whatever is left not used by training.
            num_testing = len(prots) - num_training
            # Sample without replacement so we don't get the same SID multiple times.
            # Train SIDs are randomly chosen from the protein file.
            train_sids = np.random.choice(prots.index, num_training, replace=False)
            # Test SIDs are the set difference between all SIDs and the training SIDs.
            test_sids = np.setdiff1d(prots.index, train_sids)
        else:
            print("--read_split_file was specified, trying to read '{}'.".format(args.read_split_file))
            # Read from the file instead.
            split_file = pd.read_csv(args.read_split_file, index_col=0)
            train_sids = split_file.index[split_file.split == "train"].values
            test_sids = split_file.index[split_file.split == "test"].values

        # Should be no intersection between SIDs.
        assert (len(np.intersect1d(train_sids, test_sids)) == 0)
        # Size of train_sids + test_sids should equal len(prots)
        assert (len(prots) == len(train_sids) + len(test_sids))
        print("Train/test split passes tests, good to go.")
        # If we made this split randomly, write it to file. Otherwise, don't write.
        if args.read_split_file is None:
            # Write out train and test SIDs based on the choices above.
            sid_df = pd.Series(index=prots.index, name="split", dtype=object)
            sid_df.loc[train_sids] = "train"
            sid_df.loc[test_sids] = "test"
            assert (~sid_df.isna().any())
            print("Writing train/test split to '{}'.".format(args.split_file_out))
            sid_df = sid_df.to_csv(args.split_file_out)

        # Separate data into training and testing datasets.
        train_prots = prots.loc[train_sids]
        train_snps = snps.loc[:, train_sids]

        # If all data is used for training, predict on all data too.
        if args.train_test_split == 1.0:
            print("All data is used for training. Will predict on training set as well.")
            test_sids = train_sids

        test_prots = prots.loc[test_sids]
        test_snps = snps.loc[:, test_sids]

        # Train model on only training data.
        train_model(model, train_snps, train_prots, all_classes_new)

        # Write model out to file
        print("Writing new trained model out to '%s'." % args.model_output_file)
        with open(args.model_output_file, "wb") as model_out:
            # Only need to update the trained models, everything else should(!) be identical.
            model_all["model"] = model
            pickle.dump(model_all, model_out)

        print("\nStep 3: Predict with Trained Model")
        # Get predictions for genotype
        prediction_matrix = predict_genotype(model=model,
                                             prots=test_prots)

        # Get probability matrix.
        prob_matrix = eval_model(y_pred=prediction_matrix,
                                 y_true=test_snps,
                                 class_orders=class_order)

        # Compute final outputs
        output_df = get_outputs(prob_matrix=prob_matrix,
                                sids=np.array(test_prots.index),
                                draw_probs=args.draw_probs)

        # Write outputs to file.
        out_filename = "train_" + args.output_file
        output_df.to_csv(out_filename, index=False)
        print("Wrote results to '%s'." % out_filename)
    else:
        print("\nStep 2: Model Prediction")
        # Get predictions for genotype
        prediction_matrix = predict_genotype(model=model,
                                             prots=prots)

        # Get probability matrix.
        prob_matrix = eval_model(y_pred=prediction_matrix,
                                 y_true=snps,
                                 class_orders=class_order)

        # Compute final outputs
        output_df = get_outputs(prob_matrix=prob_matrix,
                                sids=np.array(prots.index),
                                clin_data=clin,
                                draw_probs=args.draw_probs)

        # Write outputs to file.
        out_filename = args.output_file
        output_df.to_csv(out_filename, index=False)
        print("Wrote results to '%s'." % args.output_file)
