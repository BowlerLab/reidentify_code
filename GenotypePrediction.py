import datetime
import pickle
import argparse
import numpy as np
import pandas as pd
from dataloader import load_data
from metrics import train_test_accuracy
# from paths import PAIR_LIST, REFERENCE_SNPS
from model import make_train_test, train_model, predict_model, eval_model

DATASET_CHOCIES = ["COPDGene_P1",
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
                   "SPIROMICS_JHS_ONLY",
                   "COPDGene_5k_TRAIN",
                   "COPDGene_5k_TEST",
                   "COPDGene_5k_QC_TRAIN",
                   "COPDGene_5k_QC_TEST"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pqtl_file", type=str, required=True)
    parser.add_argument("--train_data", type=str, choices=DATASET_CHOCIES, required=True)
    parser.add_argument("--test_data", type=str, choices=DATASET_CHOCIES, required=True)
    parser.add_argument("--use_pqtls", nargs="+", type=int,
                        help="Use only the top N pQTLs (sorted by FDR). Pass a single value or a set of values to test.",
                        default=[100])
    parser.add_argument("--mean_adjust", action="store_true")
    parser.add_argument("--log_odds", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--draw_probs", action="store_true")
    parser.add_argument("--output_model",
                        type=str,
                        help="Name of the model file to output.",
                        default="model_{}.pkl".format(datetime.datetime.now().strftime("%Y_%b%d_%H%M_%S")))
    args = parser.parse_args()

    adj_suffix = "_adj" if args.mean_adjust else ""
    log_odds_suffix = "_lg" if args.log_odds else ""
    # Step 1: Load train/test data.
    train_data, train_proteins, train_snps, train_clinical, train_other_snps, test_data, test_proteins, test_snps, test_clinical, test_other_snps = load_data(
        args)

    # Step 2: Read pQTL list.
    pqtls = pd.read_csv(args.pqtl_file)
    # Step 2.5: Sort pQTL list by p-value.
    # Sort by this value.
    sort_pqtls = pqtls.sort_values("p-value", ascending=True)
    # Drop c-Jun, which is problematic
    sort_pqtls = sort_pqtls.drop(sort_pqtls.index[sort_pqtls.gene == "c-Jun"])
    # sort_pqtls = sort_pqtls.drop(sort_pqtls.index[~sort_pqtls.gene.isin(["DERM","sICAM-5"])])
    pqtl_pairs = [(x, y) for x, y in zip(sort_pqtls.SNP, sort_pqtls.gene)]

    # Step 3: Convert raw data into training and test datasets with x as protein measurements and y as genotype.
    x_train, y_train, train_sids, x_test, y_test, test_sids, all_classes, ref_snps = make_train_test(pqtl_pairs,
                                                                                                     train_proteins,
                                                                                                     train_snps,
                                                                                                     test_proteins,
                                                                                                     test_snps,
                                                                                                     align_to_reference=args.mean_adjust,
                                                                                                     train_other_snps=train_other_snps,
                                                                                                     test_other_snps=test_other_snps,
                                                                                                     # dump_charts=True
                                                                                                     )
    # Step 4: Train model using training dataset.
    trained_model, class_order, class_prior = train_model(x_train, y_train, all_classes, skip_train=args.skip_train)
    # Step 4.5: Dump out trained model in pickle (binary) format so that we can predict without using the training data.
    dump_obj = {"model": trained_model,
                "class_order": class_order,
                "class_prior": class_prior,
                "sort_pqtls": sort_pqtls,
                "ref_snps": ref_snps,
                "all_classes": all_classes,
                "is_trained_model": not args.skip_train}

    with open(args.output_model, "wb") as write_file:
        pickle.dump(dump_obj, write_file)
    # Step 5: Generate prediction probabilities for all 3 genotype classes at each pQTL.
    # Do this for both training and testing datasets.
    train_preds = predict_model(trained_model, x_train, log_odds=args.log_odds)
    # Don't predict twice if using the same dataset
    if test_data != train_data:
        test_preds = predict_model(trained_model, x_test, log_odds=args.log_odds)
    else:
        test_preds = train_preds

    # Step 5.5: Add in the genotypes of subjects without protein measurements to increase the pool size.
    y_train = np.concatenate([y_train, train_other_snps.loc[:, [p[0] for p in pqtl_pairs]].values.T], axis=-1)
    y_test = np.concatenate([y_test, test_other_snps.loc[:, [p[0] for p in pqtl_pairs]].values.T], axis=-1)

    train_other_sids = np.array(train_other_snps.index)
    test_other_sids = np.array(test_other_snps.index)

    # Dictionary for memoizing results
    tag_dict = {}

    # Since pQTLs are already sorted, we can just use the first N pQTLs to get the performance for using smaller numbers
    # of pQTLs
    results_index = args.use_pqtls
    results_list = []

    for res_i in results_index:
        print("Running with i=%d" % res_i)
        # Step 6: For each known protein palette, generate a probability score of that protein palette arising from each
        # genotype vector in the pool.
        print("Evaluating training data...")
        train_prob_m = eval_model(train_preds, y_train, class_order, class_prior, num_proteins=res_i,
                                  memo_tag=train_data, log_odds=args.log_odds)
        print("Evaluating testing data...")
        if test_data != train_data:
            test_prob_m = eval_model(test_preds, y_test, class_order, class_prior, num_proteins=res_i,
                                     memo_tag=test_data, log_odds=args.log_odds)
        else:
            test_prob_m = train_prob_m

        # Compute the train/test accuracy and charts for performance of the model.
        row = train_test_accuracy(train_prob_m, train_sids, train_clinical, test_prob_m, test_sids, test_clinical,
                                  train_title=train_data, test_title=test_data,
                                  fname="train_%s_test_%s_%d%s%s_proteins_accuracy.png" % (
                                      train_data, test_data, res_i, adj_suffix, log_odds_suffix),
                                  draw_probs=args.draw_probs & (res_i == 100), test_other_sids=test_other_sids,
                                  train_other_sids=train_other_sids)
        results_list.append(row)

    out_df = pd.DataFrame.from_records(results_list, index=results_index, columns=["Train Top-1 Accuracy",
                                                                                   "Train Top 3 Accuracy",
                                                                                   "Train Top 1%% Accuracy",
                                                                                   "Test Top 1 Accuracy",
                                                                                   "Test Top 3 Accuracy",
                                                                                   "Test Top 1%% Accuracy"])
    out_df.to_csv("test_results_{}_{}{}{}.csv".format(train_data, test_data, adj_suffix, log_odds_suffix))
