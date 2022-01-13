## Setup

To setup the code, I recommend using a python virtual environment so that none of the packages we install conflict or overwrite the versions you already have. 

While in the main directory, run `python3 -m venv venv` to generate a new virtual environment called `venv`.

From the same directory run `source venv/bin/activate` to activate the virtual environment.

Now run `pip install -r requirements.txt` to install all the requirements listed in the provided `requirements.txt` file.

## Data 
Data files are located in `output/1_10_22/*` relative to the main directory. I do not recommend moving these files, as the relative paths are hardcoded in `paths.py` in order to make loading datasets easier.

### List of Files in `output/1_10_22/`
* use_pqtls_599_1_10_22.csv - This is a CSV file which contains 599 identified pQTLs (associations between a continuous protein value and a discrete genotype). The `gene` column lists the name of the Protein, and the `SNP` column lists the name of the SNP. The other columns contain information about the strength of the associations, as well as which alleles exist at each SNP.

* 1_10_22_copdgene_soma_protname_jhs_599.txt - This file contains the proteomic data for subjects in the COPDGene study. Each row records protein measurements for one subject. The `SID` column identifies the subject, and the other columns are protein names. These names correspond to the `gene` column of the `use_pqtls...` file.

* 1_10_22_copdgene_p2_soma_protname_jhs_599.txt - Identical format to the file above, except for subjects from Phase 2 of the COPDGene study. These are an independent set of subjects who we use as a validation set.

* copdgene_recoded_str_1_10_22.csv - This is a file containing string representations of genotypes for 9970 subjects in the COPDGene study. In this file, rows are a SNP, while each column corresponds to an SID.

* reference_snps_599_1_10_22.csv - This file contains a list of the major allele genotype for each of the SNPs listed in the genotype file. This file is used if we want to adjust for genotype effect to prevent reidentification.

## Running the Code
The `GenotypePrediction.py` file is the main python script used to run the code. 

A typical run of the code will look like:

`python3 GenotypePrediction.py --train_data COPDGene_P1_JHS --test_data COPDGene_P2_JHS --log_odds --use_pqtls 100`

This command will load the `COPDGene_P1_JHS` set to train the model, and then perform testing/validation on the `COPDGene_P2_JHS` set.

`--use_pqtls 100` tells the program to generate results using only the top 100 pQTLs from the file. (The pQTLs are sorted by ascending p-value so that higher associated pQTLs come first.)

`--log_odds` tells the program to compute the log_odds that a genotype matches with the proteome. This argument should likely always be used, as the results are better than omitting it.

To test performance when the data has been adjusted for genotype (The simple privacy-protection measure), add the flag `--mean_adjust` when running the command. This will adjust the data for genotype effect prior to training and predicting the model.

## Output

The code will output a CSV file with performance figures into the main directory. This file will be called `test_results_*.csv`.

The code will also generate a performance figure in the `figs/` directory. 



