import pandas as pd
import numpy as np
from scipy.integrate import quad
import pickle
from Bio import SeqIO
import matplotlib.pyplot as plt
from rdf_b22_functions import corrected_rdfs_b22
import random
from energy_funcs import yukawa_potential, ah_potential, calc_dmap, genParamsDH, ah_scaled
from numba import njit
import math
from argparse import ArgumentParser
import subprocess, os



# File paths
residue_csv = "residues.csv"
fasta_file = "/Users/yuktikhanna/calvados/all_tads_rggs_dis.fasta"
#in_folder = f"/Users/yuktikhanna/calvados/datasets_predicted_{args.date:s}"
in_folder = f"/Users/yuktikhanna/calvados/"
out_folder = f"/Users/yuktikhanna/calvados/fig4/supplementary"
os.makedirs(out_folder, exist_ok=True)
box = 20  # Box size for corrected_rdfs_b22 function
rggs_file = "/Users/yuktikhanna/calvados/rg_tract_headers.txt"
tads_file = "/Users/yuktikhanna/calvados/tad_headers.txt"
# Load residue properties
residue_properties = pd.read_csv(residue_csv)
residue_properties.set_index("one", inplace=True)

# Generate names for all pairs
rggs = ["Q14152_5_3", "O00144_1_1", "Q9Y3Y2_3_1", "P19338_5_1", "Q02388_5_13"]
tads = ["P98177_463_504", "Q13285_373_392", "P49910_193_208", "Q9UJU2_14_65", "P35716_353_367"]
#names = [f"{rgg}__{tad}" for rgg in rggs for tad in tads]

rggs_3 = ["Q86V81_2_1","Q86V81_5_1", "P49959_5_1"]
#for i in rggs_3:
#    names.append(f"{i}__P04637_16_56")

# Parse sequences from the FASTA file
sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}

def file_to_list(input_file):
    f1=open(input_file,"r")
    list1=[]
    for line in f1:
        stripped=line.strip()
        list1.append(stripped)
    f1.close()
    return list1

#tads_list = file_to_list(tads_file)
#rggs_list = file_to_list(rggs_file)
#names = [f"{rgg}__{tad}" for rgg in rggs_list for tad in tads_list]

###names = file_to_list(f"/Users/yuktikhanna/calvados/lists/sim_pairs_for_training_{args.date:s}.txt")
##names = file_to_list(f"/Users/yuktikhanna/calvados/lists/sim_pairs_for_training_270525.txt")
names = file_to_list("/Users/yuktikhanna/calvados/fig4/initial_set_names_sliced.txt")
#names = 'A0A286YF58_2_1__Q06945_399_417', 'Q00839_5_1__Q9H0D2_1329_1346', 'Q9UN86_3_1__Q9H0D2_1329_1346': [[0.0, 0.0], [0.0, 0.0]], 'O95886_1_1__Q9NPC7_200_216': [[0.0, 0.0], [0.0, 0.0]], 'P48634_1_1__Q9NPC7_200_216': [[0.0, 0.0], [0.0, 0.0]], 'Q99501_3_2__O95935_239_254': [[0.0, 0.0], [0.0, 0.0]], 'O00571_1_1__O95935_239_254': [[0.0, 0.0], [0.0, 0.0]], 'Q96JY0_3_1__Q92731_352_377': [[0.0, 0.0], [0.0, 0.0]], 'P51991_2_1__Q92731_352_377', 'P0DN76_2_1__Q03112_178_193', 'P79522_1_1__Q03112_178_193']

@njit(nopython=True)
def get_qs_fast(seq):
    """ charges and absolute charges vs. residues """
    qs = np.zeros(len(seq))
    qs_abs = np.zeros(len(seq))

    # loop through sequence
    for idx in range(len(seq)):
        if seq[idx] in ['R','K']:
            qs[idx] = 1.
            qs_abs[idx] = 1.
        elif seq[idx] in ['E','D']:
            qs[idx] = -1.
            qs_abs[idx] = 1.
        else:
            qs[idx] = 0.
            qs_abs[idx] = 0.
    return qs, qs_abs

@njit(nopython=True)
def calc_SCD(seq,charge_termini=False):
    """ Sequence charge decoration, eq. 14 in Sawle & Ghosh, JCP 2015 """
    qs, _ = get_qs_fast(seq)
    if charge_termini:
        qs[0] = qs[0] + 1.
        qs[-1] = qs[-1] - 1.
    N = len(seq)
    scd = 0.
    for idx in range(1,N):
        for jdx in range(0,idx):
            s = qs[idx] * qs[jdx] * (idx - jdx)**0.5
            scd = scd + s
    scd = scd / N
    return scd

def make_lambda_map(residues):
    lambda_map = {}
    for key0, val0 in residues.iterrows():
        l0 = val0['lambdas']
        for key1, val1 in residues.iterrows():
            l1 = val1['lambdas']
            l = l0+l1
            lambda_map[(key0,key1)] = l
            lambda_map[(key1,key0)] = l
    return lambda_map

#@njit(nopython=True)
def calc_SHD(seq,beta=-1.):
    """ Sequence hydropathy decoration, eq. 4 in Zheng et al., JPC Letters 2020"""
    N = len(seq)
    shd = 0.
    # lambdas = residues.lambdas[list(seq)].to_numpy()
    # lambda_map = np.add.outer(lambdas,lambdas)
    lambda_map = make_lambda_map(residue_properties)
    for idx in range(0, N-1):
        seqi = seq[idx]
        for jdx in range(idx+1, N):
            seqj = seq[jdx]
            s = lambda_map[(seqi,seqj)] * (jdx - idx)**beta
            # s = lambda_map[idx,jdx] * (jdx - idx)**beta
            shd = shd + s
    shd = shd / N
    return shd


from re import findall
def calc_aromatics(seq):
    """ Fraction of aromatics """
    seq = str(seq)
    N = len(seq)
    rY = len(findall('Y',seq)) / N
    rF = len(findall('F',seq)) / N
    rW = len(findall('W',seq)) / N
    return int(rY+rF+rW)


'''from localcider.sequenceParameters import SequenceParameters
def calc_kappa(seq):
    seq = "".join(seq)
    SeqOb = SequenceParameters(seq)
    k = SeqOb.get_kappa()
    return k'''




# Custom function to calculate properties
def calculate_property(sequence, residue_properties, property_name):
    # This will calculate properties (lambda, sigma, charge) for each residue in the sequence
    #print(residue_properties.loc["R", property_name])
    all = np.array([residue_properties.loc[aa, property_name] for aa in sequence if aa in residue_properties.index])
    return all

def charge_complementarity(charges_rgg, charges_tad):
    qij = abs(charges_rgg + charges_tad)
    del_q = qij - 0.5 * (abs(charges_tad)+abs(charges_rgg))
    return del_q

# Function to create ah_intgrl_map
def make_ah_intgrl_map(residue_properties, rc=2.0, eps=0.2 * 4.184):
    ah_intgrl_map = {}
    residues = residue_properties.index
    for res1 in residues:
        sig1, lamb1 = residue_properties.loc[res1, "sigmas"], residue_properties.loc[res1, "lambdas"]
        for res2 in residues:
            sig2, lamb2 = residue_properties.loc[res2, "sigmas"], residue_properties.loc[res2, "lambdas"]
            sigma, lamb = 0.5 * (sig1 + sig2), 0.5 * (lamb1 + lamb2)
            integral = quad(lambda r: ah_scaled(r, sigma, eps, lamb, rc), 2**(1/6) * sigma, rc)[0]
            ah_intgrl_map[(res1, res2)] = integral
    return ah_intgrl_map

# Function to calculate interactions between two sequences
def calc_ah_between_sequences(seq1, seq2, ah_intgrl_map):
    U = 0.0
    seq1 = list(seq1)
    seq2 = list(seq2)
    N1 = len(seq1)
    N2 = len(seq2)
    
    for res1 in seq1:
        for res2 in seq2:
            ahi = ah_intgrl_map[(res1, res2)]
            U += ahi
    
    U /= (N1 * N2)
    return U

ah_intgrl_map = make_ah_intgrl_map(residue_properties)

def opp_charges(sequence_rgg, sequence_tad):

    charge_i = calculate_property(sequence_rgg, residue_properties, "q")
    charge_j = calculate_property(sequence_tad, residue_properties, "q")

    # Extract positive and negative charges
    n_plus_i = np.sum(np.where(charge_i > 0, charge_i, 0))
    n_minus_i = -np.sum(np.where(charge_i < 0, charge_i, 0))

    n_plus_j = np.sum(np.where(charge_j > 0, charge_j, 0))
    n_minus_j = -np.sum(np.where(charge_j < 0, charge_j, 0))

    '''print("seq_i: ", sequence_rgg, "charge_i: ", charge_i)
    print("neg_charge_i: ", np.where(charge_i < 0, charge_i, 0), "pos_charge_i: ", np.where(charge_i > 0, charge_i, 0))
    print("tot_neg_charge_i: ", n_minus_i, "tot_pos_charge_i: ", n_plus_i)
    print("seq_j: ", sequence_tad, "charge_j: ", charge_j)
    print("neg_charge_j: ", np.where(charge_j < 0, charge_j, 0), "pos_charge_j: ", np.where(charge_j > 0, charge_j, 0))
    print("tot_neg_charge_j: ", n_minus_j, "tot_pos_charge_j: ", n_plus_j)'''

    # Compute the result
    result = (n_plus_i * n_minus_j + n_minus_i * n_plus_j) - (n_plus_i * n_plus_j + n_minus_i * n_minus_j)
    '''print("n_plus_i * n_minus_j: ", n_plus_i * n_minus_j, "; n_minus_i * n_plus_j", n_minus_i * n_plus_j, "; n_plus_i * n_plus_j: ", n_plus_i * n_plus_j, "; n_minus_i * n_minus_j: ", n_minus_i * n_minus_j)
    print("term_1: ", n_plus_i * n_minus_j + n_minus_i * n_plus_j, "; term_2: ", n_plus_i * n_plus_j + n_minus_i * n_minus_j)
    print("final_opp_charge", result)'''
    return result

def sampling_error_calc(data):
    n = len(data)
    mean = data.mean()
    s   = data.std(ddof=1)        # sample standard deviation
    se  = s / np.sqrt(n)          # analytic standard error

    # bootstrap
    B = 1000
    boot_means = np.random.choice(data, size=(B, n), replace=True).mean(axis=1)
    se_boot    = boot_means.std(ddof=1)

    print(f"Mean          = {mean:.3f}")
    print(f"SE (analyt.)  = {se:.3f}")
    print(f"SE (bootstrap)= {se_boot:.3f}")
    return s, se

def plot_se(se,mean):
    
    '''# prepare bar chart
    labels   = ['-B22', 'log10(-B22)']
    se_vals  = [se1, se2]
    x        = np.arange(len(labels))

    plt.bar(x, se_vals)          # bars of height SE
    plt.xticks(x, labels)        # label the bars
    plt.ylabel('Standard Error')
    plt.title('Comparison of Sampling Error of B22')'''

    ci   = 1.96 * se
    plt.errorbar(1, mean, yerr=ci, fmt='o', capsize=5)
    plt.xlim(0.5, 1.5)
    plt.xticks([1], ["Sample mean"])
    plt.ylabel("Value")
    plt.title("Mean ± 95% CI")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"b22s_sampling_error.png", dpi=300)
    plt.close()



# Analyze each pair
def analyse_features(names, residue_properties, sequences):
    seq_dict = {}
    results = []
    b22s_ganguly = []
    b22s_hummer = []
    b22s_hummer_log = []
    count_negs = 0
    for i, name in enumerate(names):
        rgg, tad = name.split("__", 1)
        sequence_rgg = sequences[rgg]
        sequence_tad = sequences[tad]

        # Calculate lambda, sigma, charge for RGG and TAD
        lambdas_rgg = np.sum(calculate_property(sequence_rgg, residue_properties, "lambdas"))
        lambdas_tad = np.sum(calculate_property(sequence_tad, residue_properties, "lambdas"))
        sigmas_rgg = np.sum(calculate_property(sequence_rgg, residue_properties, "sigmas"))
        sigmas_tad = np.sum(calculate_property(sequence_tad, residue_properties, "sigmas"))
        charges_rgg = np.sum(calculate_property(sequence_rgg, residue_properties, "q"))
        charges_tad = np.sum(calculate_property(sequence_tad, residue_properties, "q"))

        # Calculate average lambda, sigma, charge for the pair
        avg_lambda = (lambdas_rgg / len(sequence_rgg) + lambdas_tad / len(sequence_tad)) / 2
        avg_sigma = (sigmas_rgg + sigmas_tad) / 2
        avg_charge = (charges_rgg + charges_tad) / 2
        
        # Calculate net charge per residue (ncpr)
        #total_charge_rgg = np.sum(charges_rgg)
        #total_charge_tad = np.sum(charges_tad)
        ncpr_rgg = charges_rgg / len(sequence_rgg) if len(sequence_rgg) > 0 else 0
        ncpr_tad = charges_tad / len(sequence_tad) if len(sequence_tad) > 0 else 0
        avg_ncpr = (ncpr_rgg + ncpr_tad) / 2
        charge_comp = charge_complementarity(charges_rgg, charges_tad)
        average_interaction = calc_ah_between_sequences(sequence_rgg, sequence_tad, ah_intgrl_map)

        # Additional metrics
        scd_rgg = calc_SCD(sequence_rgg)
        scd_tad = calc_SCD(sequence_tad)
        avg_scd = (scd_rgg + scd_tad) / 2

        shd_rgg = calc_SHD(sequence_rgg)
        shd_tad = calc_SHD(sequence_tad)
        avg_shd = (shd_rgg + shd_tad) / 2

        aromatics_rgg = calc_aromatics(sequence_rgg)
        aromatics_tad = calc_aromatics(sequence_tad)
        avg_aromatics = (aromatics_rgg + aromatics_tad)/2

        '''kappa_rgg = calc_kappa(sequence_rgg)
        kappa_tad = calc_kappa(sequence_tad)
        avg_kappa = (kappa_rgg + kappa_tad) / 2'''

        opp_charge = opp_charges(sequence_rgg, sequence_tad)

                
        '''# Store results
        results.append({
            "name": name,
            "rgg": rgg,
            "RGG_length": len(sequence_rgg),
            "charge_RGG": charges_rgg,
            "lambda_RGG": lambdas_rgg,
            "NCPR_RGG": ncpr_rgg,
            "SCD_RGG": scd_rgg,
            "SHD_RGG": shd_rgg,
            "tad": tad,
            "TAD_length": len(sequence_tad),
            "charge_TAD": charges_tad,
            "lambda_TAD": lambdas_tad,
            "NCPR_TAD": ncpr_tad,
            "SCD_TAD": scd_tad,
            "SHD_TAD": shd_tad,
            "Average_lambda": avg_lambda,
            "Min_lambda": min(lambdas_rgg / len(sequence_rgg), lambdas_tad / len(sequence_tad)),
            "Average_sigma": avg_sigma,
            "Average_charge": avg_charge,
            "Opposite_Charge_Number": opp_charge,
            "Average_NCPR": avg_ncpr,
            "Charge_Complementarity": charge_comp,
            "Total_Length": (len(sequence_rgg)+len(sequence_tad)),
            "Average_Interaction": average_interaction,
            "Average_SCD": avg_scd,
            "Average_Aromatics": avg_aromatics,
            "Average_SHD": avg_shd
        })'''

        '''#Store sequences
        results.append({
            "name": name,
            "seq_rgg": sequence_rgg,
            "seq_tad": sequence_tad,
            "total_seq": sequence_rgg+sequence_tad
        })

        seq_dict.setdefault(name, sequence_rgg+sequence_tad)'''

        
        #print(charge_comp)
        # Load RDFs and calculate b22
        df_file = f"{in_folder}/test_rdfs/rdfs_{name}_{box}nm.pkl"
        with open(df_file, "rb") as f:
            df = pickle.load(f)
        
        last = 7.5
        delta = 2
        _, b22s_dict, kds_dict = corrected_rdfs_b22(df, last, box_size=box, delta=delta, rc_nm=4.0)

        # Collect b22 values
        b22_values = [b22s_dict[key]["b22"] for key in b22s_dict.keys()]
        b22_corr1_values = [b22s_dict[key]["b22_corr1"] for key in b22s_dict.keys()]
        b22_corr2_values = [b22s_dict[key]["b22_corr2"] for key in b22s_dict.keys()]

        b22_corr1_mean = np.mean(b22_corr1_values)
        b22_corr2_mean = np.mean(b22_corr2_values)
        
        if b22_corr2_mean > -1:
            b22_corr2_mean = -5.302348488437912 ## second highest negative b22 value
        b22_corr2_non_neg  = np.mean(b22_corr2_values)
        if b22_corr2_non_neg > 0:
            b22_corr2_non_neg = -b22_corr2_non_neg

        # Collect kd values
        kd_values = [kds_dict[key]["Kd"] for key in kds_dict.keys()]
        kd_corr1_values = [kds_dict[key]["Kd_corr1"] for key in kds_dict.keys()]
        kd_corr2_values = [kds_dict[key]["Kd_corr2"] for key in kds_dict.keys()]

        kd_corr1_mean = np.mean(kd_corr1_values)
        kd_corr2_mean = np.mean(kd_corr2_values)
        
        charge_assy=np.abs(charges_tad - charges_rgg)
        lambdas_ratio=lambdas_tad/lambdas_rgg+1e-06
        # Store results
        results.append({
            "name": name,
            "rgg": rgg,
            "RGG_length": len(sequence_rgg),
            "charge_RGG": charges_rgg,
            "lambda_RGG": lambdas_rgg,
            "NCPR_RGG": ncpr_rgg,
            "SCD_RGG": scd_rgg,
            "SHD_RGG": shd_rgg,
            "tad": tad,
            "TAD_length": len(sequence_tad),
            "charge_TAD": charges_tad,
            "lambda_TAD": lambdas_tad,
            "NCPR_TAD": ncpr_tad,
            "SCD_TAD": scd_tad,
            "SHD_TAD": shd_tad,
            "Average_lambda": avg_lambda,
            "Min_lambda": min(lambdas_rgg / len(sequence_rgg), lambdas_tad / len(sequence_tad)),
            "Average_sigma": avg_sigma,
            "Average_charge": avg_charge,
            "Opposite_Charge_Number": opp_charge,
            "Average_NCPR": avg_ncpr,
            "Charge_Complementarity": charge_comp,
            "Total_Length": (len(sequence_rgg)+len(sequence_tad)),
            "Average_Interaction": average_interaction,
            "Average_SCD": avg_scd,
            "Average_Aromatics": avg_aromatics,
            "Average_SHD": avg_shd,
            "Normalized_OCN": opp_charge / (len(sequence_rgg)+len(sequence_tad)),
            "Charge_Asymmetry": charge_assy,
            "Hydropathy_Diff": shd_tad - shd_rgg,
            "Lambda_Ratio": lambdas_ratio,
            #"Average_Kappa": avg_kappa,
            "B22_corr1_log10": np.log10(-b22_corr1_mean),
            "B22_corr2_log10": np.log10(-b22_corr2_mean),
            "B22_corr2_log10_no_pos": np.log10(-b22_corr2_non_neg),
            "B22": np.mean(b22_values),
            "Corrected_B22_Hummer": np.mean(b22_corr1_values),
            "Corrected_B22_Ganguly": np.mean(b22_corr2_values),
            #"Kd": np.mean(b22_values),
            #"Corrected_Kd_Hummer": np.mean(kd_corr1_values),
            #"Corrected_Kd_Ganguly": np.mean(kd_corr2_values),
        })
        #print(i)#
        b22s_ganguly.append(np.log10(-b22_corr2_mean))
        b22s_hummer_log.append(np.log10(-b22_corr1_mean))
        b22s_hummer.append(-b22_corr1_mean)
        #print(np.log10(-b22_corr1_mean), -b22_corr1_mean)

        if np.log10(-b22_corr2_mean) < 0:
            print("this one is strange: ", name, "with B22: ", b22_corr2_mean)
            count_negs = count_negs+1

    print("number of strangers: ", count_negs)
    
    print("min_log_b22_ganguly: ", min(b22s_ganguly))
    return results, np.array(b22s_hummer), np.array(b22s_hummer_log)
    #return seq_dict


#store seqs

#out_folder = f"/Users/yuktikhanna/calvados/datasets/"
results, b22s_hummer, b22s_hummer_log = analyse_features(names, residue_properties, sequences)
s_hummer, se_hummer = sampling_error_calc(b22s_hummer)
s_hummer_log, se_hummer_log = sampling_error_calc(b22s_hummer_log)
plot_se(se_hummer, se_hummer.mean())
# Convert results to DataFrame
results_df = pd.DataFrame(results)
#print(results_df)
#print(results_df.columns)
#print(results_df["Charge_Complementarity"].head())
#results_df.to_pickle(f'{out_folder}/pairs_features_data_simulations_20nm.pkl')
results_df.to_csv(f'{out_folder}/initial_set_features_data.csv')
#results_df.to_csv(f'{out_folder}/test_pairs_features_data_all_big.csv')
#sim_pairs_data_file = f'{out_folder}/sim_pairs_added_features_data_{args.date:s}.csv'



'''all_pairs_data_file = f'{out_folder}/all_pairs_for_training_features_data.csv'
results_df = pd.read_csv(sim_pairs_data_file)
df_initial = pd.read_csv(sim_pairs_data_file)
df_big = pd.read_csv(all_pairs_data_file)

###sim_pairs_data_file = f'{in_folder}/output.csv'
sim_pairs_data_file = f'{in_folder}/sim_pairs_added_features_data_090425.csv'
unsim_pairs_data_file = f'{in_folder}/unsim_pairs_for_training_220425_1.csv'
results_df = pd.read_csv(sim_pairs_data_file)
df_add = pd.read_csv(unsim_pairs_data_file)'''


'''def split_large_bins(df, col, initial_edges, max_count):
    """
    Recursively split any bin whose count > max_count into equal‑count sub‑bins.
    
    df              : DataFrame (must contain col)
    col             : the numeric column to bin (string)
    initial_edges   : array of bin edges (length M+1) from your qcut
    max_count       : threshold; any bin with > max_count rows gets split
    """
    df = df.copy()
    # 1) assign the initial bins 0…M‑1
    df['bin_id'] = pd.cut(df[col], bins=initial_edges, labels=False, include_lowest=True)
    # label them as strings so sublabels don’t collide
    df['final_bin'] = df['bin_id'].astype(str)

    # 2) find bins that exceed max_count
    counts = df['final_bin'].value_counts()
    too_big = counts[counts > max_count].index.tolist()

    # 3) loop until no bin is too big
    while too_big:
        for b in too_big:
            mask = df['final_bin'] == b
            subset = df.loc[mask, col]
            n = len(subset)
            # how many sub‑bins we need
            n_sub = math.ceil(n / max_count)

            # rank within this bin, then floor to sub‑bin indices
            ranks = subset.rank(method='first')
            sub_ids = np.floor((ranks - 1) * n_sub / n).astype(int)

            # relabel as “b_0”, “b_1”, …
            df.loc[mask, 'final_bin'] = sub_ids.map(lambda i, b=b: f"{b}_{i}")

        # recompute which new labels are still too big
        counts = df['final_bin'].value_counts()
        too_big = counts[counts > max_count].index.tolist()

    return df

# —––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––—

# 1) Initial quantile-based bins on the big set
n_bins = 50
df_big['qbin'], bin_edges = pd.qcut(
    df_big['Opposite_Charge_Number'],
    q=n_bins,
    labels=False,
    retbins=True,
    duplicates='drop'
)

# 2) Recursive splitter to get final_bin on the big set
ideal     = len(df_big) / n_bins
max_count = math.ceil(ideal)

def split_large_bins(df, col, edges, max_count):
    df = df.copy()
    df['bin_id']    = pd.cut(df[col], bins=edges, labels=False, include_lowest=True)
    df['final_bin'] = df['bin_id'].astype(str)
    while True:
        counts  = df['final_bin'].value_counts()
        too_big = counts[counts > max_count].index.tolist()
        if not too_big:
            break
        for b in too_big:
            mask   = df['final_bin'] == b
            subset = df.loc[mask, col]
            n      = len(subset)
            n_sub  = math.ceil(n / max_count)
            ranks  = subset.rank(method='first')
            sub_ids = np.floor((ranks - 1) * n_sub / n).astype(int)
            df.loc[mask, 'final_bin'] = [f"{b}_{i}" for i in sub_ids]
    return df

df_split = split_large_bins(df_big, "Opposite_Charge_Number", bin_edges, max_count)

# 3) Get numeric bounds per final_bin
bounds = (df_split
            .groupby('final_bin')['Opposite_Charge_Number']
            .agg(min_val='min', max_val='max')
            .sort_index())

# 4) Assign final_bin to small & additional sets
for df in (df_initial, df_add):
    df['final_bin'] = None
    for lbl, (mn, mx) in bounds.iterrows():
        df.loc[df['Opposite_Charge_Number'].between(mn, mx), 'final_bin'] = lbl
    df['final_bin'] = df['final_bin'].astype(pd.CategoricalDtype(categories=bounds.index))

counts_big   = df_split   ['final_bin'].value_counts().sort_index()
counts_small = df_initial['final_bin'].value_counts().sort_index()
# 2) Compute target counts in 1000 rows
target_total = 1000
total_big    = counts_big.sum()
# floating-point ideal
target_float = counts_big / total_big * target_total
# integer targets (round to nearest)
target_counts = target_float.round().astype(int)

# 3) Compute how many to add per bin (deficit)
deficits = (target_counts - counts_small).clip(lower=0)
print("deficits", deficits)

# 4) Sample exactly that many from your pool (df_add)
samples = []
for bin_label, need in deficits.items():
    if need <= 0:
        continue
    pool = df_add[df_add['final_bin'] == bin_label]
    # if not enough in pool, take all; otherwise take exactly `need`
    take = min(len(pool), need)
    samples.append(pool.sample(take, random_state=42))
    print(bin_label, need, pool.sample(take, random_state=42))
    
df_sample = pd.concat(samples, ignore_index=True)

# 5) Augment and verify
df_augmented = pd.concat([df_initial, df_sample], ignore_index=True)
counts_augmented = df_augmented['final_bin'].value_counts().sort_index()

# 6) Show results side by side
summary = pd.DataFrame({
    'big'      : counts_big,
    'small'    : counts_small,
    'target'   : target_counts,
    'added'    : deficits.clip(upper=df_add['final_bin'].value_counts().reindex(deficits.index, fill_value=0)),
    'augmented': counts_augmented
}).fillna(0).astype(int)

print(summary)
df_sample.to_csv(f'{in_folder}/ocn_binned_features_data_230425.csv')
'''

'''# 2) Create 100 quantile bins and capture the edges
n_bins = 30

df_big['qbin'], bin_edges = pd.qcut(
    df_big['Opposite_Charge_Number'],
    q=n_bins,
    labels=False,
    retbins=True,
    duplicates='drop'
)

# 3) Decide your max_count threshold
#    e.g. the “ideal” per‑bin count is len(df_big)/n_bins,
#    you can allow a bit of slack or take the ceiling.
ideal = len(df_big) / n_bins
max_count = math.ceil(ideal)

# 4) Run the recursive splitter using those edges
df_split = split_large_bins(df_big, 
                            col="Opposite_Charge_Number", 
                            initial_edges=bin_edges, 
                            max_count=max_count)



# 5) Inspect how many values ended up in each final_bin
counts = df_split['final_bin'].value_counts().sort_index()
print(counts)

# 1) Compute per‑bin min/max on the big set
bounds = (
    df_split
      .groupby('final_bin')['Opposite_Charge_Number']
      .agg(min_val='min', max_val='max')
      .sort_values('min_val')
)

# 2) Print any zero-width bins
zero_width = bounds[bounds['min_val'] == bounds['max_val']]
if not zero_width.empty:
    print("Zero-width bins (min == max):")
    for lbl, row in zero_width.iterrows():
        print(f"  Bin '{lbl}': edge = {row['min_val']}")
else:
    print("No zero-width bins.")

# 2) Build a “combined label” for each unique (min, max) pair
#    – zero-width bins become just their value (“0”, “3”, “4”, …)
#    – wider bins become “min-max” strings (e.g. “-98--6”, “-6--3”, …)
def make_label(row):
    if row.min_val == row.max_val:
        return f"{row.min_val}"
    else:
        return f"{row.min_val}-{row.max_val}"


# 2) Drop any zero‑width bins (where max_val == min_val)
bounds = bounds[bounds['max_val'] > bounds['min_val']]

# 3) Build your unique, strictly increasing edges list
edges = [bounds['min_val'].iloc[0]] + bounds['max_val'].tolist()

# 4) These are the labels you’ll attach (one per interval)
labels = bounds.index.tolist()

print(f"Unique edges ({len(edges)}):\n", edges)
print(f"Labels ({len(labels)}):\n", labels)

# 5) Now cut the small DataFrame with exactly those edges & labels
df_initial['final_bin'] = pd.cut(
    df_initial['Opposite_Charge_Number'],
    bins=edges,
    labels=labels,
    include_lowest=True
)

# 6) Quick sanity check
print(df_initial['final_bin'].value_counts().sort_index())
'''


'''n_bins = 100

# qcut → labels (0…n_bins-1) and the numeric edges
df_big['bin_id'], bin_edges = pd.qcut(
    df_big['Opposite_Charge_Number'],
    q=n_bins,
    labels=False,
    retbins=True,
    duplicates='drop'
)
# now each non-empty bin has ~len(df_big)/n_bins rows

# how many in each?
counts_big = df_big['bin_id'].value_counts().sort_index()
print(counts_big)

# use cut with the exact edges from your big set
df_initial['bin_id'] = pd.cut(
    df_initial['Opposite_Charge_Number'],
    bins=bin_edges,
    labels=False,
    include_lowest=True
)

counts_small = df_initial['bin_id'].value_counts().sort_index()
print(counts_small)

# 3) Now print or inspect the edges
print("Number of edges:", len(bin_edges))
print("Bin edges:\n", bin_edges)'''



'''# Define number of bins
num_bins = 100  

print(df_initial["Opposite_Charge_Number"].dtype)  # Should be float
print(df_initial["Opposite_Charge_Number"].isna().sum())  # Check for NaNs
print(df_initial["Opposite_Charge_Number"].describe())  # Check min/max values

### STEP 1: Bin df_initial ###
df_initial["Opposite_Charge_Bin"], bin_edges = pd.qcut(
    df_initial["Opposite_Charge_Number"], 
    q=num_bins, 
    retbins=True,  # Get bin edges
    duplicates="drop"
)

### STEP 2: Assign bins to df_big using the same bin edges ###
df_big["Opposite_Charge_Bin"] = pd.cut(
    df_big["Opposite_Charge_Number"], 
    bins=bin_edges, 
    labels=False,  # Get bin index (0 to num_bins-1)
    include_lowest=True
)

### STEP 3: Sample 100 points from df_big ###
bin_counts_initial = df_initial["Opposite_Charge_Bin"].value_counts(normalize=True)  # Proportion of bins
sampled_df = df_big.groupby("Opposite_Charge_Bin").apply(
    lambda x: x.sample(n=max(1, int(bin_counts_initial.get(x.name, 0) * 150)), random_state=42)
).reset_index(drop=True)

sampled_df.to_csv(f'{out_folder}/ocn_binned_pairs_added_features_data.csv')

# Ensure exactly 100 points by downsampling if needed
if len(sampled_df) > 100:
    sampled_df = sampled_df.sample(n=100, random_state=42)

print(f"Final dataset size: {len(sampled_df)}")

### STEP 4: Plot histogram to visualize binning ###
plt.figure(figsize=(12, 6))
sampled_df["Opposite_Charge_Bin"].value_counts().sort_index().plot(kind="bar", color="skyblue")

plt.xlabel("Bin Number")
plt.ylabel("Count")
plt.title("Sampled Points Distribution Across Bins")
plt.xticks(rotation=90)

plt.show()
'''

#store seqs
###results_df.to_csv(f'{out_folder}/all_pairs_sequences_data.csv')


'''with open("pairs_sequences.fasta", "w") as fasta_file:
    #print(f"File opened: {output_file}")
    #print("Data length: ", len(seq_dict))
    for header, sequence in results.items():
        #print(f"Writing header: {header}")
        fasta_file.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            fasta_file.write(sequence[i:i+80] + "\n")'''

'''# Apply filtering conditions
filtered_df = df[
    (df["Charge_Complementarity"] < 0) & 
    (df["Average_ncpr"].between(-0.05, 0.05)) & 
    (df["Average_lambda"].between(0.45, 0.56))
]'''

'''from sklearn.model_selection import train_test_split

filtered_df = df[
    (df["Charge_Complementarity"] >= 0) & 
    (~df["Average_ncpr"].between(-0.05, 0.05)) & 
    (~df["Average_lambda"].between(0.45, 0.56))
]

# Add a bin column to group `Charge_Complementarity` into strata
filtered_df["Charge_Complementarity_bin"] = pd.qcut(filtered_df["Charge_Complementarity"], q=10, duplicates='drop')

# Stratified sampling
_, random_sample = train_test_split(
    filtered_df,
    test_size=100 / len(filtered_df),
    stratify=filtered_df["Charge_Complementarity_bin"],
    random_state=42
)

# Drop the bin column if no longer needed
random_sample = random_sample.drop(columns=["Charge_Complementarity_bin"])
'''


'''filtered_df = df[
    (df["Charge_Complementarity"] >= 0) & 
    (~df["Average_ncpr"].between(-0.05, 0.05)) & 
    (~df["Average_lambda"].between(0.45, 0.56))
]

random_sample = filtered_df.sample(n=100, random_state=42)

random_sample.to_csv("filtered_outside_features_data.csv", index=False)'''

'''# Load filtered csv
filtered_file = "filtered_outside_features_data.csv"
filtered_df = pd.read_csv(filtered_file)


# Initialize the final list of selected rows
final_rows = []

# Keep track of used Rggs
used_rggs = set()

# Shuffle the order of unique `tad` to avoid biases
random.shuffle(tads_list)


# Process each `tad` group
for tad in tads_list:
    # Get all rows for the current `tad`
    tad_group = filtered_df[filtered_df["tad"] == tad].sort_values(by="Total_Length")
    
    # Remove rows with already used Rggs
    tad_group = tad_group[~tad_group["rgg"].isin(used_rggs)]
    
    # Skip if the group becomes empty after filtering
    if tad_group.empty:
        continue
    
    # Select the row with the smallest length as the first row
    first_row = tad_group.iloc[0]
    used_rggs.add(first_row["rgg"])  # Mark the Rgg as used
    
    # Now look for the second row with the highest length difference from the first row
    tad_group_rest = tad_group.iloc[1:]  # Exclude the first row
    if not tad_group_rest.empty:
        # Compute the length differences and sort by descending difference
        tad_group_rest = tad_group_rest.assign(Length_Difference=abs(tad_group_rest["Total_Length"] - first_row["Total_Length"]))
        tad_group_rest = tad_group_rest.sort_values(by="Length_Difference", ascending=False)
        
        # Find the first valid row with an unused Rgg
        second_row = None
        for _, row in tad_group_rest.iterrows():
            if row["rgg"] not in used_rggs:
                second_row = row
                used_rggs.add(second_row["rgg"])  # Mark the Rgg as used
                break
        
        if second_row is not None:
            final_rows.extend([first_row, second_row])
        else:
            # If no valid second row is found, just add the first row
            final_rows.append(first_row)
    else:
        # If no other entries exist, only add the first row
        final_rows.append(first_row)

# Convert the final list of rows to a DataFrame
final_df = pd.DataFrame(final_rows)

# Drop the temporary 'Length_Difference' column if it exists
if "Length_Difference" in final_df.columns:
    final_df = final_df.drop(columns=["Length_Difference"])

# Display the final DataFrame
print(final_df)

# Save the results to a new CSV if needed
final_df.to_csv("filtered_with_length_diversity.csv", index=False)'''




'''# Group by 'tad' and select the row with the lowest Charge_Complementarity for each group
result_df = filtered_df.loc[filtered_df.groupby("tad")["Charge_Complementarity"].idxmin()]

# Display the results
#print(result_df)

# Save the results to a new CSV if needed
#result_df.to_csv("filtered_and_grouped_results.csv", index=False)

# Initialize the set to track used `rgg` values
used_rgg = set()
final_rows = []
random.shuffle(tads_list)
# Process each `tad` in the randomized order
for tad in tads_list:
    # Filter rows for the current `tad` and sort by Charge_Complementarity
    tad_group = filtered_df[filtered_df["tad"] == tad].sort_values(by="Charge_Complementarity")
    
    # Iterate through the rows for this `tad`
    for _, row in tad_group.iterrows():
        rgg = row["rgg"]
        if rgg not in used_rgg:
            # If the `rgg` hasn't been used, add this row and mark `rgg` as used
            final_rows.append(row)
            used_rgg.add(rgg)
            break

# Convert the selected rows to a DataFrame
final_df = pd.DataFrame(final_rows)

# Save the results to a new CSV if needed
final_df.to_csv("unique_rgg_per_tad_results.csv", index=False)'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fisher_ci(r, n, alpha=0.05):
    """95% CI for Pearson r using Fisher z transform."""
    if n <= 3 or np.isclose(abs(r), 1):
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zcrit = norm.ppf(1 - alpha/2)
    lo, hi = z - zcrit*se, z + zcrit*se
    return (np.tanh(lo), np.tanh(hi))

def scatter_with_corr(ax, df, xcol, ycol, method="pearson", show_fit=True, point_kwargs=None, line_kwargs=None):
    """
    Plot scatter of df[xcol] vs df[ycol], annotate correlation (r, p, n), and draw a fit line.
    method: 'pearson' | 'spearman' | 'kendall'
    """
    point_kwargs = point_kwargs or dict(s=12, alpha=0.7)
    line_kwargs  = line_kwargs  or dict(lw=1)

    # coerce numeric and drop NaNs
    s = df[[xcol, ycol]].apply(pd.to_numeric, errors='coerce').dropna()
    n = len(s)
    if n < 3:
        ax.text(0.05, 0.95, f"n={n} (too few)", transform=ax.transAxes, va="top")
        return

    # correlation
    try:
        from scipy import stats
        if method == "pearson":
            r, p = stats.pearsonr(s[xcol], s[ycol])
        elif method == "spearman":
            r, p = stats.spearmanr(s[xcol], s[ycol])
        elif method == "kendall":
            r, p = stats.kendalltau(s[xcol], s[ycol])
        else:
            raise ValueError("method must be 'pearson', 'spearman', or 'kendall'")
    except Exception:
        # fallback if SciPy isn’t available (Pearson only, no p-value)
        r = np.corrcoef(s[xcol], s[ycol])[0,1]
        p = np.nan

    # plot points
    ax.scatter(s[xcol], s[ycol], **point_kwargs)

    # optional fit line (simple OLS)
    if show_fit:
        try:
            from scipy.stats import linregress
            lr = linregress(s[xcol], s[ycol])
            xline = np.linspace(s[xcol].min(), s[xcol].max(), 100)
            yline = lr.intercept + lr.slope * xline
            ax.plot(xline, yline, **line_kwargs)
        except Exception:
            # numpy fallback
            m, b = np.polyfit(s[xcol], s[ycol], 1)
            xline = np.linspace(s[xcol].min(), s[xcol].max(), 100)
            ax.plot(xline, m*xline + b, **line_kwargs)

    # CI for Pearson r (only meaningful for Pearson)
    ci_txt = ""
    if method == "pearson" and not np.isnan(p):
        lo, hi = fisher_ci(r, n)
        if not np.isnan(lo):
            ci_txt = f" (95% CI {lo:.2f}–{hi:.2f})"

    # annotate
    ax.text(
        0.05, 0.95,
        f"{method.title()} r={r:.2f}, p={p:.1e}, n={n}{ci_txt}",
        transform=ax.transAxes, va="top", ha="left"
    )
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)


# Save scatter plots
def save_scatter_plots(df, x_features, y_features, output_folder):
    # Check for missing or non-numeric data in the features
    for x in x_features:
        for y in y_features:
            # Ensure data is numeric
            df[x] = pd.to_numeric(df[x], errors='coerce')
            df[y] = pd.to_numeric(df[y], errors='coerce')

            # Remove rows with NaN or non-positive y values
            df_clean = df.dropna(subset=[x, y])

            # Apply log10 transformation to the absolute values of y
            df_clean[y] = -df_clean[y]

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.scatter(df_clean[x], df_clean[y], alpha=0.7, color='blue', edgecolor='black', s=50)
            ##plt.ylim(150, 31000) ##for hummer correction the limits of b2
            ###plt.xscale('linear')  # Linear scale for x-axis
            ###plt.yscale('log')  # Linear scale for y-axis since y is already log-transformed
            plt.xlabel(x)
            plt.ylabel(f"{y}")
            plt.title(f"{y} vs {x}")
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{y}_vs_{x}_b22_scatter.png", dpi=300)
            plt.close()

# Save scatter plots
def save_scatter_plots_with_corr(df, x_features, y_features, output_folder):
    # Check for missing or non-numeric data in the features
    for x in x_features:
        for y in y_features:
            # Ensure data is numeric
            df[x] = pd.to_numeric(df[x], errors='coerce')
            df[y] = pd.to_numeric(df[y], errors='coerce')

            # Remove rows with NaN or non-positive y values
            df_clean = df.dropna(subset=[x, y])

            # Apply log10 transformation to the absolute values of y
            #df_clean[y] = -df_clean[y]

            # Plotting
            plt.figure(figsize=(10, 5))
            plt.scatter(df_clean[x], df_clean[y], alpha=0.7, color='blue', edgecolor='black', s=50)
            ##plt.ylim(150, 31000) ##for hummer correction the limits of b2
            ###plt.xscale('linear')  # Linear scale for x-axis
            ###plt.yscale('log')  # Linear scale for y-axis since y is already log-transformed
            
            fig, ax = plt.subplots(figsize=(5,4), dpi=150)
            scatter_with_corr(ax, results_df, x, y, method="pearson")
            plt.xlabel(x)
            plt.ylabel(f"{y}")
            plt.title(f"{y} vs {x}")
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(f"{output_folder}/{y}_vs_{x}_b22_scatter_pred_with_corr.png", dpi=300)
            plt.close()


def save_bar_plots_with_binning(df, x_features, y_features, output_folder, bins=50):
    """
    Saves plots of binned y_features vs x_features with means and error bars.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_features (list): List of column names to use as x-axis features.
        y_features (list): List of column names to use as y-axis features.
        output_folder (str): Folder path to save the plots.
        bins (int): Number of bins for grouping data points. Default is 50.
    """
    for x in x_features:
        for y in y_features:
            # Ensure data is numeric
            df[x] = pd.to_numeric(df[x], errors='coerce')
            df[y] = pd.to_numeric(df[y], errors='coerce')
            
            # Remove rows with NaN values
            df_clean = df.dropna(subset=[x, y])
            
            # Binning
            bin_edges = np.linspace(df_clean[x].min(), df_clean[x].max(), bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            binned_means = []
            binned_stds = []
            
            for i in range(bins):
                bin_data = df_clean[(df_clean[x] >= bin_edges[i]) & (df_clean[x] < bin_edges[i + 1])][y]
                binned_means.append(bin_data.mean())
                binned_stds.append(bin_data.std())
            
            # Plotting
            plt.figure(figsize=(10, 5))
            
            # Bar plot
            plt.bar(
                bin_centers, -np.array(binned_means), width=(bin_edges[1] - bin_edges[0]),
                alpha=0.2, color='skyblue', label="Binned Data"
            )
            
            # Error bars
            plt.errorbar(
                bin_centers, -np.array(binned_means), yerr=binned_stds, fmt='o', color='black',
                ecolor='grey', elinewidth=1.5, capsize=3, label="Mean with Std Dev"
            )
            
            # Scales and labels
            ###plt.xscale('linear')  # Default linear scale for x-axis
            ###plt.yscale('log')     # Logarithmic scale for y-axis
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"Binned {y} vs {x} (Bar Plot with Error Bars)")
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(f"{output_folder}/{y}_vs_{x}_binned_with_error_bars.png", dpi=300)
            plt.close()


def plots_with_log_b22(df, x_features, y_features, output_folder, bins=100):
    """
    Plots the log10 of b22 values with error bars for binned data.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_column (str): Column name for the x-axis.
        y_column (str): Column name for the y-axis (e.g., 'b22').
        bins (int): Number of bins for grouping data points. Default is 50.
    """
    for x in x_features:
        for y in y_features:
            # Ensure data is numeric and clean NaNs
            df[x] = pd.to_numeric(df[x], errors='coerce')
            df[y] = pd.to_numeric(df[y], errors='coerce')
            df_clean = df.dropna(subset=[x, y])
            
            # Take log10 of b22 values
            df_clean[y] = np.log10(df_clean[y])
            
            # Binning
            bin_edges = np.linspace(df_clean[x].min(), df_clean[x].max(), bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            binned_means = []
            binned_stds = []

            for i in range(bins):
                bin_data = df_clean[(df_clean[x] >= bin_edges[i]) & (df_clean[x] < bin_edges[i + 1])][y]
                binned_means.append(bin_data.mean())
                binned_stds.append(bin_data.std())

            
            # Convert to numpy arrays for plotting
            binned_means = np.array(binned_means)
            binned_stds = np.array(binned_stds)
            
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.errorbar(
                bin_centers, binned_means, yerr=binned_stds, fmt='o', color='black',
                ecolor='red', elinewidth=1.5, capsize=3, label="Log10 Mean with Std Dev"
            )
            plt.xscale('linear')  # Default linear scale for x-axis
            plt.yscale('linear')  # Linear scale since y-axis is already log-transformed
            plt.xlabel(x)
            plt.ylabel(f"log10({y})")
            plt.title(f"log10({y}) vs {x}")
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.legend()
            # Save the plot
            plt.savefig(f"{output_folder}/{y}_vs_{x}_log_b22s.png", dpi=300)
            plt.close()

def save_scatter_plots_with_sub(df_main, df_sub, x_features, y_features, output_folder):
    """
    For each combination of x in x_features and y in y_features, 
    clean both df_main and df_sub, then plot them together.
    """
    for x in x_features:
        for y in y_features:
            # Copy & coerce to numeric
            main = df_main[[x, y]].copy()
            sub  = df_sub[[x, y]].copy()
            main[x] = pd.to_numeric(main[x], errors='coerce')
            main[y] = pd.to_numeric(main[y], errors='coerce')
            sub[x]  = pd.to_numeric(sub[x], errors='coerce')
            sub[y]  = pd.to_numeric(sub[y], errors='coerce')

            # Drop NaNs
            main = main.dropna(subset=[x, y])
            sub  = sub.dropna(subset=[x, y])

            # (Optional) transform y if you need to flip or log it
            #main[y] = -main[y]
            #sub[y]  = -sub[y]

            # Plot
            plt.figure(figsize=(10, 5))
            plt.scatter(main[x], main[y],
                        alpha=0.7, color='blue',
                        edgecolor='black', s=50,
                        label='Main dataset')
            if not sub.empty:
                plt.scatter(sub[x], sub[y],
                            alpha=0.9, color='red',
                            edgecolor='black', s=60,
                            label='Sub-dataset')

            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f"{y} vs {x}")
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout()

            # Save
            outname = f"{output_folder}/{y}_vs_{x}_b22_scatter_with_pairs_in_lab.png"
            plt.savefig(outname, dpi=300)
            plt.close()
            print(f"Saved: {outname}")


#sim_pairs_data_file = f'{out_folder}/sim_pairs_with_kds_data.csv'
#results_df = pd.read_csv(sim_pairs_data_file)

#pred_pairs_data_file = f'{in_folder}/output_mlp_merged.csv'
#results_df = pd.read_csv(pred_pairs_data_file)
# Define features for plotting
x_features = ["Average_lambda", "Min_lambda", "Average_sigma", "Average_charge", "Average_NCPR", "Charge_Complementarity", "Total_Length", "Average_Interaction", "Average_SCD", "Average_SHD", "Opposite_Charge_Number", "RGG_length", "charge_RGG", "lambda_RGG", "NCPR_RGG", "SCD_RGG", "SHD_RGG", "TAD_length", "charge_TAD", "lambda_TAD", "NCPR_TAD", "SCD_TAD", "SHD_TAD"]
#y_features = ["B22", "Corrected_B22_Hummer", "Corrected_B22_Ganguly"]
#y_features= ["Corrected_B22_Hummer"]
#y_features= ["B22_predicted"]
y_features= ["Corrected_Kd_Hummer"]

# Generate plots
##save_scatter_plots(results_df, x_features, y_features, out_folder)
##save_bar_plots_with_binning(results_df, x_features, y_features, out_folder)
#plots_with_log_b22(results_df, x_features, y_features, out_folder)
#pred_pairs_data_file = f'../output_xgb.csv'

#results_df = pd.read_csv(f"{in_folder}/sim_pairs_added_features_data.csv")

###save_scatter_plots(results_df, x_features, y_features, out_folder)

###save_scatter_plots_with_corr(results_df, x_features, y_features, out_folder)

###save_bar_plots_with_binning(results_df, x_features, y_features, out_folder)

##data_main = pd.read_csv(f"{in_folder}/output_mlp_merged.csv")
##data_sub = pd.read_csv(f"{in_folder}/validated_1.csv")
##out_folder = f"/Users/yuktikhanna/calvados/pred_figs_sub_{args.date:s}/val1/"
##os.makedirs(out_folder, exist_ok=True)
##save_scatter_plots_with_sub(data_main, data_sub, x_features, y_features, out_folder)
