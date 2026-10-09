"""
tests/generate_synthetic_data.py
─────────────────────────────────
Creates small synthetic versions of all four required data files
so the CI pipeline can run end-to-end without the real (large) datasets.

Generates files at the same paths config.py expects:
  data/raw/tp53/tp53_mutations.csv
  data/raw/string/string_links_human_v12.txt
  data/raw/string/9606_protein_info_v12_0.txt
  data/raw/biogrid/biogrid_all_4.4.243.tab3.txt
"""

import os
import sys
import numpy as np
import pandas as pd

# Make sure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    P53_DATASET, STRING_LINKS, STRING_PROTEIN_INFO, BIOGRID_FILE, DATA_DIR
)

RNG = np.random.default_rng(42)
N   = 800      # small but representative

# ─────────────────────────────────────────────────────────────────────────────
#  1. tp53_mutations.csv
# ─────────────────────────────────────────────────────────────────────────────

def make_tp53_csv():
    os.makedirs(os.path.dirname(P53_DATASET), exist_ok=True)

    domains   = ["DNA binding", "HCD IV - DNA binding", "HCD V - DNA binding",
                 "Oligomerization", "Transactivation TAD1", "Proline Rich"]
    aa_list   = list("ACDEFGHIKLMNPQRSTVWY")
    mut_events= ["G>A", "C>T", "G>T", "A>G", "C>A"]
    diseases  = ["Breast adenocarcinoma", "Lung carcinoma", "Colorectal carcinoma",
                 "Thyroid carcinoma", "Ovarian carcinoma", "Bladder carcinoma"]

    codon  = RNG.integers(1, 394, N)
    wt_aa  = RNG.choice(aa_list, N)
    mut_aa = RNG.choice(aa_list, N)
    activity = RNG.uniform(0, 200, (N, 8))

    df = pd.DataFrame({
        "Database_ID":        np.arange(1, N + 1),
        "UMD_ID":             [f"UMD_{i}" for i in range(N)],
        "COSMIC_ID":          [f"COSM{i}" for i in range(N)],
        "HG38_Start":         RNG.integers(7_670_000, 7_690_000, N),
        "HG38_End":           RNG.integers(7_670_000, 7_690_000, N),
        "HG38_Variant":       [f"chr17:g.{x}C>T" for x in RNG.integers(7_670_000, 7_690_000, N)],
        "NG_017013.2_Variant":[f"NG_017013.2:g.{x}C>T" for x in RNG.integers(10000, 20000, N)],
        "Start_cDNA":         codon * 3,
        "End_cDNA":           codon * 3,
        "Exon:intron_start":  RNG.integers(1, 12, N),
        "Exon:intron_End":    RNG.integers(1, 12, N),
        "Genome_base_coding": RNG.choice(["C", "G", "A", "T"], N),
        "cDNA_variant":       [f"c.{c*3}C>T" for c in codon],
        "Codon":              codon,
        "WT_AA_1":            wt_aa,
        "WT_AA_3":            ["Ala"] * N,
        "Mutant_AA_1":        mut_aa,
        "Mutant_AA_3":        ["Gly"] * N,
        "Domain":             RNG.choice(domains, N),
        "Base_Change_Size":   RNG.choice([1, 2, 3], N),
        "Ins_Size":           np.zeros(N, int),
        "Del_Size":           np.zeros(N, int),
        "Mutant_Allele":      RNG.choice(["A", "C", "G", "T"], N),
        "Wt_Codon":           ["CGT"] * N,
        "Mutant_Codon":       ["TGT"] * N,
        "Mutational_event":   RNG.choice(mut_events, N),
        "Type":               RNG.choice(["Ts", "Tv"], N),
        "Mutation_Type":      ["B"] * N,
        "CpG":                RNG.choice(["Yes", "No"], N),
        "Variant_Classification": RNG.choice(["Missense_Mutation", "Nonsense_Mutation"], N),
        "Variant Comment":    ["Exonic mutation"] * N,
        "Variant_Type":       ["SNP"] * N,
        "Mutation_origin":    RNG.choice(["Somatic", "Germline"], N, p=[0.98, 0.02]),
        "Disease":            RNG.choice(diseases, N),
        "":                   RNG.uniform(0, 1, N),
        "Sample_pathology":   ["Cancer"] * N,
        "Sample_origin":      ["Tumour"] * N,
        "Internal":           ["Internal tumour"] * N,
        "Solid":              ["Solid tumour"] * N,
        "Complexity":         ["SM"] * N,
        "Unidentified_Mutations": np.zeros(N, int),
        "Leukemia_Stat":      RNG.uniform(0, 5, N),
        "Solid_Stat":         RNG.uniform(0, 5, N),
        "Tumor_Stat":         RNG.uniform(0, 5, N),
        "Cell_line_Stat":     RNG.uniform(0, 5, N),
        "Somatic_Stat":       RNG.uniform(0, 5, N),
        "Germline_Stat":      RNG.uniform(0, 5, N),
        "Tumor_Repetition":   RNG.integers(1, 10, N),
        "Sift Score":         RNG.uniform(0, 1, N),
        "Sift Prediction":    RNG.choice(["Tolerated", "Damaging"], N),
        "Provean_Score":      RNG.uniform(-10, 5, N),
        "Provean_prediction": RNG.choice(["Neutral", "Deleterious"], N),
        "Mutassessor_score":  RNG.uniform(0, 6, N),
        "Mutassessor_prediction:": RNG.choice(["neutral", "low", "medium", "high"], N),
        "Polyphen":           RNG.uniform(0, 1, N),
        "Condel":             RNG.uniform(0, 1, N),
        "MutPred_Splice_General_Score": RNG.uniform(0, 1, N),
        "Prediction_Label":   RNG.choice(["Splice Affecting Variant (SAV)", "Non-SAV"], N),
        "Transcript t1":      ["c.817C>T"] * N,
        "Transcript t2":      ["c.817C>T"] * N,
        "Transcript t3":      ["c.817C>T"] * N,
        "Transcript t4":      ["c.817C>T"] * N,
        "Transcript t5":      ["c.421C>T"] * N,
        "Transcript t6":      ["c.421C>T"] * N,
        "Transcript t7":      ["c.421C>T"] * N,
        "Transcript t8":      ["c.700C>T"] * N,
        "WAF1_":              activity[:, 0],
        "MDM2_":              activity[:, 1],
        "BAX_":               activity[:, 2],
        "__14_3_3_s":         activity[:, 3],
        "AIP_":               activity[:, 4],
        "GADD45_":            activity[:, 5],
        "NOXA_":              activity[:, 6],
        "p53R2_":             activity[:, 7],
        "PCA_Score":          RNG.uniform(0, 1, N),
        "Pathogenicity":      RNG.choice(
            ["Pathogenic", "Likely Pathogenic", "Possibly pathogenic", "VUS", "Benign"],
            N, p=[0.50, 0.15, 0.20, 0.10, 0.05]
        ),
    })

    df.to_csv(P53_DATASET, index=False)
    print(f"✓ Synthetic TP53 dataset  -> {P53_DATASET}  ({N} rows)")


# ─────────────────────────────────────────────────────────────────────────────
#  2. string_links_human_v12.txt  (just ~50 TP53 interactions)
# ─────────────────────────────────────────────────────────────────────────────

TP53_STRING_ID = "9606.ENSP00000269305"
PARTNER_IDS    = [f"9606.ENSP{str(i).zfill(11)}" for i in range(100, 160)]

def make_string_links():
    os.makedirs(os.path.dirname(STRING_LINKS), exist_ok=True)
    rows = [("protein1", "protein2", "combined_score")]
    for pid in PARTNER_IDS:
        score = RNG.integers(400, 1000)
        rows.append((TP53_STRING_ID, pid, str(score)))
    with open(STRING_LINKS, "w") as f:
        for r in rows:
            f.write(" ".join(r) + "\n")
    print(f"✓ Synthetic STRING links  -> {STRING_LINKS}  ({len(rows)-1} interactions)")


def make_string_info():
    os.makedirs(os.path.dirname(STRING_PROTEIN_INFO), exist_ok=True)
    gene_names = [
        "MDM2", "MDM4", "BRCA1", "ATM", "CHEK2", "PTEN", "BCL2", "BAX",
        "CDKN1A", "EP300", "SIRT1", "RB1", "VHL", "KRAS", "AKT1", "MYC",
        "CREBBP", "ASPP1", "ASPP2", "HIPK2", "TIGAR", "YWHAZ", "CHEK1",
        "USP7", "COP1", "PIRH2", "HDAC1", "PARP1", "HMGB1", "XPO1",
        "RASSF1", "KPNA2", "PLK3", "VRK1", "PPM1D", "NF1", "CDK4",
        "SMAD2", "BRAF", "MYCN", "EGFR", "ALK", "PIK3CA", "STK11",
        "CDK6", "SMAD3", "SMAD4", "RELA", "BBC3", "GADD45A",
    ] + [f"GENE{i}" for i in range(len(PARTNER_IDS) - 50)]

    rows = [("#string_protein_id\tpreferred_name\tprotein_size\tannotation")]
    rows.append(f"{TP53_STRING_ID}\tTP53\t393\tTumor suppressor protein p53")
    for pid, gname in zip(PARTNER_IDS, gene_names):
        rows.append(f"{pid}\t{gname}\t{RNG.integers(200,1000)}\tProtein annotation")
    with open(STRING_PROTEIN_INFO, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"✓ Synthetic STRING info   -> {STRING_PROTEIN_INFO}  ({len(rows)-1} proteins)")


# ─────────────────────────────────────────────────────────────────────────────
#  3. biogrid_all_4.4.243.tab3.txt
# ─────────────────────────────────────────────────────────────────────────────

BIOGRID_GENE_PARTNERS = [
    "MDM2", "MDM4", "BRCA1", "ATM", "CHEK2", "EP300",
    "SIRT1", "PARP1", "XPO1", "HIPK2", "ASPP1", "ASPP2",
    "YWHAZ", "KPNA2", "HDAC1", "CREBBP", "VHL", "PTEN",
    "TIGAR", "BCL2",
]

def make_biogrid():
    os.makedirs(os.path.dirname(BIOGRID_FILE), exist_ok=True)
    phys_systems = [
        "Affinity Capture-MS", "Two-hybrid", "Affinity Capture-Western",
        "Biochemical Activity", "Co-crystal Structure",
    ]
    header = "\t".join([
        "#BioGRID Interaction ID",
        "Entrez Gene Interactor A", "Entrez Gene Interactor B",
        "BioGRID ID Interactor A", "BioGRID ID Interactor B",
        "Systematic Name Interactor A", "Systematic Name Interactor B",
        "Official Symbol Interactor A", "Official Symbol Interactor B",
        "Synonyms Interactor A", "Synonyms Interactor B",
        "Experimental System", "Experimental System Type",
        "Author", "Publication Source",
        "Organism ID Interactor A", "Organism ID Interactor B",
        "Throughput", "Score", "Modification",
        "Qualifications", "Tags", "Source Database",
        "SWISS-PROT Accessions Interactor A",
        "SWISS-PROT Accessions Interactor B",
        "Ontology Term IDs", "Ontology Term Names",
        "Ontology Term Categories",
        "Ontology Term Qualifier IDs", "Ontology Term Qualifier Names",
        "Ontology Term Types", "Organism Name Interactor A",
        "Organism Name Interactor B",
    ])
    rows = [header]
    for i, partner in enumerate(BIOGRID_GENE_PARTNERS):
        sys_name = RNG.choice(phys_systems)
        rows.append("\t".join([
            str(i + 1000),          # Interaction ID
            "7157", str(i + 10000), # Entrez IDs
            "108352", str(i + 200), # BioGRID IDs
            "TP53", partner,         # Systematic names
            "TP53", partner,         # Official symbols
            "-", "-",               # Synonyms
            sys_name, "physical",   # Experimental system
            "Smith 2020", "PUBMED:12345678",
            "9606", "9606",         # Organism IDs
            "Low Throughput", "-",  # Throughput, Score
            "-", "-", "-", "BioGRID",
            "P04637", "-",          # SWISS-PROT
            "-", "-", "-", "-", "-", "-", "-",
            "Homo sapiens", "Homo sapiens",
        ]))

    with open(BIOGRID_FILE, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"✓ Synthetic BioGRID file  -> {BIOGRID_FILE}  ({len(rows)-1} interactions)")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating synthetic datasets for CI …\n")
    make_tp53_csv()
    make_string_links()
    make_string_info()
    make_biogrid()
    print("\nAll synthetic data files created successfully.")
