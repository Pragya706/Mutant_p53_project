"""
ppi_graph_builder.py  –  Build a Protein–Protein Interaction subgraph for TP53
Sources: STRING v12  (string_links_human_v12.txt)
         BioGRID 4.4 (biogrid_all_4.4.243.tab3.txt)
         STRING protein info (9606_protein_info_v12_0.txt)
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import networkx as nx

from config import (
    STRING_LINKS, STRING_PROTEIN_INFO, BIOGRID_FILE,
    PPI_GRAPH_CACHE, OUTPUT_DIR,
    TP53_STRING_ID, TP53_GENE_SYMBOL, HUMAN_TAXON_ID,
    STRING_SCORE_THRESHOLD, BIOGRID_PHYSICAL_SYSTEMS,
    TP53_DOMAINS, TP53_CANCER_DRIVER_INTERACTORS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  STRING loader
# ─────────────────────────────────────────────────────────────────────────────

def load_string_interactions(
    links_file: str,
    protein_info_file: str,
    score_threshold: int = STRING_SCORE_THRESHOLD,
    tp53_id: str = TP53_STRING_ID,
) -> pd.DataFrame:
    """
    Load STRING v12 human protein links, filter for TP53 neighbourhood.
    Returns DataFrame with columns: gene_a, gene_b, combined_score (0-1).
    """
    logger.info("Loading STRING protein info …")
    info = pd.read_csv(protein_info_file, sep="\t", usecols=[0, 1])
    info.columns = ["string_id", "gene_name"]
    id_to_gene = dict(zip(info["string_id"], info["gene_name"]))

    logger.info("Loading STRING links (this may take a moment for large file) …")
    # STRING links can be space-separated or tab-separated
    try:
        links = pd.read_csv(
            links_file,
            sep=r"\s+",
            usecols=[0, 1, 2],
            names=["protein1", "protein2", "combined_score"],
            skiprows=1,
            engine="python",
        )
    except Exception:
        links = pd.read_csv(
            links_file,
            sep="\t",
            usecols=[0, 1, 2],
            names=["protein1", "protein2", "combined_score"],
            skiprows=1,
        )

    logger.info(f"  Total STRING links loaded: {len(links):,}")

    # Filter: at least one side must be TP53
    mask = (links["protein1"] == tp53_id) | (links["protein2"] == tp53_id)
    tp53_links = links[mask & (links["combined_score"] >= score_threshold)].copy()

    # Normalise score to 0-1
    tp53_links["combined_score"] = tp53_links["combined_score"] / 1000.0

    # Map IDs to gene names
    tp53_links["gene_a"] = tp53_links["protein1"].map(id_to_gene).fillna(tp53_links["protein1"])
    tp53_links["gene_b"] = tp53_links["protein2"].map(id_to_gene).fillna(tp53_links["protein2"])

    logger.info(f"  TP53 STRING interactions (score >= {score_threshold}): {len(tp53_links)}")
    return tp53_links[["gene_a", "gene_b", "combined_score"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  BioGRID loader
# ─────────────────────────────────────────────────────────────────────────────

def load_biogrid_interactions(
    biogrid_file: str,
    tp53_symbol: str = TP53_GENE_SYMBOL,
    taxon_id: int = HUMAN_TAXON_ID,
    physical_systems: set = BIOGRID_PHYSICAL_SYSTEMS,
) -> pd.DataFrame:
    """
    Load BioGRID tab3 file and extract TP53 interactions.
    Returns DataFrame with columns: gene_a, gene_b, experimental_system, score.
    """
    logger.info("Loading BioGRID interactions …")
    usecols = [
        "Official Symbol Interactor A",
        "Official Symbol Interactor B",
        "Experimental System",
        "Experimental System Type",
        "Organism ID Interactor A",
        "Organism ID Interactor B",
        "Score",
    ]
    try:
        bg = pd.read_csv(
            biogrid_file,
            sep="\t",
            usecols=usecols,
            low_memory=False,
            on_bad_lines="skip",
        )
    except ValueError:
        # Some BioGRID versions differ slightly in column names
        bg = pd.read_csv(biogrid_file, sep="\t", low_memory=False, on_bad_lines="skip")
        # Rename to standard names
        rename = {c: c.strip() for c in bg.columns}
        bg.rename(columns=rename, inplace=True)

    logger.info(f"  Total BioGRID records: {len(bg):,}")

    # Filter for human × human
    bg = bg[
        (bg["Organism ID Interactor A"].astype(str) == str(taxon_id)) &
        (bg["Organism ID Interactor B"].astype(str) == str(taxon_id))
    ]

    # Filter for TP53 involvement
    bg = bg[
        (bg["Official Symbol Interactor A"].str.upper() == tp53_symbol.upper()) |
        (bg["Official Symbol Interactor B"].str.upper() == tp53_symbol.upper())
    ]

    # Filter for physical interactions
    bg = bg[bg["Experimental System"].isin(physical_systems)]

    bg["gene_a"] = bg["Official Symbol Interactor A"].str.upper()
    bg["gene_b"] = bg["Official Symbol Interactor B"].str.upper()

    # Normalize BioGRID score (often "-" for non-scored entries)
    def safe_score(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return 0.5   # default medium confidence for physical interaction

    bg["score"] = bg["Score"].apply(safe_score)

    # Normalise to 0-1 if raw values are high
    if bg["score"].max() > 1:
        bg["score"] = bg["score"] / bg["score"].max()

    logger.info(f"  TP53 BioGRID physical interactions: {len(bg)}")
    return bg[["gene_a", "gene_b", "score"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Merge STRING + BioGRID into a unified PPI graph
# ─────────────────────────────────────────────────────────────────────────────

def build_tp53_ppi_graph(
    string_df: pd.DataFrame | None = None,
    biogrid_df: pd.DataFrame | None = None,
) -> nx.Graph:
    """
    Merge STRING and BioGRID interactions into a weighted, undirected NetworkX graph.
    Nodes: gene symbols.  Edge attribute: 'weight' (mean confidence 0-1).
    """
    G = nx.Graph()

    def add_edges(df, score_col):
        for _, row in df.iterrows():
            a = str(row["gene_a"]).upper()
            b = str(row["gene_b"]).upper()
            w = float(row[score_col])
            if G.has_edge(a, b):
                # Take max if duplicate
                G[a][b]["weight"] = max(G[a][b]["weight"], w)
                G[a][b]["sources"].add("STRING" if score_col == "combined_score" else "BioGRID")
            else:
                G.add_edge(a, b, weight=w,
                           sources={"STRING" if score_col == "combined_score" else "BioGRID"})

    if string_df is not None and len(string_df):
        add_edges(string_df, "combined_score")
    if biogrid_df is not None and len(biogrid_df):
        add_edges(biogrid_df, "score")

    logger.info(
        f"PPI graph: {G.number_of_nodes()} proteins, {G.number_of_edges()} interactions"
    )
    return G


# ─────────────────────────────────────────────────────────────────────────────
#  PPI-derived feature extraction for mutations
# ─────────────────────────────────────────────────────────────────────────────

def compute_ppi_features(
    G: nx.Graph,
    tp53_node: str = "TP53",
) -> dict:
    """
    Compute network-level statistics that will be used as features per mutation.
    Returns a dict of global features + per-neighbour features.
    """
    if tp53_node not in G:
        logger.warning("TP53 node not found in PPI graph – using empty PPI features")
        return {"ppi_degree": 0, "ppi_neighbours": [], "ppi_neighbour_weights": {}}

    neighbours = list(G.neighbors(tp53_node))
    weights     = {n: G[tp53_node][n]["weight"] for n in neighbours}
    driver_hits = {n for n in neighbours if n in TP53_CANCER_DRIVER_INTERACTORS}

    # Degree and weighted degree
    ppi_degree          = len(neighbours)
    ppi_weighted_degree = sum(weights.values())
    mean_weight         = ppi_weighted_degree / max(ppi_degree, 1)
    driver_score        = sum(weights[n] for n in driver_hits)

    # Compute betweenness centrality (on subgraph for speed)
    subgraph_nodes = {tp53_node} | set(neighbours)
    sg = G.subgraph(subgraph_nodes)
    try:
        bc = nx.betweenness_centrality(sg, weight="weight", normalized=True)
        tp53_bc = bc.get(tp53_node, 0.0)
    except Exception:
        tp53_bc = 0.0

    logger.info(
        f"TP53 PPI degree={ppi_degree}, weighted_degree={ppi_weighted_degree:.2f}, "
        f"driver_interactors={len(driver_hits)}"
    )

    return {
        "ppi_degree":           ppi_degree,
        "ppi_weighted_degree":  ppi_weighted_degree,
        "ppi_mean_weight":      mean_weight,
        "ppi_driver_score":     driver_score,
        "ppi_n_drivers":        len(driver_hits),
        "ppi_betweenness":      tp53_bc,
        "ppi_neighbours":       neighbours,
        "ppi_neighbour_weights": weights,
        "ppi_driver_interactors": list(driver_hits),
    }


def domain_ppi_features(
    G: nx.Graph,
    domain_interface_map: dict = None,
    tp53_node: str = "TP53",
) -> dict:
    """
    Map TP53 structural domains to known interaction partners in the PPI graph.
    Returns a dict:  domain_name -> {n_partners_in_graph, max_score, mean_score, has_driver}
    """
    if domain_interface_map is None:
        # Build from curated known interfaces
        # Loosely based on published TP53 interactome papers
        domain_interface_map = {
            "Transactivation TAD1":   ["MDM2", "MDM4", "TAFII31", "TAFII70", "EP300", "CREBBP", "RELA"],
            "Transactivation TAD2":   ["MDM2", "NCOA3", "HMGB1", "PARP1"],
            "Proline Rich":           ["PMAIP1", "SH3BP5", "BCL2"],
            "DNA binding":            ["BRCA1", "ASPP1", "ASPP2", "53BP1", "BCL-XL", "HIPK2", "PTEN"],
            "HCD II - DNA binding":   ["BRCA1", "ASPP1", "ASPP2"],
            "HCD III - DNA binding":  ["BRCA1", "HIPK2", "PTEN"],
            "HCD IV - DNA binding":   ["MDM2", "BRCA1", "53BP1"],
            "HCD V - DNA binding":    ["BRCA1", "BCL-XL", "BCL2"],
            "Linker":                 ["YWHAZ", "YWHAE", "YWHAB"],   # 14-3-3 family
            "NLS":                    ["KPNA2", "KPNA1"],
            "Oligomerization":        ["MDM2", "RASSF1", "PIAS1"],
            "NES - Oligomerization":  ["XPO1"],
            "Regulatory":             ["YWHAZ", "EP300", "SIRT1", "HDAC1"],
        }

    result = {}
    for domain, partners in domain_interface_map.items():
        partners_upper = [p.upper() for p in partners]
        in_graph = [p for p in partners_upper if G.has_node(p)]
        scores   = [G[tp53_node][p]["weight"] if G.has_edge(tp53_node, p) else 0.3
                    for p in in_graph]
        drivers  = [p for p in in_graph if p in TP53_CANCER_DRIVER_INTERACTORS]

        result[domain] = {
            "n_domain_partners":    len(partners),
            "n_in_graph":           len(in_graph),
            "max_score":            max(scores) if scores else 0.0,
            "mean_score":           float(np.mean(scores)) if scores else 0.0,
            "has_driver":           int(len(drivers) > 0),
            "n_drivers":            len(drivers),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point: build or load cached graph
# ─────────────────────────────────────────────────────────────────────────────

def get_ppi_graph(force_rebuild: bool = False) -> tuple[nx.Graph, dict, dict]:
    """
    Load or build the TP53 PPI graph.
    Returns: (graph, global_ppi_features, domain_ppi_features)
    """
    if not force_rebuild and os.path.exists(PPI_GRAPH_CACHE):
        logger.info(f"Loading cached PPI graph from {PPI_GRAPH_CACHE}")
        with open(PPI_GRAPH_CACHE, "rb") as f:
            return pickle.load(f)

    # ── Load STRING ──────────────────────────────────────────────────────────
    string_df = None
    if os.path.exists(STRING_LINKS) and os.path.exists(STRING_PROTEIN_INFO):
        string_df = load_string_interactions(STRING_LINKS, STRING_PROTEIN_INFO)
    else:
        logger.warning(f"STRING files not found at {STRING_LINKS} / {STRING_PROTEIN_INFO}")

    # ── Load BioGRID ─────────────────────────────────────────────────────────
    biogrid_df = None
    if os.path.exists(BIOGRID_FILE):
        biogrid_df = load_biogrid_interactions(BIOGRID_FILE)
    else:
        logger.warning(f"BioGRID file not found at {BIOGRID_FILE}")

    # ── Fallback: hardcoded TP53 interactome if files unavailable ────────────
    if string_df is None and biogrid_df is None:
        logger.info("Using built-in curated TP53 interactome as fallback …")
        string_df = _builtin_tp53_interactome()

    # ── Build graph ──────────────────────────────────────────────────────────
    G = build_tp53_ppi_graph(string_df, biogrid_df)

    # Make sure TP53 is always present
    if "TP53" not in G:
        G.add_node("TP53")

    g_feats  = compute_ppi_features(G)
    d_feats  = domain_ppi_features(G)

    result = (G, g_feats, d_feats)
    with open(PPI_GRAPH_CACHE, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"PPI graph cached -> {PPI_GRAPH_CACHE}")

    return result


def _builtin_tp53_interactome() -> pd.DataFrame:
    """
    Curated TP53 interactors with approximate STRING-equivalent scores.
    Used as fallback when PPI files are unavailable.
    """
    interactions = [
        ("TP53", "MDM2",   0.999), ("TP53", "MDM4",   0.970), ("TP53", "BRCA1",  0.940),
        ("TP53", "ATM",    0.960), ("TP53", "CHEK2",  0.930), ("TP53", "HIPK2",  0.880),
        ("TP53", "EP300",  0.950), ("TP53", "CREBBP", 0.920), ("TP53", "SIRT1",  0.870),
        ("TP53", "HDAC1",  0.810), ("TP53", "YWHAZ",  0.850), ("TP53", "PTEN",   0.780),
        ("TP53", "BCL2",   0.760), ("TP53", "ASPP1",  0.840), ("TP53", "ASPP2",  0.845),
        ("TP53", "PARP1",  0.820), ("TP53", "HMGB1",  0.790), ("TP53", "XPO1",   0.750),
        ("TP53", "KPNA2",  0.780), ("TP53", "RELA",   0.730), ("TP53", "CDK4",   0.710),
        ("TP53", "CDKN1A", 0.999), ("TP53", "BBC3",   0.990), ("TP53", "BAX",    0.990),
        ("TP53", "PUMA",   0.990), ("TP53", "NOXA",   0.980), ("TP53", "GADD45A",0.980),
        ("TP53", "RB1",    0.720), ("TP53", "AKT1",   0.700), ("TP53", "PIK3CA", 0.680),
        ("TP53", "VHL",    0.760), ("TP53", "NF1",    0.650), ("TP53", "RASSF1", 0.820),
        ("TP53", "PIAS1",  0.760), ("TP53", "COP1",   0.800), ("TP53", "PIRH2",  0.780),
        ("TP53", "TIGAR",  0.960), ("TP53", "CHEK1",  0.890), ("TP53", "PLK3",   0.820),
        ("TP53", "VRK1",   0.850), ("TP53", "PPM1D",  0.870), ("TP53", "USP7",   0.890),
        ("TP53", "HAUSP",  0.880), ("TP53", "TWIST1", 0.640), ("TP53", "MYC",    0.730),
    ]
    rows = [{"gene_a": a, "gene_b": b, "combined_score": w} for a, b, w in interactions]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick diagnostics
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G, g_feats, d_feats = get_ppi_graph(force_rebuild=True)
    print("\n=== Global PPI features ===")
    for k, v in g_feats.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k}: {v}")

    print("\n=== Domain PPI features ===")
    for domain, vals in d_feats.items():
        print(f"  {domain}: {vals}")
