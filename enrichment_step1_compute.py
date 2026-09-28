"""
STANDALONE reproduction pipeline for enrichment analysis (1/2) — compute the per-gene table that backs
both figures (fig_enrichment_and_length and venn_feedback_thresholds).

Self-contained: depends only on the raw databases (cached under ../data, or
downloaded on first run) and standard scientific Python + goatools. It does NOT
import any of the exploratory project scripts.

Produces exactly the quantities the two figures need, for every network gene:
  - signed <=3-node feedback loop participation, split by scale (1/2/3) and by
    polarity (positive / negative);
  - a degree-preserving null-model z-score (z, z_pos, z_neg) = "more signed
    feedback than the gene's degree predicts";
  - membership in four functional gene-sets (curated GO + MSigDB stemness):
        stem_progenitor, cellular_development, differentiation, homeostasis
  - in/out degree.

Output: gene_table.csv     (input to step2_figures.py)

Env: N_NULL (default 200), SEED (default 0).
"""
from __future__ import annotations
import io, os, json, time, random, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
DATA.mkdir(exist_ok=True)
N_NULL = int(os.environ.get("N_NULL", "200"))
SEED = int(os.environ.get("SEED", "0"))
SWAPS_PER_EDGE = 10

# ------------------------------------------------------------------ downloads #
URLS = {
    "trrust_human": "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv",
    "trrust_mouse": "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv",
    "mgi_homology": "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt",
    "signor_human": "https://signor.uniroma2.it/getData.php?organism=9606",
    "signor_mouse": "https://signor.uniroma2.it/getData.php?organism=10090",
    "regnetwork_zip": "https://regnetworkweb.org/download/RegulatoryDirections.zip",
}
FNAMES = {"trrust_human": "trrust_rawdata.human.tsv", "trrust_mouse": "trrust_rawdata.mouse.tsv",
          "mgi_homology": "HOM_MouseHumanSequence.rpt", "signor_human": "signor_human.tsv",
          "signor_mouse": "signor_mouse.tsv", "regnetwork_zip": "RegulatoryDirections.zip"}


def fetch_path(key: str) -> Path:
    p = DATA / FNAMES[key]
    if p.exists() and p.stat().st_size > 0:
        return p
    print(f"  downloading {key} ...")
    r = requests.get(URLS[key], timeout=180); r.raise_for_status()
    p.write_bytes(r.content)
    return p


def fetch(key: str) -> str:
    return fetch_path(key).read_text(encoding="utf-8")


# ------------------------------------------------------------- network build #
SIGN_NUM = {"Activation": 1, "Repression": -1, "Unknown": 0, "Conflict": 0}


def load_trrust(sp):
    df = pd.read_csv(io.StringIO(fetch(f"trrust_{sp}")), sep="\t", header=None,
                     names=["regulator", "target", "mode", "pmids"])
    df["sign"] = df["mode"].where(df["mode"].isin(["Activation", "Repression"]), "Unknown")
    df["species"] = sp; df["source"] = "TRRUST"
    return df[["regulator", "target", "sign", "species", "source"]]


def _sig(effect):
    e = effect.strip().lower()
    return "Activation" if e.startswith("up-regulates") else \
           "Repression" if e.startswith("down-regulates") else "Unknown"


def load_signor(sp):
    recs = []
    for line in fetch(f"signor_{sp}").splitlines():
        f = line.split("\t")
        if len(f) < 9 or f[1] != "protein" or f[5] != "protein":
            continue
        if f[8].strip().lower() == "form complex":
            continue
        recs.append((f[0], f[4], _sig(f[8])))
    df = pd.DataFrame(recs, columns=["regulator", "target", "sign"])
    df["species"] = sp; df["source"] = "SIGNOR"
    return df


_RN = {"-->": "Activation", "--|": "Repression"}


def load_regnetwork(sp):
    with zipfile.ZipFile(fetch_path("regnetwork_zip")) as z:
        data = z.read(f"new_kegg.{sp}.reg.direction.txt").decode("utf-8", "replace")
    recs = []
    for line in data.splitlines():
        tok = line.split()
        if line.strip().startswith("#") or len(tok) < 5:
            continue
        recs.append((tok[0], tok[2], _RN.get(tok[-1], "Unknown")))
    df = pd.DataFrame(recs, columns=["regulator", "target", "sign"])
    df["species"] = sp; df["source"] = "RegNetwork"
    return df


def mouse_to_human():
    hom = pd.read_csv(io.StringIO(fetch("mgi_homology")), sep="\t")[
        ["DB Class Key", "Common Organism Name", "Symbol"]]
    hom["Symbol"] = hom["Symbol"].astype(str)
    g = hom.groupby("DB Class Key")
    ms = g.apply(lambda d: sorted(set(d.loc[d["Common Organism Name"].str.startswith("mouse"), "Symbol"])))
    hs = g.apply(lambda d: sorted(set(d.loc[d["Common Organism Name"] == "human", "Symbol"])))
    m = {}
    for k in ms.index:
        if len(ms[k]) == 1 and len(hs[k]) == 1:
            m[ms[k][0]] = hs[k][0]
    return m


def build_network():
    frames = [load_trrust("human"), load_trrust("mouse"), load_signor("human"),
              load_signor("mouse"), load_regnetwork("human"), load_regnetwork("mouse")]
    raw = pd.concat(frames, ignore_index=True)
    m2h = mouse_to_human()

    def mp(sym, sp):
        s = str(sym)
        if sp == "human":
            return s.upper()
        return m2h.get(s) or m2h.get(s.capitalize()) or m2h.get(s.upper())
    raw["r"] = [mp(s, sp) for s, sp in zip(raw["regulator"], raw["species"])]
    raw["t"] = [mp(s, sp) for s, sp in zip(raw["target"], raw["species"])]
    junk = {"", "NA", "N/A", "NAN", "NONE", "NULL", "-", "."}
    ok = raw["r"].notna() & raw["t"].notna()
    ok &= ~raw["r"].fillna("").str.upper().isin(junk)
    ok &= ~raw["t"].fillna("").str.upper().isin(junk)
    raw = raw[ok]

    def consensus(grp):
        known = set(grp.loc[grp["sign"].isin(["Activation", "Repression"]), "sign"])
        sign = "Unknown" if not known else (known.pop() if len(known) == 1 else "Conflict")
        return SIGN_NUM[sign]
    pooled = raw.groupby(["r", "t"]).apply(consensus).reset_index(name="sign_num")
    pooled.columns = ["regulator", "target", "sign_num"]
    print(f"  network: {len(pooled)} edges, "
          f"{len(set(pooled['regulator']) | set(pooled['target']))} nodes")
    return pooled


# ---------------------------------------------- signed <=3-node loop counting #
def build_succ_sign(edges):
    succ = defaultdict(set); sign = {}
    for u, v, s in edges:
        succ[u].add(v); sign[(u, v)] = s
    return succ, sign


def count_signed(succ, sign):
    """Per-gene participation in signed 2- & 3-node cycles.
    Returns pos, neg (combined-scale, for the headline z) and c2, c3 (per-scale
    2-node / 3-node counts, for the per-scale length-gradient z)."""
    pos, neg = Counter(), Counter()
    c2, c3 = Counter(), Counter()
    for u, outs in succ.items():
        for v in outs:
            if u < v and u in succ.get(v, ()):
                s1, s2 = sign[(u, v)], sign[(v, u)]
                if s1 and s2:
                    (pos if s1 * s2 > 0 else neg)[u] += 1
                    (pos if s1 * s2 > 0 else neg)[v] += 1
                    c2[u] += 1; c2[v] += 1
    for u, outs in succ.items():
        for v in outs:
            sv = sign[(u, v)]
            if not sv:
                continue
            for w in succ.get(v, ()):
                sw = sign[(v, w)]
                if not sw or u >= w or u >= v:
                    continue
                if u in succ.get(w, ()):
                    su = sign[(w, u)]
                    if not su:
                        continue
                    b = pos if sv * sw * su > 0 else neg
                    b[u] += 1; b[v] += 1; b[w] += 1
                    c3[u] += 1; c3[v] += 1; c3[w] += 1
    return pos, neg, c2, c3


def randomize(nonself, rng):
    by_sign = defaultdict(list); E = set()
    for u, v, s in nonself:
        by_sign[s].append([u, v]); E.add((u, v))
    for s, lst in by_sign.items():
        m = len(lst)
        if m < 2:
            continue
        target = SWAPS_PER_EDGE * m; done = tries = 0; maxt = target * 20
        while done < target and tries < maxt:
            tries += 1
            i, j = rng.randrange(m), rng.randrange(m)
            if i == j:
                continue
            a, b = lst[i]; c, d = lst[j]
            if len({a, b, c, d}) < 4 or (a, d) in E or (c, b) in E:
                continue
            E.discard((a, b)); E.discard((c, d)); E.add((a, d)); E.add((c, b))
            lst[i] = [a, d]; lst[j] = [c, b]; done += 1
    return [(u, v, s) for s, lst in by_sign.items() for u, v in lst]


# --------------------------------------------------------------- GO gene-sets #
FUNCTION_ANCHORS = {
    "stem_progenitor": ["GO:0019827", "GO:0035019", "GO:0017145", "GO:0072089"],
    "cellular_development": ["GO:0048869"],
    "differentiation": ["GO:0030154", "GO:0045595"],
    "homeostasis": ["GO:0042592", "GO:0001894"],
}
EXPERIMENTAL = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}
CURATED = EXPERIMENTAL | {"TAS", "IC"}
MSIGDB_STEMNESS = ["WONG_EMBRYONIC_STEM_CELL_CORE", "BENPORATH_ES_1",
                   "BHATTACHARYA_EMBRYONIC_STEM_CELL", "RAMALHO_STEMNESS_UP",
                   "MUELLER_PLURINET", "BENPORATH_NANOG_TARGETS",
                   "BENPORATH_OCT4_TARGETS", "BENPORATH_SOX2_TARGETS"]


def get_dag():
    from goatools.obo_parser import GODag
    from goatools.base import download_go_basic_obo
    obo = download_go_basic_obo(str(DATA / "go-basic.obo"))
    return GODag(obo, optional_attrs={"relationship"}, prt=None)


def term_descendants(dag, gid):
    seen, stack = set(), [gid]
    while stack:
        t = stack.pop()
        if t in seen or t not in dag:
            continue
        seen.add(t); node = dag[t]
        stack.extend(c.id for c in node.children)
        stack.extend(x.id for x in getattr(node, "relationship_rev", {}).get("part_of", set()))
    return seen


def get_gene_go(genes):
    cache = DATA / "gene_go_bp.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import mygene
    res = mygene.MyGeneInfo().querymany(genes, scopes="symbol", fields="go.BP",
                                        species="human", returnall=True, verbose=False)
    out = {}
    for r in res["out"]:
        q = r.get("query")
        bp = r.get("go", {}).get("BP", [])
        if isinstance(bp, dict):
            bp = [bp]
        out.setdefault(q, []).extend([[t.get("id"), t.get("evidence")] for t in bp if t.get("id")])
    cache.write_text(json.dumps(out))
    return out


def msigdb_stemness():
    d = DATA / "msigdb"; d.mkdir(exist_ok=True)
    genes = set()
    for name in MSIGDB_STEMNESS:
        p = d / f"{name}.txt"
        if not p.exists() or p.stat().st_size == 0:
            url = ("https://www.gsea-msigdb.org/gsea/msigdb/human/download_geneset.jsp"
                   f"?geneSetName={name}&fileType=txt")
            r = requests.get(url, timeout=90); r.raise_for_status(); p.write_text(r.text)
        genes |= {l.strip().upper() for l in p.read_text().splitlines()
                  if l.strip() and not l.startswith(">") and l.strip() != name}
    return genes


# ----------------------------------------------------------------------- main #
def main():
    print(f"[reproduce step1] N_NULL={N_NULL}, SEED={SEED}")
    print("[1/5] build pooled signed GRN ...")
    net = build_network()
    edges = [(r.regulator, r.target, int(r.sign_num)) for r in net.itertuples(index=False)
             if r.regulator != "" and r.target != ""]
    nonself = [(u, v, s) for u, v, s in edges if u != v]
    selfsign = {u: s for u, v, s in edges if u == v}

    print("[2/5] observed signed <=3-node feedback participation ...")
    succ0, sign0 = build_succ_sign(nonself)
    pos0, neg0, c2_0, c3_0 = count_signed(succ0, sign0)
    genes = sorted(set(succ0) | {v for vs in succ0.values() for v in vs} | set(selfsign))
    indeg = Counter(); outdeg = Counter()
    for u, v, s in edges:
        outdeg[u] += 1; indeg[v] += 1
    sig1 = {g for g, s in selfsign.items() if s != 0}          # signed self-loop

    print(f"[3/5] degree-preserving null model ({N_NULL} reps) ...")
    obs = {g: pos0.get(g, 0) + neg0.get(g, 0) for g in genes}
    obsp = {g: pos0.get(g, 0) for g in genes}; obsn = {g: neg0.get(g, 0) for g in genes}
    obs2 = {g: c2_0.get(g, 0) for g in genes}; obs3 = {g: c3_0.get(g, 0) for g in genes}
    rng = random.Random(SEED)
    acc = {k: pd.Series(0.0, index=genes) for k in
           ("s", "q", "sp", "qp", "sn", "qn", "s2", "q2", "s3", "q3")}
    t0 = time.time()
    for k in range(N_NULL):
        p, n, c2, c3 = count_signed(*build_succ_sign(randomize(nonself, rng)))
        vp = pd.Series({g: p.get(g, 0) for g in genes}).reindex(genes).fillna(0)
        vn = pd.Series({g: n.get(g, 0) for g in genes}).reindex(genes).fillna(0)
        v2 = pd.Series({g: c2.get(g, 0) for g in genes}).reindex(genes).fillna(0)
        v3 = pd.Series({g: c3.get(g, 0) for g in genes}).reindex(genes).fillna(0)
        vs = vp + vn
        acc["s"] += vs; acc["q"] += vs * vs
        acc["sp"] += vp; acc["qp"] += vp * vp
        acc["sn"] += vn; acc["qn"] += vn * vn
        acc["s2"] += v2; acc["q2"] += v2 * v2
        acc["s3"] += v3; acc["q3"] += v3 * v3
        if k == 0:
            print(f"      ~{time.time()-t0:.1f}s/rep (est {N_NULL*(time.time()-t0):.0f}s)")

    def z_of(obsd, s, q):
        mean = acc[s] / N_NULL
        std = np.sqrt(((acc[q] / N_NULL) - mean ** 2).clip(lower=0))
        ob = pd.Series(obsd).reindex(genes).fillna(0)
        return np.where(std > 0, (ob - mean) / std, 0.0)

    df = pd.DataFrame({"gene": genes})
    df["z"] = z_of(obs, "s", "q")
    df["z_pos"] = z_of(obsp, "sp", "qp")
    df["z_neg"] = z_of(obsn, "sn", "qn")
    df["z2"] = z_of(obs2, "s2", "q2")          # per-scale (2-node) null z
    df["z3"] = z_of(obs3, "s3", "q3")          # per-scale (3-node) null z
    df["out_degree"] = df["gene"].map(lambda g: outdeg.get(g, 0))
    df["degree"] = df["gene"].map(lambda g: indeg.get(g, 0) + outdeg.get(g, 0))
    df["sig_scale1"] = df["gene"].isin(sig1)
    df["sig_scale2"] = df["gene"].map(lambda g: c2_0.get(g, 0) > 0)
    df["sig_scale3"] = df["gene"].map(lambda g: c3_0.get(g, 0) > 0)
    print(f"      genes z>=2: {int((df['z']>=2).sum())}  (total {time.time()-t0:.0f}s)")

    print("[4/5] functional gene-sets (curated GO + MSigDB stemness) ...")
    dag = get_dag()
    termsets = {name: set().union(*[term_descendants(dag, a) for a in anchors])
                for name, anchors in FUNCTION_ANCHORS.items()}
    gene_go = get_gene_go(genes)
    gt = {g: {t for t, ev in pairs if ev in CURATED} for g, pairs in gene_go.items()}
    df["n_go_bp"] = df["gene"].map(lambda g: len(gt.get(g, set())))
    msig = msigdb_stemness()
    for name, tset in termsets.items():
        df[name] = df["gene"].map(lambda g: len(gt.get(g, set()) & tset) > 0)
    df["stem_progenitor"] = df["stem_progenitor"] | df["gene"].isin(msig)

    print("[5/5] write gene_table.csv")
    df.to_csv(ROOT / "gene_table.csv", index=False)
    bg = df[(df["out_degree"] > 0) & (df["n_go_bp"] > 0)]
    print(f"      background (annotated regulators) = {len(bg)}")
    for c in ("stem_progenitor", "cellular_development", "differentiation", "homeostasis"):
        print(f"        {c:22s} in background: {int(bg[c].sum())}")


if __name__ == "__main__":
    main()
