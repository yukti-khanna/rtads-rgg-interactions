"""Shared utilities; paths are anchored to this fig6 folder, never the caller's cwd."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "updated_validation"
DIAG = ROOT / "prediction_diagnostics"


def inside(path):
    p = Path(path)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Output must be inside fig6: {p}")
    return p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seqhash(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def write_json(path, value):
    p = inside(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, default=str) + "\n")


def read_fasta(path):
    """Retain duplicate records; never silently replace a repeated FASTA identity."""
    records, name, seq = [], None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(seq).upper()))
            name, seq = line[1:].split()[0], []
        elif line.strip():
            if name is None:
                raise ValueError(f"Sequence before FASTA header: {path}")
            seq.append("".join(line.split()))
    if name is not None:
        records.append((name, "".join(seq).upper()))
    d = pd.DataFrame(records, columns=["name", "sequence"])
    d["length"] = d.sequence.str.len()
    d["sha256"] = d.sequence.map(seqhash)
    d["duplicate_name"] = d.name.duplicated(keep=False)
    d["conflicting_name"] = d.name.map(d.groupby("name").sha256.nunique()).gt(1)
    d["duplicate_sequence"] = d.sequence.duplicated(keep=False)
    d["invalid_residue"] = ~d.sequence.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")
    return d


def unique(df, key, label):
    if df[key].isna().any() or df[key].duplicated().any():
        raise ValueError(f"Missing or duplicate {key} in {label}; see mapping QC")


def pkd(kd):
    k = np.asarray(kd, float)
    result = np.full(k.shape, np.nan)
    m = np.isfinite(k) & (k > 0)
    result[m] = 6 - np.log10(k[m])
    return result


def log_negative(values):
    a = np.asarray(values, float)
    result = np.full(a.shape, np.nan)
    m = np.isfinite(a) & (a < 0)
    result[m] = np.log10(-a[m])
    return result


def rdf_inventory(directory, aliases_file=None):
    """Exact oriented pair IDs only. Alias application requires explicit approval."""
    approved = {}
    if aliases_file and Path(aliases_file).exists():
        aliases = pd.read_csv(aliases_file).fillna("")
        a = aliases[aliases.approved.astype(str).str.lower().eq("true")]
        unique(a, "rdf_pair_id", "approved RDF aliases")
        if (a.evidence.astype(str).str.len() == 0).any():
            raise ValueError("An approved RDF alias needs sequence/construct evidence")
        approved = dict(zip(a.rdf_pair_id, a.prediction_pair_id))
    rows = []
    for f in sorted(Path(directory).glob("*.pkl")):
        import re
        m = re.fullmatch(r"rdfs_(.+)_(\d+(?:\.\d+)?)nm", f.stem)
        if not m or len(m.group(1).split("__")) != 2:
            raise ValueError(f"Unrecognized RDF filename: {f}")
        original = m.group(1)
        rows.append(dict(rdf_pair_id=original, pair_id=approved.get(original, original),
                         box_size_nm=float(m.group(2)), rdf_file=str(f.relative_to(ROOT)),
                         rdf_alias_applied=original in approved))
    return pd.DataFrame(rows, columns=["rdf_pair_id", "pair_id", "box_size_nm", "rdf_file", "rdf_alias_applied"])
