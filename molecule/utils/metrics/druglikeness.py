
from rdkit import Chem
from rdkit.Chem import QED, Crippen

# ---------- QED ----------
def calc_qed(mol: Chem.Mol) -> float:
    """Bickerton et al. QED score (0~1; higher is better)."""
    return float(QED.qed(mol))


# ---------- SA (Synthetic Accessibility) ----------
# Uses the standard Ertl-Schuffenhauer SA score implementation ("sascorer").
# If you don't already have it, drop the `sascorer.py` from RDKit's Contrib into your path.
#   https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py
try:
    import sascorer  # noqa: E402

    def calc_sa(mol: Chem.Mol) -> float:
        """Ertl-Schuffenhauer synthetic accessibility score (~1 easy … ~10 hard)."""
        return float(sascorer.calculateScore(mol))

except ImportError:
    def calc_sa(mol: Chem.Mol) -> float:
        raise ImportError(
            "sascorer not found. Get RDKit Contrib SA_Score/sascorer.py and put it on PYTHONPATH.\n"
            "URL: https://github.com/rdkit/rdkit/blob/master/Contrib/SA_Score/sascorer.py"
        )


# ---------- penalized logP ----------
def _largest_ring_penalty(mol: Chem.Mol) -> int:
    """Cycle penalty used in common benchmarks: max(largest_ring_size - 6, 0)."""
    rings = Chem.GetSymmSSSR(mol)
    if not rings:
        return 0
    largest = max(len(r) for r in rings)
    return max(largest - 6, 0)

def calc_plogp(mol: Chem.Mol) -> float:
    """
    Unnormalized penalized logP used in many optimization benchmarks:
        pLogP = logP(mol) - SA(mol) - cycle_penalty
    """
    logp = Crippen.MolLogP(mol)
    sa   = calc_sa(mol)
    cyc  = _largest_ring_penalty(mol)
    return float(logp - sa - cyc)

def calc_plogp_normalized(mol: Chem.Mol, stats: dict) -> float:
    """
    Normalized version (z-scored components), if you want dataset-comparable values.
    stats expects:
        stats = {
          'logP':  (mean_logP,  std_logP),
          'SA':    (mean_SA,    std_SA),
          'cycle': (mean_cycle, std_cycle)
        }
    Value = z(logP) - z(SA) - z(cycle)
    """
    logp = Crippen.MolLogP(mol)
    sa   = calc_sa(mol)
    cyc  = _largest_ring_penalty(mol)

    def z(x, mu_sigma): 
        mu, sig = mu_sigma
        return (x - mu) / (sig if sig > 1e-12 else 1.0)

    return float(z(logp, stats['logP']) - z(sa, stats['SA']) - z(cyc, stats['cycle']))
