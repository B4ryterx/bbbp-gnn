# Blood-brain barrier penetration from molecular graphs

Does a bond-aware graph neural network predict blood-brain barrier penetration
better than a 2015-vintage fingerprint and a linear model? And how much of any
answer survives being asked about chemistry the model has never seen?

Dataset: **BBBP** from MoleculeNet (Wu et al., 2018; labels from Martins et al.,
2012), loaded through PyTorch Geometric.

## Headline

1,965 molecules after cleaning, 76.3% positive. GNN figures are mean +/- sd over
three seeds; the baseline is deterministic. Each test set was scored once.

| model | split | ROC-AUC | PR-AUC | accuracy | balanced acc | sensitivity | **specificity** |
|---|---|---|---|---|---|---|---|
| ECFP + LR | random | 0.913 | 0.969 | 0.854 | 0.824 | 0.880 | 0.768 |
| GNN (GINE) | random | 0.929 +/- 0.011 | 0.978 +/- 0.003 | 0.876 +/- 0.023 | 0.792 +/- 0.031 | 0.951 +/- 0.016 | 0.633 +/- 0.047 |
| ECFP + LR | scaffold | 0.760 | 0.946 | 0.874 | 0.607 | 0.996 | **0.217** |
| GNN (GINE) | scaffold | 0.791 +/- 0.031 | 0.957 +/- 0.007 | 0.833 +/- 0.012 | 0.597 +/- 0.040 | 0.941 +/- 0.024 | **0.254 +/- 0.098** |

Three findings matter more than the leaderboard position.

**1. The split moves the result several times more than the model does.**
Switching from random to scaffold costs the GNN 0.138 ROC-AUC and the baseline
0.153. The gap between the two models is 0.016 (random) and 0.031 (scaffold).
On the scaffold split that gap is exactly one seed standard deviation, so **the
GNN's advantage over a 2015 fingerprint is not distinguishable from noise on
the split that matters.** 250,244 parameters and bond-aware message passing buy
nothing reliable here. Reporting a single random-split number on BBBP reports
mostly the split.

**2. Accuracy and F1 hide the failure; they move the wrong way.** Going to the
harder split, the baseline's accuracy *rises* 0.854 -> 0.874 and its F1 *rises*
0.902 -> 0.930, while its specificity falls 0.768 -> 0.217. The models respond
to unfamiliar chemistry by predicting the 76% majority class: sensitivity near
1.0, specificity near zero. That is the base rate wearing a model's clothes.
Balanced accuracy and specificity are the only columns that show it.

**3. Error rate tracks distance from the training set, and significantly.** On
the scaffold test set, molecules in the *least* similar third of Tanimoto
distance to training are predicted correctly 72.5% of the time; the *most*
similar third, 90.8%. Correct predictions have a higher mean similarity than
wrong ones (0.398 vs 0.340, Mann-Whitney p = 0.0008). The model is not
uniformly 84% accurate — it is reliable on familiar chemistry and close to
coin-flipping on the rest, and nothing in its output says which case you are
in.

## Why the scaffold split is the one to believe

A random split puts close analogues of the same drug on both sides of the
train/test line: the same parent compound as a salt, an ester, a homologue. A
model can score well on that by recognising the family rather than learning
anything transferable.

A Bemis-Murcko scaffold split forces every molecule sharing a ring framework
into the same subset, so test scaffolds are genuinely unseen. That is the
question a drug programme actually asks: *will this work on a new series?*

Acyclic molecules have an empty Murcko scaffold. They are kept as a **single
group** rather than one group each, which is the conservative choice: it
guarantees no acyclic molecule in test has an acyclic twin in train.

## Deliverables

| file | what it is |
|---|---|
| `BBBP_molecular_property_prediction.ipynb` | the whole study, self-contained and executed: every function is defined in a cell, nothing is imported from `bbbp/`, all outputs and figures visible |
| `report/ENG2440_A3_BBBP_report.pdf` | 7-page report — 6 pages of body, then references and two appendices (reproducibility, per-seed results) |
| `report/report.html` | the report's source; regenerate the PDF with headless Chrome |
| `results/` | `results.json`, `comparison.csv`, `error_rows_scaffold.csv`, 11 figures |

## Layout

```
bbbp/
  config.py      every number that affects a result
  data.py        load, canonicalize, de-duplicate, remove label conflicts
  splits.py      stratified random + Bemis-Murcko scaffold splits, with checks
  features.py    Morgan/ECFP fingerprints, Tanimoto similarity
  models.py      GINEConv network with categorical embeddings for atoms/bonds
  train.py       training loop, early stopping, frozen threshold
  baseline.py    ECFP + logistic regression
  metrics.py     metrics and validation-only threshold selection
  errors.py      error analysis by similarity to the training set
  plots.py       every figure
scripts/run_all.py   the whole pipeline, end to end
tests/test_pipeline.py   35 fast checks, no dataset needed
```

## Run

```bash
python -m venv --system-site-packages .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python tests/test_pipeline.py     # ~20 s, no download
./.venv/bin/python scripts/run_all.py --quick # ~2 min smoke test
./.venv/bin/python scripts/run_all.py         # full run, 3 seeds per split
```

The first run downloads BBBP (~1 MB) into `data/`, which is gitignored.
Everything else is written to `results/` and `results/figures/`.

## Method, and the decisions behind it

**Cleaning, before any split is made.** Drop unparseable SMILES and empty
graphs; canonicalize with RDKit; collapse duplicate structures to one record;
**delete entire groups whose labels disagree**. That last rule costs molecules
but is the only defensible option: keeping either label guarantees a known-wrong
example in training or test. De-duplicating on the raw SMILES string is not
enough, because two different strings routinely denote the same molecule — BBBP
contains aspirin twice, under two spellings, **with opposite labels**.

Ten molecules carry contradictory labels, and the source CSV's own `name`
column shows how: *BRL53080* / *loperamide*, *acetylsalicylate* / *aspirin*,
*atropine* / *atropine(hyoscyamine)*, *Trimetrexate* / *trimetrexate*,
*indomethacin* / *indomethacin(indometacin)*, plus levodopa, mequitazine,
methylprednisolone, miconazole and loratadine. Case variants, internal codes
and synonym parentheticals: a compilation merged from sources that were never
reconciled. **All ten pairs are written as different SMILES strings**, so none
is detectable without canonicalizing first.

**Feature encoding.** Every column of `x` and `edge_attr` is a category code.
Atomic number 6 is not "twice" atomic number 3. Each of the 9 atom columns and
3 bond columns gets its own trainable embedding table, summed — the OGB
convention. Treating the codes as continuous numbers would impose an ordering
that does not exist.

**Model.** GINEConv, the edge-aware Graph Isomorphism Network. Each node update
is `MLP((1+eps)·h_v + Σ ReLU(h_u + e_uv))`, so the bond embedding enters *inside*
the sum: bond information shapes every message rather than being concatenated at
the end. Three layers, residual connections, batch norm, mean pooling over
atoms, then a two-layer head. A separate bond encoder per layer, because what
matters about a bond differs between the first hop and the third.

**Honest evaluation.**
- The threshold is chosen on validation and frozen. Picking the threshold that
  flatters the test set is fitting to the test set by hand.
- Model selection is on validation ROC-AUC, not validation loss: loss is
  dominated by the majority class and keeps improving after ranking quality has
  peaked.
- `pos_weight` is computed from the training subset only.
- Each test set is scored exactly once, after everything is frozen.
- Three seeds per split, reported as mean +/- sd, so seed noise can be told
  apart from the effect of changing the split.

## Error analysis

On the scaffold test set, every error is matched to its most similar training
molecule by Tanimoto similarity over Morgan fingerprints. This separates two
very different failure modes:

- **Errors far from training** are an applicability-domain problem. The right
  response is to refuse to predict, not to train harder.
- **Errors with a close, correctly-labelled neighbour** are a modelling problem,
  an activity cliff, or a label error.

`results/error_rows_scaffold.csv` has one row per test molecule with its nearest
training neighbour, that neighbour's label, and whether it agrees.

## Known limitations

1. **The labels are not measurements.** BBBP is a curated literature
   compilation. The ten conflicting-label groups this pipeline removes are
   direct evidence of that: the same molecule appears with both labels. Nothing
   here establishes a ceiling on achievable accuracy, but the ceiling is not
   1.0.
2. **The dataset is small and narrow.** About 2,000 compounds after cleaning,
   76% of one class, heavily weighted toward marketed drugs. A 15% test set is
   roughly 290 molecules, so a difference of two points in any metric is well
   inside the noise.
3. **No external validation.** Every number comes from one dataset split three
   ways. There is no held-out experimental measurement anywhere in this
   pipeline.
4. **Correlation only.** A fingerprint bit firing with a label says nothing
   about mechanism. Nothing here identifies a cause of penetration, and nothing
   here is evidence that a compound is safe, effective, or worth testing.
5. **Scaffold split is one definition of novelty.** Bemis-Murcko frameworks are
   a crude proxy for "new chemistry". Two molecules with different frameworks
   can be near-identical in the properties that govern penetration.

## References

- Wu et al., *MoleculeNet: a benchmark for molecular machine learning*, Chem.
  Sci. 9:513-530, 2018.
- Martins et al., *A Bayesian approach to in silico blood-brain barrier
  penetration modeling*, J. Chem. Inf. Model. 52:1686-1697, 2012.
- Bemis and Murcko, *The properties of known drugs. 1. Molecular frameworks*,
  J. Med. Chem. 39:2887-2893, 1996.
- Hu et al., *Strategies for pre-training graph neural networks*, ICLR 2020
  (GINEConv, and the atom/bond encoder convention).
- Fey and Lenssen, *Fast graph representation learning with PyTorch Geometric*,
  ICLR workshop, 2019.
- Landrum et al., RDKit. Pinned to 2025.03.6: in 2026.09.2
  `Chem.MurckoDecompose` raises "bad any cast" and scaffold extraction fails.
