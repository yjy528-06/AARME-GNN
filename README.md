# AARME-GNN

Code and datasets for *Adaptive attention routing multimodal expert graph neural network for
drug–drug interaction prediction*.

A single code base covers both benchmarks and the three prediction settings of the manuscript.
The benchmark is selected at run time with `--dataset 1` or `--dataset 2`.

## Contents

```
code/          AARME-GNN-task{1,2,3}.py   training and evaluation for the three settings
               modeltask{1,2,3}.py        model definition
dataset1/      the four knowledge-graph parts of Dataset 1 (572 drugs, 65 event types, 37,264 pairs)
dataset2/      the four knowledge-graph parts of Dataset 2 (1,258 drugs, 100 event types, 323,539 pairs)
requirements.txt
```

`dataset2/stats.json` records the statistics of the constructed knowledge graph of Dataset 2: the
number of drugs, entities, relation types and triples of each part, and the range of drug pairs per
event type.

If you use the Supporting Information archive as distributed, `code/`, `dataset1/` and `dataset2/`
sit one level lower, under `AARME-GNN_Code_and_Datasets/`.

## Environment

* Python 3.7 was used for the runs reported in the manuscript. The scripts also run with current
  PyTorch 2.x releases, and a CPU-only PyTorch build is sufficient.
* Install the dependencies with `pip install -r requirements.txt`.
* The device is selected once at start-up (`cuda` when available, otherwise `cpu`), so the scripts
  run unchanged on GPU and CPU machines.
* Deterministic behaviour is enabled (fixed seeds and `torch.use_deterministic_algorithms(True)`).
  On a GPU, export `CUBLAS_WORKSPACE_CONFIG=:4096:8` **before** starting Python, otherwise PyTorch
  raises an error for deterministic CUDA operations.

## Running

All commands are issued from inside the `code` directory. The defaults of the scripts are the
settings used in the manuscript, so no further options are needed.

```bash
cd code
export CUBLAS_WORKSPACE_CONFIG=:4096:8      # required on a GPU, ignored on a CPU

# Dataset 1
python -u AARME-GNN-task1.py --dataset 1
python -u AARME-GNN-task2.py --dataset 1
python -u AARME-GNN-task3.py --dataset 1

# Dataset 2
python -u AARME-GNN-task1.py --dataset 2
python -u AARME-GNN-task2.py --dataset 2
python -u AARME-GNN-task3.py --dataset 2
```

`--dataset` sets the drug count, the number of event types and the data directory:

| `--dataset` | drugs | event types | data directory | drug pairs |
|---|---|---|---|---|
| 1 | 572 | 65 | `../dataset1` | 37,264 |
| 2 | 1,258 | 100 | `../dataset2` | 323,539 |

A different data location can be given with `--data_dir`.

### Keeping the two benchmarks apart

Each run writes its output to `../result/`, and the file names do not contain the benchmark, so a
run on Dataset 2 overwrites the files written for Dataset 1. To reproduce both benchmarks, change
the two output paths in the script before running:

* `save_result("../result/", ...)`
* `np.save(os.path.join("../result", ...))`

to `../result1/` for Dataset 1 and `../result2/` for Dataset 2. Both directories are created by the
script if they do not exist.

## Hyperparameters (script defaults)

Task 1 uses four modality branches. Tasks 2 and 3 follow the released implementation of MKG-FENN
for those two settings and use three of the four branches, the molecular-structure part being used
in Task 1 only.

| Setting | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| Modality branches | 4 | 3 | 3 |
| Embedding dimension | 128 | 256 | 256 |
| Learning rate | 1e-2 | 5e-3 | 5e-3 |
| Training epochs | 120 | 100 | 120 |

Shared by the three tasks:

| Setting | Value |
|---|---|
| Batch size | 1,024 |
| Weight decay | 1e-8 |
| Neighbour sampling size | 6 |
| Dropout | 0.3 |
| Bottleneck dimension | 64 |
| Loss coefficients | λ_balance = 0.2, λ_dist = 0.1, λ_uni = 0.1 |
| Auxiliary-head input | raw modality representation `h_i^(m)` (Eq. (11) of the manuscript) |

## Outputs

Each run writes into `../result/`:

* `alltask<N>_AARME-GNN<N>.csv` — one row per metric, eleven in all; the six metrics reported in the
  manuscript are accuracy, micro-averaged AUPR, micro-averaged AUC, macro-averaged F1, macro-averaged
  precision and macro-averaged recall (Tables 1–3 for Dataset 1 and Table 5 for Dataset 2).
* `eachtask<N>_AARME-GNN<N>.csv` — the per-event values of the 65 (Dataset 1) and 100 (Dataset 2)
  event types: accuracy, AUPR, AUC, F1, precision and recall for each event. The event-level
  comparisons of Tables 4 and 6 are obtained from the same runs.
* `y_true`, `y_pred` and `y_score` arrays (`.npy`). The `y_score` matrices hold the per-sample class
  probabilities and allow AUPR and AUC to be recomputed at the level of individual event types. They
  are large (about 700 MB in total for Dataset 2) and are therefore not shipped in this archive;
  re-running the scripts regenerates them.

## Notes

* The auxiliary heads are fed by the **raw** modality representation, as specified by Eq. (11) of
  the manuscript; a sensitivity analysis of this choice is given in S3 Table.
* Dataset 1 is the four-part knowledge graph released with MKG-FENN. Dataset 2 is the public
  benchmark distributed with MDDI-SCL and MDF-SA-DDI, for which the four parts were constructed
  following the design of Dataset 1. The `dataset1/` and `dataset2/` folders contain the exact files
  used in the experiments.
* The MKG-FENN values in the Dataset 2 tables were obtained by re-running its released
  implementation on the identical five-fold split; that code is available from its authors.
