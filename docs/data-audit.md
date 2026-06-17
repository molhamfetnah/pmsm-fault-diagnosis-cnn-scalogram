# Dataset Audit (Task 0.3)

> Status: **TEMPLATE — to be completed by the team.** The dev environment had no
> network access, so the candidate sources below could not be opened/verified
> here. Fill the tables in by visiting each link, then record the two selections.

## Decision (fill in after auditing)

- **Chosen CURRENT dataset:** _TBD_
- **Chosen VIBRATION dataset:** _TBD_
- **Label mapping** (their raw labels → our `classes`):

| Dataset raw label | Our class (`Healthy`/`InterTurn`/`Demagnetization`/`Overload`) |
|---|---|
| _e.g. "normal"_ | Healthy |
| | |

- **Demagnetization availability:** if absent from real data, note it will be
  simulation-only (or dropped to a 3-class problem) — record the call here.

## Candidate sources (from references.md)

| name | url | signal_type | classes covered | fs | license | notes |
|---|---|---|---|---|---|---|
| Kaggle `ziya07/pmsm-smart-control-dataset` | https://www.kaggle.com/datasets/ziya07/pmsm-smart-control-dataset | ? | ? | ? | ? | control dataset — verify it has fault labels |
| Mendeley | https://data.mendeley.com/datasets/rgn5brrgrn/1 | ? | ? | ? | ? | |
| Zenodo | https://zenodo.org/records/13974503 | ? | ? | ? | ? | |
| IEEE-DataPort PMSM ITSC | https://ieee-dataport.org/documents/three-phase-pmsm-itsc-faults-stator-winding-dataset | current | Healthy + inter-turn (ITSC) | ? | ? | strong candidate for CURRENT |
| ResearchGate ResNet PMSM faults | https://www.researchgate.net/publication/379692489 | ? | electrical faults | ? | ? | check the dataset it cites |

## How to verify access (per source)

```bash
curl -sI "<dataset-url>" | head -5          # expect HTTP 200/302
```
For login-gated sources (Kaggle, IEEE-DataPort) record the manual download steps
and where the files land under `data/raw/<dataset_name>/`.
