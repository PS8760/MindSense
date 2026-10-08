# Decision Log — MindSense

Every deviation from `prompt.md` and every judgement call is recorded here (Rule 6).

| # | Date | Area | Decision | Rationale |
|---|------|------|----------|-----------|
| 1 | 2026-10-08 | Repo layout | The repository root **is** the `mindsense/` project root (no nested folder). | Standard for a git clone; paths in `config.yaml` stay relative to root. |
| 2 | 2026-10-08 | Environment | Python 3.11 venv at `.venv`, dependencies managed with `uv` but pinned in plain `requirements*.txt` files. | `uv` is only a fast installer; the pinned requirement files remain the source of truth so graders can `pip install -r`. |
| 3 | 2026-10-08 | Dependencies | Runtime vs training split exactly as Section 4: no torch/transformers in `requirements.txt`. | Keeps the deployed image light (Section 0.4). |
| 4 | 2026-10-08 | Quality tools | `pytest` and `ruff` live in `requirements-train.txt`, not in the runtime file. | They are needed by `make test`/`make lint` and CI but never at inference. |
| 5 | 2026-10-08 | App export | Placeholders for dataset availability are filled only after `scripts/download_data.py` runs; synthetic fallback rows are tagged `data_source="synthetic"`. | Rule 2 / Section 5.6. |

## Open items / user actions

- **Kaggle credentials**: place `~/.kaggle/kaggle.json` (chmod 600) to fetch ranks 1, 2, 6, 9. Without it, those datasets fall back to manual placement or (last resort) labelled synthetic data.
- **Drugs.com terms**: dataset is research-use only, no redistribution → never committed to the repo; only aggregates published.
- **Helpline verification**: numbers in `config/helplines.yaml` were checked on 2026-10-08; re-verify before any public deployment.
