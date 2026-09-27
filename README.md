# SILK-MARL: Screening and Injecting Multi-LLM Knowledge for Multi-Agent Reinforcement Learning

Research-code release for candidate generation, verification, short-run screening, training, and evaluation on three tasks:

| Task | Environment | Selected candidate |
| --- | --- | ---: |
| Spread3 | `pz-mpe-simple-spread-3` | 1 |
| Spread4 | `pz-mpe-simple-spread-4` | 2 |
| LBF2 | `lbforaging:Foraging-8x8-2p-2f-coop-v3` | 1 |

The release also includes the QMIX, MADDPG, and MAPPO baseline runner. It deliberately excludes papers, drafts, credentials, training logs, Sacred records, model checkpoints, and figures.

## Important reproducibility status

This is a curated copy of the local `baseline-260/YOLO-MARL` source, **not yet a complete archive of the reported Ours experiments**. Some functions generated and used on the training servers were not present in that local tree. See [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md) before attempting to reproduce the reported Ours runs. In particular, do not substitute newly generated functions and describe the resulting runs as exact reproductions: LLM output can differ.

## Setup

The original experiments used a Python 3.10 environment with CUDA on the training servers. From the repository root, install the dependencies in `requirements.txt` into a suitable isolated environment. The package list is a minimal dependency list, not a verified lockfile. For candidate generation, set `DASHSCOPE_API_KEY` in the environment; no key file is included or required in this release.

## Experiments

After restoring the selected generated modules listed in [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md), the 2M-step, 1k-evaluation-interval runs can be launched with:

```bash
python tools/run_spread3_2m_full_pipeline.py --skip-generation --skip-verifier --skip-screening --best-candidate 1 --formal-interval 1000
python tools/run_spread4_2m_full_pipeline.py --skip-generation --skip-verifier --skip-screening --best-candidate 2 --formal-interval 1000
python tools/run_lbf2_2m_full_pipeline.py --skip-generation --skip-verifier --skip-screening --best-candidate 1 --formal-interval 1000
```

Each formal pipeline runs four conditions across seeds 0, 1, and 2: baseline, Exp1+Exp3, Exp2+Exp3, and Exp1+Exp2+Exp3. The selected candidate IDs above refer to the reported experiments. For pure RL baselines:

```bash
python tools/run_pure_marl_baselines.py --tasks spread3,spread4,lbf2 --algorithms qmix,maddpg,mappo --seeds 0,1,2 --interval 1000
```

This runs 27 additional training jobs. Run one task at a time with `--tasks` if distributing them across machines. The Sacred and model outputs are written under `results/` and are ignored by Git. Check each script's `--help` for screening, evaluation, and dry-run options.

## Code map

- `src/`: QMIX/MADDPG/MAPPO training, Gymma/PettingZoo/LBF wrappers, task metrics, and LLM injection.
- `src/prompts/`: Qwen candidate-generation code, task descriptions, and available generated action/credit modules.
- `tools/run_*_2m_full_pipeline.py`: generation, screening, and formal training for the three tasks.
- `tools/run_pure_marl_baselines.py`: pure RL baselines.
- `tools/run_*1000ep_eval.py`: saved-model final evaluation.
- `tools/export_*` and `tools/plot_ablation_1000step.py`: tabulation and plotting from local experiment records.

## Attribution

This project is adapted from YOLO-MARL and PyMARL. The upstream Apache-2.0 license and NOTICE are retained. Generated candidate functions may have additional provenance from the respective model providers; review them before adding missing artifacts to a public release.
