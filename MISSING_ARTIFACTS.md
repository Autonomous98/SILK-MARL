# Generated artifacts absent from the local source snapshot

The local `baseline-260/YOLO-MARL` directory did not contain all functions loaded by the reported training-server runs. These files must be copied from the original server runs, with their contents verified against the corresponding experiment, to reproduce the exact Ours conditions:

| Task | Required selected files |
| --- | --- |
| Spread3 | A planning/reward module in `src/prompts/gen_code/mpe_simple_spread_v3_3/code/`. Candidate 1 action definitions, action mapping, and difference credit modules are already included. |
| Spread4 | A planning/reward module in `src/prompts/gen_code/mpe_simple_spread_v3_4/code/`, plus `knowledge_action_defs_mpe_simple_spread_v3_4_candidate2.py`, `knowledge_actions_mpe_simple_spread_v3_4_candidate2.py`, and `difference_credit_mpe_simple_spread_v3_4_candidate2.py` under `src/prompts/env_code/mpe/`. |
| LBF2 | The selected planning/reward module in `src/prompts/gen_code/lbf_2p_2f_coop/code/` (the server log identified `qwen_turbo_generated_code_2.py`), plus `knowledge_action_defs_lbf_2p_2f_coop_candidate1.py`, `knowledge_actions_lbf_2p_2f_coop_candidate1.py`, and `difference_credit_lbf_2p_2f_coop_candidate1.py` under `src/prompts/env_code/lbf/`. |

The local LBF2 tree contained `claude_generated_code_0.py`, but it is **not** the module named by the formal server log; it has not been substituted here. Check the exact selected planning/reward module on each server because the loader chooses the highest numeric `*_generated_code_*.py` present, then the newest by modification time.

Regenerating with the supplied code is possible after configuring the API key, but this creates a new set of candidates, not the original functions. Keep regenerated runs separate from the reported results.
