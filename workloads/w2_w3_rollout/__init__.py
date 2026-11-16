"""W2 / W3: packed shared-prefix rollout groups (one prompt + 16 sampled replies) from LongReason.

W3 uses the Qwen3-4B attention shape (32 query / 8 KV heads); W2 the same 24 sequences with the Qwen1.5-1.8B shape
(16 / 16). Pipeline: sample_rollouts (vLLM) -> select_groups -> gen_masks."""
