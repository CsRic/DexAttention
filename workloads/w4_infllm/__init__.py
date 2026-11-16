"""W4: MiniCPM4.1-8B with InfLLM-v2 block-sparse attention (32 query heads, 2 KV heads, head dim 128, BF16).

minicpm_model.py   the model pieces (LongRoPE, RMSNorm, the InfLLM-v2 selection and attention on the official kernels)
prepare_data.py    LongBench v2 prompts -> token ids per length bucket
sample_masks.py    layer-major prefill that records every layer's selection (and the attention inputs of sampled layers)
stage1_fast.py     InfLLM-v2 stage-1 scoring with a large tile (Triton) and the LSE merges of partial outputs
common.py          context-parallel layout (zigzag chunks), selection under CP, row layouts, the recompute-reuse cache
"""
