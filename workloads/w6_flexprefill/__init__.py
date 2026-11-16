"""W6: Llama-3.1-8B-Instruct prefill with FlexPrefill block selection (32 query heads, 8 KV heads, d = 128, 128 x 128 tiles).

Pieces: `common` (head views, kernel tables, mask exchange), `selection` (distributed
FlexPrefill indexing), `model` (the model's forward pieces), `prep_inputs` + `dump_qkv` + `official_reference` (data prep).
Data lives under $DEX_DATA_ROOT/w6_flexprefill: inputs/<input>.npy, qkv/<input>/L<li>_{q,k,v,mask_g095}.npy,
official/<input>_{logits,hid,tok}.npy. The official FlexPrefill package (flex_prefill @ a24b27bf) must be importable."""
