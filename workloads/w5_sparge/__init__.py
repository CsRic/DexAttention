"""W5: Wan2.1-T2V-1.3B video generation with SpargeAttention2 block selection (12 heads, d = 128, 128 x 64 blocks).

Pieces: `selection` (SpargeAttention2 top-k / top-p block selection), `cases` (operator-test case loading and timing), `model` (the Wan transformer harness of the training step),
`dump_hook` + `generate.py` (the q / k dumps of the 32 generations), `prep_e2e_inputs` (latents and text embeddings).
Data lives under $DEX_DATA_ROOT/w5_sparge: dumps/<generation>/rank<r>/call<i>.npz, videos/<generation>.mp4,
e2e_inputs/<generation>.pt."""
