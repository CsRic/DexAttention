"""Ring baseline: dense KV circulation (upstream USP / Yunchang ring communication and merge) with sparse per-step
compute. ring_sparse holds the workload-independent ring loops and the tile compute objects; static (W1-W3), infllm
(W4), sparge (W5) and flexprefill (W6) bind them to a workload."""
