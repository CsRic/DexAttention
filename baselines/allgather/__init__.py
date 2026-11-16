"""All-gather baseline: gather K / V on every rank, one single-GPU sparse kernel call per rank, reduce-scatter dK / dV
back to the owners. static binds it to W1-W3; infllm, sparge and flexprefill bind it to W4, W5 and W6 (W6 forward
only); sparse_allgather holds the Sparse All-gather pieces shared by the W5 and W6 ops."""
