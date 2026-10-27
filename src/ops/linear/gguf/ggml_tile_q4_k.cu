// The tile product for one block type, in its own unit so the heavy template builds in parallel
// with the others.
#include "ggml_bridge_tile.cuh"

NINFER_GGUF_TILE_INSTANCE(GGML_TYPE_Q4_K);

// The NVFP4 two-pass product's types without a tile kernel.
NINFER_GGUF_FP4_INSTANCE(GGML_TYPE_IQ1_S);
NINFER_GGUF_FP4_INSTANCE(GGML_TYPE_IQ1_M);
NINFER_GGUF_FP4_INSTANCE(GGML_TYPE_Q6_K);
