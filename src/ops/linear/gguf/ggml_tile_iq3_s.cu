// The tile product for one block type, in its own unit so the heavy template builds in parallel
// with the others.
#include "ggml_bridge_tile.cuh"

NINFER_GGUF_TILE_INSTANCE(GGML_TYPE_IQ3_S);
