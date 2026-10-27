#include "ops/linear/gguf/gguf_linear.h"

#include "core/layout.h"
#include "ops/linear/sm70/sm70_gemm.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::detail {
namespace {

// Rows the dequantize-and-GEMM route converts per pass for a type with no integer kernel.
constexpr std::size_t kDequantizedScratchBytes = std::size_t{16} << 20;
// Up to this many columns such a type loops the vector kernel instead: those are decode and verify
// batches, captured in graphs, where a cuBLAS call is best avoided.
constexpr std::int32_t kDequantizedVectorColumns = 64;
// The widest stream-k fixup plane any launch can ask for: one 128 x 128 FP32 tile per SM.
constexpr std::size_t kFixupTileBytes = 128 * 128 * sizeof(float);

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

bool matrix_kernel(const Weight& w) { return gguf::has_matrix_kernel(gguf_type(w.qtype)); }

gguf::Epilogue bridge_epilogue(GgufEpilogue epilogue) {
    switch (epilogue) {
    case GgufEpilogue::Store: return gguf::Epilogue::Store;
    case GgufEpilogue::Accumulate: return gguf::Epilogue::Accumulate;
    case GgufEpilogue::GateProduct: return gguf::Epilogue::GateProduct;
    }
    throw std::invalid_argument("gguf: invalid epilogue");
}

// The destination of columns [first, ...) of a product.
gguf::VectorOutput output(const GgufProduct& p, std::int32_t first) {
    gguf::VectorOutput out;
    out.epilogue = bridge_epilogue(p.epilogue);
    if (p.f32 != nullptr) {
        out.f32           = p.f32 + std::int64_t(first) * p.f32_rows + p.row;
        out.column_stride = p.f32_rows;
    } else {
        out.bf16 = static_cast<__nv_bfloat16*>(p.out->data) + std::int64_t(first) * p.out->ne[0] +
                   p.row;
        out.column_stride = p.out->ne[0];
    }
    if (p.epilogue == GgufEpilogue::GateProduct) {
        out.gate               = p.gate + std::int64_t(first) * p.weight->n;
        out.gate_column_stride = p.weight->n;
        out.activation         = p.activation;
    }
    return out;
}

void require_activation(const Tensor& x, const char* label) {
    if (x.dtype != DType::BF16 || x.ne[0] <= 0 || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || x.data == nullptr) {
        throw std::invalid_argument(std::string(label) + ": x must be contiguous BF16 [K,T]");
    }
}

void require_product(const Tensor& x, const GgufProduct& p) {
    require_gguf(*p.weight, "gguf product");
    if (p.weight->k != x.ne[0]) { throw std::invalid_argument("gguf product: K differs from x"); }
    if (p.f32 != nullptr) {
        if (p.row < 0 || p.row + p.weight->n > p.f32_rows) {
            throw std::invalid_argument("gguf product: rows exceed the FP32 plane");
        }
        return;
    }
    if (p.out == nullptr || p.out->dtype != DType::BF16 || !p.out->is_contiguous() ||
        p.out->ne[1] != x.ne[1] || p.row < 0 || p.row + p.weight->n > p.out->ne[0]) {
        throw std::invalid_argument("gguf product: rows exceed the BF16 destination");
    }
    if (p.epilogue == GgufEpilogue::GateProduct && p.gate == nullptr) {
        throw std::invalid_argument("gguf product: the gate product needs its gate plane");
    }
}

std::size_t fixup_bound() {
    return std::size_t(gguf::multiprocessor_count()) * kFixupTileBytes;
}

// The destination of rows [first, ...) of an output.
gguf::VectorOutput from_row(gguf::VectorOutput out, std::int32_t first) {
    if (out.bf16 != nullptr) { out.bf16 += first; }
    if (out.f32 != nullptr) { out.f32 += first; }
    if (out.gate != nullptr) { out.gate += first; }
    return out;
}

// Workspace of sm70_project for `products` products of up to `rows` rows over `t` columns.
std::size_t sm70_project_bytes(std::size_t products, std::int32_t rows, std::int32_t k,
                               std::int32_t t) {
    const std::size_t slab = static_cast<std::size_t>(detail::sm70::slab_rows(rows, k));
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc_bytes(slab * t * sizeof(float));
    (void)layout.alloc_bytes(slab * k * sizeof(__half));
    for (std::size_t i = 0; i < products; ++i) {
        (void)layout.alloc_bytes(std::size_t(k) * t * 2);
        (void)layout.alloc_bytes(detail::sm70::activation_bytes(k, t));
    }
    return layout.peak_bytes(1);
}

// sm_70 (ops/linear/sm70): every product converts its activation to FP16 once per input gather,
// dequantizes its weight to FP16 a slab of rows at a time and multiplies each slab through cuBLAS
// into an FP32 plane of the slab's rows, which the product's epilogue then stores.
void sm70_project(const Tensor& x, std::span<const GgufProduct> products, WorkspaceArena& workspace,
                  cudaStream_t stream) {
    const std::int32_t k    = x.ne[0];
    const std::int32_t t    = x.ne[1];
    const auto* xb          = static_cast<const __nv_bfloat16*>(x.data);
    std::int32_t widest = 0;
    for (const auto& p : products) { widest = std::max(widest, p.weight->n); }
    const std::size_t widest_slab = static_cast<std::size_t>(detail::sm70::slab_rows(widest, k));
    auto* plane =
        static_cast<float*>(workspace.alloc_bytes(widest_slab * t * sizeof(float)).data);
    auto* weights =
        static_cast<__half*>(workspace.alloc_bytes(widest_slab * k * sizeof(__half)).data);

    struct Activation {
        const std::int32_t* columns;
        const __half* data;
    };

    std::vector<Activation> activations;
    for (const auto& p : products) {
        const Weight& w = *p.weight;
        auto found = std::find_if(activations.begin(), activations.end(), [&](const Activation& a) {
            return a.columns == w.input_columns;
        });
        if (found == activations.end()) {
            const __nv_bfloat16* source = xb;
            if (w.input_columns != nullptr) {
                auto* gathered =
                    static_cast<__nv_bfloat16*>(workspace.alloc_bytes(std::size_t(k) * t * 2).data);
                gguf::gather_columns(xb, k, t, w.input_columns, gathered, stream);
                source = gathered;
            }
            auto* converted = static_cast<__half*>(
                workspace.alloc_bytes(detail::sm70::activation_bytes(k, t)).data);
            detail::sm70::convert_activation(source, k, t, converted, stream);
            activations.push_back({w.input_columns, converted});
            found = activations.end() - 1;
        }
        const auto type         = gguf_type(w.qtype);
        const bool direct       = p.f32 != nullptr && p.epilogue == GgufEpilogue::Store;
        const std::int32_t slab = detail::sm70::slab_rows(w.n, k);
        for (std::int32_t first = 0; first < w.n; first += slab) {
            const std::int32_t count = std::min(slab, w.n - first);
            gguf::dequantize_rows(type, static_cast<const char*>(w.qdata) + first * row_bytes(w),
                                  row_bytes(w), k, nullptr, count, weights, k, stream);
            if (direct) {
                detail::sm70::gemm(weights, count, k, found->data, t, p.f32 + p.row + first,
                                   p.f32_rows, stream);
            } else {
                detail::sm70::gemm(weights, count, k, found->data, t, plane, count, stream);
                gguf::store_plane(plane, count, count, t, from_row(output(p, 0), first), stream);
            }
        }
    }
}

} // namespace

gguf::GgmlType gguf_type(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return gguf::GgmlType::Q8_0;
    case QType::GGUF_Q2_K: return gguf::GgmlType::Q2_K;
    case QType::GGUF_Q3_K: return gguf::GgmlType::Q3_K;
    case QType::GGUF_Q4_K: return gguf::GgmlType::Q4_K;
    case QType::GGUF_Q5_K: return gguf::GgmlType::Q5_K;
    case QType::GGUF_Q6_K: return gguf::GgmlType::Q6_K;
    case QType::GGUF_IQ2_XXS: return gguf::GgmlType::IQ2_XXS;
    case QType::GGUF_IQ2_XS: return gguf::GgmlType::IQ2_XS;
    case QType::GGUF_IQ2_S: return gguf::GgmlType::IQ2_S;
    case QType::GGUF_IQ3_XXS: return gguf::GgmlType::IQ3_XXS;
    case QType::GGUF_IQ3_S: return gguf::GgmlType::IQ3_S;
    case QType::GGUF_IQ1_S: return gguf::GgmlType::IQ1_S;
    case QType::GGUF_IQ1_M: return gguf::GgmlType::IQ1_M;
    case QType::GGUF_IQ4_NL: return gguf::GgmlType::IQ4_NL;
    case QType::GGUF_IQ4_XS: return gguf::GgmlType::IQ4_XS;
    case QType::GGUF_Q4_0:
        return gguf::GgmlType::Q4_0;
    case QType::GGUF_Q5_0:
        return gguf::GgmlType::Q5_0;
    case QType::GGUF_Q2_0:
        return gguf::GgmlType::Q2_0;
    default: break;
    }
    throw std::invalid_argument("gguf: not a GGUF block format");
}

void require_gguf(const Weight& w, const char* label) {
    const auto block = gguf_block_shape(w.qtype);
    if (!is_gguf(w.qtype) || w.layout != QuantLayout::GgufBlocks || w.qdata == nullptr ||
        w.n <= 0 || w.k <= 0 || w.k % block.elements != 0 || w.k % 32 != 0) {
        throw std::invalid_argument(std::string(label) +
                                    ": expected a resident GGUF matrix over whole blocks");
    }
}

namespace {

// The tile kernel's MXFP8 form, where the build has it and the policy allows A8 activations.
bool tile_mxfp8(LinearPolicy policy) { return gguf::kTileMxfp8Built && allows_a8(policy); }

// The two-pass form of products of a matrix (gate = up) or a gated pair of these formats over
// K = k: NVFP4 where the policy allows A4 activations, else MXFP8 where it allows A8; none where
// neither serves both.
std::optional<gguf::LtForm> lt_form(QType gate, QType up, std::int32_t k, LinearPolicy policy) {
    for (const gguf::LtForm form : {gguf::LtForm::Nvfp4, gguf::LtForm::Mxfp8}) {
        const bool allowed = form == gguf::LtForm::Nvfp4 ? allows_a4(policy) : allows_a8(policy);
        if (allowed && gguf::has_lt_product(gguf_type(gate), k, form) &&
            gguf::has_lt_product(gguf_type(up), k, form)) {
            return form;
        }
    }
    return std::nullopt;
}

// That form where t columns reach its gguf::lt_min_columns, else none.
std::optional<gguf::LtForm> lt_form_at(QType gate, QType up, std::int32_t k, std::int32_t t,
                                       LinearPolicy policy) {
    const auto form = lt_form(gate, up, k, policy);
    if (form && t >= gguf::lt_min_columns(*form)) { return form; }
    return std::nullopt;
}

// The weight max |w| a product in `form` takes: NVFP4's over both matrices, each measured at load.
float lt_amax(gguf::LtForm form, const Weight& gate, const Weight& up) {
    if (form != gguf::LtForm::Nvfp4) { return 0.0F; }
    if (!(gate.gguf_amax > 0.0F) || !(up.gguf_amax > 0.0F)) {
        throw std::invalid_argument(
            "gguf product: a matrix allowed A4 activations lacks its max |w| (gguf_max_abs)");
    }
    return std::max(gate.gguf_amax, up.gguf_amax);
}

// The weight's choice table for a product (gguf::Fp4Choices).
gguf::Fp4Choices fp4_choices(const Weight& w) { return {w.fp4_choices, w.fp4_choices_amax}; }

} // namespace

std::size_t gguf_project_workspace_bytes(std::span<const GgufShape> shapes, LinearPolicy policy,
                                         std::int32_t min_tokens, std::int32_t max_tokens,
                                         bool x_activation) {
    if (shapes.empty() || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gguf workspace: invalid products or token interval");
    }
    const std::int32_t k = shapes.front().k;
    std::size_t rows = 0;
    for (const GgufShape& shape : shapes) {
        const auto block = gguf_block_shape(shape.qtype);
        if (!is_gguf(shape.qtype) || shape.rows <= 0 || shape.k != k || k <= 0 ||
            k % block.elements != 0 || k % 32 != 0) {
            throw std::invalid_argument("gguf workspace: unsupported product geometry");
        }
        rows = std::max<std::size_t>(rows, shape.rows);
    }
    // Every product may carry its own input gather, so each may quantize its own activation.
    const std::size_t activations = shapes.size();
    const bool mxfp8              = tile_mxfp8(policy);
    // The wide route at width t (gguf_project): products the tile kernel takes need neither the
    // FP32 plane nor the stream-k fixup.
    const auto matrix_bytes = [&](std::int32_t t) {
        std::vector<int> layouts;
        std::size_t plane_rows = 0;
        bool untiled = false, tiled = false, dequantized = false, narrowed = false;
        for (const GgufShape& shape : shapes) {
            const auto type = gguf_type(shape.qtype);
            if (gguf::takes_narrow_kernel(type, k, t)) {
                narrowed = true;
                continue;
            }
            if (gguf::takes_tile_kernel(type, shape.rows, k, t, mxfp8)) {
                tiled = true;
                continue;
            }
            untiled    = true;
            plane_rows = std::max<std::size_t>(plane_rows, shape.rows);
            if (gguf::has_matrix_kernel(type)) {
                const int layout = gguf::matrix_activation_layout(type);
                if (std::find(layouts.begin(), layouts.end(), layout) == layouts.end()) {
                    layouts.push_back(layout);
                }
            } else {
                dequantized = true;
            }
        }
        // A product that may take the two-pass product (its output decides) is planned for both
        // routes.
        std::size_t two_pass = 0, two_pass_activation = 0;
        for (const GgufShape& shape : shapes) {
            if (const auto form = lt_form_at(shape.qtype, shape.qtype, k, t, policy)) {
                two_pass = std::max(
                    two_pass, gguf::lt_planned_scratch_bytes(k, shape.rows, t, false, *form));
                two_pass_activation =
                    std::max(two_pass_activation, gguf::lt_activation_bytes(k, t, *form));
            }
        }
        WorkspaceLayoutBuilder layout;
        if (untiled) {
            (void)layout.alloc_bytes(plane_rows * t * sizeof(float));
            (void)layout.alloc_bytes(fixup_bound());
        }
        if (dequantized) {
            (void)layout.alloc_bytes(gguf::vector_activation_bytes(k, gguf::kMaxVectorColumns));
            (void)layout.alloc_bytes(kDequantizedScratchBytes);
        }
        if (two_pass > 0) { (void)layout.alloc_bytes(two_pass); }
        if (untiled) {
            for (std::size_t i = 0; i < std::max<std::size_t>(layouts.size(), 1) * activations;
                 ++i) {
                (void)layout.alloc_bytes(
                    gguf::matrix_activation_bytes(gguf::matrix_padded_k(k), t));
            }
        }
        if (tiled) {
            for (std::size_t i = 0; i < activations; ++i) {
                (void)layout.alloc_bytes(gguf::tile_activation_bytes(k, t));
            }
        }
        if (narrowed) {
            for (std::size_t i = 0; i < activations; ++i) {
                (void)layout.alloc_bytes(gguf::narrow_activation_bytes(k, t));
            }
        }
        if (two_pass > 0) {
            // A provided activation stands for the ungathered NVFP4 one.
            for (std::size_t i = x_activation ? 1 : 0; i < activations; ++i) {
                (void)layout.alloc_bytes(two_pass_activation);
            }
        }
        if (dequantized) { (void)layout.alloc_bytes(std::size_t(k) * t * 2); }
        return layout.peak_bytes(1);
    };
    std::size_t widest            = 0;
    if (detail::sm70::takes(max_tokens)) {
        widest = sm70_project_bytes(activations, static_cast<std::int32_t>(rows), k, max_tokens);
    }
    // Each product quantizes its own vector activation, or the narrow one where it takes the narrow
    // kernel.
    for (std::int32_t t = min_tokens; t <= std::min(max_tokens, gguf::kMaxVectorColumns); ++t) {
        WorkspaceLayoutBuilder layout;
        for (const GgufShape& shape : shapes) {
            (void)layout.alloc_bytes(gguf::takes_narrow_kernel(gguf_type(shape.qtype), k, t)
                                         ? gguf::narrow_activation_bytes(k, t)
                                         : gguf::vector_activation_bytes(k, t));
        }
        widest = std::max(widest, layout.peak_bytes(1));
    }
    // Which products take the tile kernel changes with the width and not monotonically (its grid's
    // wave fill), so every wide width is planned.
    for (std::int32_t t = std::max(min_tokens, gguf::kMaxVectorColumns + 1); t <= max_tokens; ++t) {
        widest = std::max(widest, matrix_bytes(t));
    }
    return widest;
}

void gguf_project(const Tensor& x, std::span<const GgufProduct> products, LinearPolicy policy,
                  WorkspaceArena& workspace, cudaStream_t stream, const void* x_activation,
                  const std::int32_t* x_activation_columns) {
    require_activation(x, "gguf product");
    if (products.empty()) { return; }
    for (const auto& p : products) { require_product(x, p); }
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const auto* xb       = static_cast<const __nv_bfloat16*>(x.data);
    auto scope           = workspace.scope();

    if (detail::sm70::takes(t)) {
        sm70_project(x, products, workspace, stream);
        return;
    }

    // Products the narrow kernel takes, at any width: one narrow activation per input gather.
    struct NarrowActivation {
        const std::int32_t* columns;
        void* data;
    };
    std::vector<NarrowActivation> narrow_activations;
    const auto narrow = [&](const GgufProduct& p) {
        const Weight& w = *p.weight;
        const auto type = gguf_type(w.qtype);
        if (!gguf::takes_narrow_kernel(type, k, t)) { return false; }
        auto found = std::find_if(narrow_activations.begin(), narrow_activations.end(),
                                  [&](const NarrowActivation& a) { return a.columns == w.input_columns; });
        if (found == narrow_activations.end()) {
            void* data = workspace.alloc_bytes(gguf::narrow_activation_bytes(k, t)).data;
            gguf::quantize_narrow_activation(xb, k, t, w.input_columns, data, stream);
            narrow_activations.push_back({w.input_columns, data});
            found = narrow_activations.end() - 1;
        }
        gguf::narrow_product(type, w.qdata, row_bytes(w), w.n, k, found->data, t, output(p, 0),
                             stream);
        return true;
    };

    if (t <= gguf::kMaxVectorColumns) {
        struct Activation {
            const std::int32_t* columns;
            void* data;
        };
        std::vector<Activation> activations;
        for (const auto& p : products) {
            if (narrow(p)) { continue; }
            const Weight& w = *p.weight;
            auto found      = std::find_if(activations.begin(), activations.end(),
                                           [&](const Activation& a) { return a.columns == w.input_columns; });
            if (found == activations.end()) {
                void* data = workspace.alloc_bytes(gguf::vector_activation_bytes(k, t)).data;
                gguf::quantize_vector_activation(xb, k, t, w.input_columns, data, stream);
                activations.push_back({w.input_columns, data});
                found = activations.end() - 1;
            }
            gguf::vector_product(gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, k, found->data,
                                 t, output(p, 0), stream);
        }
        return;
    }

    struct Activation {
        int layout; // the matrix kernel's layout, kTileLayout or a two-pass form's
        const std::int32_t* columns;
        void* data;
    };
    constexpr int kTileLayout  = -1;
    constexpr int kMxfp8Layout = -2;
    constexpr int kNvfp4Layout = -3;
    const bool mxfp8           = tile_mxfp8(policy);
    // From its gguf::lt_min_columns the BF16 Store and Accumulate products of the types with a
    // two-pass form take it.
    const auto two_pass = [&](const GgufProduct& p) -> std::optional<gguf::LtForm> {
        if (!gguf::lt_takes_output(output(p, 0))) { return std::nullopt; }
        return lt_form_at(p.weight->qtype, p.weight->qtype, k, t, policy);
    };
    // The tile kernel's epilogue is fused: its products need no plane.
    const auto tiled = [&](const Weight& w) {
        return gguf::takes_tile_kernel(gguf_type(w.qtype), w.n, k, t, mxfp8);
    };
    std::vector<Activation> activations;
    std::int32_t plane_rows = 0;
    bool untiled            = false;
    bool dequantized        = false;
    std::size_t lt_bytes    = 0;
    for (const auto& p : products) {
        if (gguf::takes_narrow_kernel(gguf_type(p.weight->qtype), k, t)) { continue; }
        if (const auto form = two_pass(p)) {
            lt_bytes = std::max(lt_bytes,
                                gguf::lt_planned_scratch_bytes(k, p.weight->n, t, false, *form));
            continue;
        }
        if (tiled(*p.weight)) { continue; }
        untiled    = true;
        plane_rows = std::max(plane_rows, p.weight->n);
        dequantized |= !matrix_kernel(*p.weight);
    }
    float* plane = nullptr;
    void* fixup  = nullptr;
    if (untiled) {
        plane = static_cast<float*>(
            workspace.alloc_bytes(std::size_t(plane_rows) * t * sizeof(float)).data);
        fixup = workspace.alloc_bytes(fixup_bound()).data;
    }
    void* vector_activation = nullptr;
    void* scratch           = nullptr;
    if (dequantized) {
        vector_activation =
            workspace.alloc_bytes(gguf::vector_activation_bytes(k, gguf::kMaxVectorColumns)).data;
        scratch = workspace.alloc_bytes(kDequantizedScratchBytes).data;
    }
    void* lt_scratch = lt_bytes > 0 ? workspace.alloc_bytes(lt_bytes).data : nullptr;

    for (const auto& p : products) {
        if (narrow(p)) { continue; }
        const Weight& w  = *p.weight;
        const auto type  = gguf_type(w.qtype);
        if (const auto form = two_pass(p)) {
            const int layout = *form == gguf::LtForm::Nvfp4 ? kNvfp4Layout : kMxfp8Layout;
            auto found = std::find_if(activations.begin(), activations.end(), [&](const Activation& a) {
                return a.layout == layout && a.columns == w.input_columns;
            });
            if (found == activations.end()) {
                void* data = nullptr;
                if (x_activation != nullptr && *form == gguf::LtForm::Nvfp4 &&
                    w.input_columns == x_activation_columns) {
                    data = const_cast<void*>(x_activation);
                } else {
                    data = workspace.alloc_bytes(gguf::lt_activation_bytes(k, t, *form)).data;
                    gguf::quantize_lt_activation(xb, k, t, w.input_columns, *form, data, stream);
                }
                activations.push_back({layout, w.input_columns, data});
                found = activations.end() - 1;
            }
            gguf::lt_product(type, w.qdata, row_bytes(w), w.n, k, *form, lt_amax(*form, w, w),
                             fp4_choices(w), found->data, t, output(p, 0), lt_scratch, lt_bytes,
                             stream);
            continue;
        }
        if (tiled(w)) {
            auto found = std::find_if(activations.begin(), activations.end(), [&](const Activation& a) {
                return a.layout == kTileLayout && a.columns == w.input_columns;
            });
            if (found == activations.end()) {
                void* data = workspace.alloc_bytes(gguf::tile_activation_bytes(k, t)).data;
                gguf::quantize_tile_activation(xb, k, t, w.input_columns, mxfp8, data, stream);
                activations.push_back({kTileLayout, w.input_columns, data});
                found = activations.end() - 1;
            }
            gguf::tile_product(type, w.qdata, row_bytes(w), w.n, k, found->data, mxfp8, t,
                               output(p, 0), stream);
            continue;
        }
        const bool direct = p.f32 != nullptr && p.epilogue == GgufEpilogue::Store;
        float* out        = direct ? p.f32 + p.row : plane;
        const std::int64_t stride = direct ? p.f32_rows : w.n;
        if (gguf::has_matrix_kernel(type)) {
            const int layout = gguf::matrix_activation_layout(type);
            auto found = std::find_if(activations.begin(), activations.end(), [&](const Activation& a) {
                return a.layout == layout && a.columns == w.input_columns;
            });
            // Rows that are not whole 256-value steps (Gemma 4's 2112- and 704-wide inputs) read
            // past their end into the next row and the zeros the loader places after the last
            // one, against an activation that is zero there.
            const bool padded = k % 512 != 0;
            if (found == activations.end()) {
                void* data = workspace.alloc_bytes(
                    gguf::matrix_activation_bytes(gguf::matrix_padded_k(k), t)).data;
                if (padded) {
                    gguf::quantize_matrix_activation_padded(type, xb, k, t, w.input_columns, data,
                                                            stream);
                } else {
                    gguf::quantize_matrix_activation(type, xb, k, t, w.input_columns, data, stream);
                }
                activations.push_back({layout, w.input_columns, data});
                found = activations.end() - 1;
            }
            gguf::matrix_product(type, w.qdata, row_bytes(w), w.n, k, found->data, t, out, stride,
                                 fixup, stream, k % 256 != 0);
        } else if (t <= kDequantizedVectorColumns) {
            for (std::int32_t first = 0; first < t; first += gguf::kMaxVectorColumns) {
                const std::int32_t columns = std::min(gguf::kMaxVectorColumns, t - first);
                gguf::quantize_vector_activation(xb + std::int64_t(first) * k, k, columns,
                                                 w.input_columns, vector_activation, stream);
                gguf::vector_product(type, w.qdata, row_bytes(w), w.n, k, vector_activation,
                                     columns, output(p, first), stream);
            }
            continue;
        } else {
            const __nv_bfloat16* source = xb;
            if (w.input_columns != nullptr) {
                auto* gathered = static_cast<__nv_bfloat16*>(
                    workspace.alloc_bytes(std::size_t(k) * t * 2).data);
                gguf::gather_columns(xb, k, t, w.input_columns, gathered, stream);
                source = gathered;
            }
            gguf::dequantized_product(type, w.qdata, row_bytes(w), w.n, k, source, t, out, stride,
                                      scratch, kDequantizedScratchBytes, stream);
        }
        if (!direct) { gguf::store_plane(plane, w.n, w.n, t, output(p, 0), stream); }
    }
}

void gguf_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    const GgufProduct product{&w, &out, 0, GgufEpilogue::Store};
    gguf_project(x, {&product, 1}, policy, workspace, stream);
}

void gguf_linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream, const void* x_activation) {
    const GgufProduct product{&w, &residual, 0, GgufEpilogue::Accumulate};
    gguf_project(x, {&product, 1}, policy, workspace, stream, x_activation, w.input_columns);
}

bool gguf_takes_nvfp4_activation(const Weight& w, LinearPolicy policy, std::int32_t t) {
    return is_gguf(w.qtype) && w.layout == QuantLayout::GgufBlocks &&
           lt_form_at(w.qtype, w.qtype, w.k, t, policy) == gguf::LtForm::Nvfp4;
}

namespace {

// The two-pass form a gated pair of `rows` rows takes at width t, if any.
std::optional<gguf::LtForm> two_pass_gated(QType gate, QType up, std::int32_t rows, std::int32_t k,
                                           std::int32_t t, LinearPolicy policy) {
    if (rows % 8 != 0) { return std::nullopt; }
    return lt_form_at(gate, up, k, t, policy);
}

// Its workspace: an activation each for gate and up (their gathers may differ; one fewer where
// the caller provides the ungathered one) and the scratch.
std::size_t two_pass_gated_bytes(std::int32_t rows, std::int32_t k, std::int32_t t,
                                 gguf::LtForm form, bool x_activation) {
    return (x_activation ? 1 : 2) * (gguf::lt_activation_bytes(k, t, form) + 256) +
           gguf::lt_planned_scratch_bytes(k, rows, t, true, form) + 256;
}

// Whether gguf_swiglu_add fuses at width t: the gated pair takes the NVFP4 two-pass product over
// whole 256-row tiles, which encodes its output as the NVFP4 activation of down, and down (over
// that activation, ungathered) takes the NVFP4 two-pass product.
bool fused_swiglu_add(QType gate, QType up, std::int32_t rows, std::int32_t k, QType down,
                      std::int32_t t, LinearPolicy gate_policy, LinearPolicy down_policy) {
    return rows % 256 == 0 &&
           two_pass_gated(gate, up, rows, k, t, gate_policy) == gguf::LtForm::Nvfp4 &&
           lt_form_at(down, down, rows, t, down_policy) == gguf::LtForm::Nvfp4;
}

} // namespace

std::size_t gguf_swiglu_workspace_bytes(const GgufShape& gate, const GgufShape* up,
                                        LinearPolicy policy, std::int32_t min_tokens,
                                        std::int32_t max_tokens, bool x_activation) {
    const std::int32_t rows = up == nullptr ? gate.rows / 2 : gate.rows;
    const QType up_type     = up == nullptr ? gate.qtype : up->qtype;
    // The planes route (gate products into an FP32 plane, up products through GateProduct).
    const auto planes = [&](std::int32_t first, std::int32_t last) {
        if (up == nullptr) {
            const GgufShape halves[] = {{gate.qtype, rows, gate.k}, {gate.qtype, rows, gate.k}};
            return gguf_project_workspace_bytes(halves, policy, first, last) +
                   std::size_t(rows) * last * sizeof(float) + 256;
        }
        const GgufShape parents[] = {gate, *up};
        return gguf_project_workspace_bytes(parents, policy, first, last) +
               std::size_t(rows) * last * sizeof(float) + 256;
    };
    const auto form = two_pass_gated(gate.qtype, up_type, rows, gate.k, max_tokens, policy);
    if (!form) { return planes(min_tokens, max_tokens); }
    const std::size_t wide = two_pass_gated_bytes(rows, gate.k, max_tokens, *form, x_activation);
    const std::int32_t floor = gguf::lt_min_columns(*form);
    if (min_tokens >= floor) { return wide; }
    return std::max(wide, planes(min_tokens, floor - 1));
}

namespace {

// gguf_swiglu; where it takes the NVFP4 two-pass product, a non-null out_activation (see
// gguf::lt_gated_product) also receives its output's NVFP4 activation, and it returns true.
bool swiglu(const Tensor& x, const Weight& gate, const Weight* up, Tensor& out,
            LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream,
            gguf::GateActivation activation, void* out_activation, const void* x_activation) {
    require_activation(x, "gguf swiglu");
    require_gguf(gate, "gguf swiglu gate");
    const std::int32_t t    = x.ne[1];
    const std::int32_t rows = out.ne[0];
    if (out.dtype != DType::BF16 || !out.is_contiguous() || out.ne[1] != t) {
        throw std::invalid_argument("gguf swiglu: out must be contiguous BF16 [N,T]");
    }
    auto scope = workspace.scope();
    Weight gate_rows = gate;
    Weight up_rows;
    if (up == nullptr) {
        if (gate.n != 2 * rows) { throw std::invalid_argument("gguf swiglu: parent is not [gate; up]"); }
        if (t <= gguf::kMaxVectorColumns &&
            !gguf::takes_narrow_kernel(gguf_type(gate.qtype), gate.k, t)) {
            void* vector_input =
                workspace.alloc_bytes(gguf::vector_activation_bytes(gate.k, t)).data;
            gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data), gate.k, t,
                                             gate.input_columns, vector_input, stream);
            gguf::vector_swiglu(gguf_type(gate.qtype), gate.qdata, row_bytes(gate), rows, gate.k,
                                vector_input, t, static_cast<__nv_bfloat16*>(out.data), rows,
                                stream, activation);
            return false;
        }
        gate_rows.n = rows;
        up_rows     = gate;
        up_rows.n   = rows;
        up_rows.qdata = static_cast<const std::byte*>(gate.qdata) + std::int64_t(rows) * row_bytes(gate);
        up_rows.fp4_choices = gguf::fp4_choice_rows(gate.fp4_choices, rows, gate.k);
    } else {
        require_gguf(*up, "gguf swiglu up");
        if (gate.n != rows || up->n != rows || up->k != gate.k) {
            throw std::invalid_argument("gguf swiglu: gate and up must both be [N,K]");
        }
        up_rows = *up;
    }
    auto* out_data = static_cast<__nv_bfloat16*>(out.data);
    const auto form = two_pass_gated(gate_rows.qtype, up_rows.qtype, rows, gate.k, t, policy);
    if (form && reinterpret_cast<std::uintptr_t>(out_data) % 16 == 0) {
        const auto* xb = static_cast<const __nv_bfloat16*>(x.data);
        const std::size_t activation_bytes = gguf::lt_activation_bytes(gate.k, t, *form);
        // The provided activation of x where the form is NVFP4 and the half ungathered.
        const auto input = [&](const std::int32_t* columns) {
            if (x_activation != nullptr && *form == gguf::LtForm::Nvfp4 && columns == nullptr) {
                return const_cast<void*>(x_activation);
            }
            void* data = workspace.alloc_bytes(activation_bytes).data;
            gguf::quantize_lt_activation(xb, gate.k, t, columns, *form, data, stream);
            return data;
        };
        void* gate_x = input(gate_rows.input_columns);
        void* up_x   = nullptr;
        if (up_rows.input_columns != gate_rows.input_columns) { up_x = input(up_rows.input_columns); }
        const std::size_t scratch_bytes =
            gguf::lt_planned_scratch_bytes(gate.k, rows, t, true, *form);
        void* scratch = workspace.alloc_bytes(scratch_bytes).data;
        // The output's own activation only in the NVFP4 form, over contiguous output.
        void* encoded = *form == gguf::LtForm::Nvfp4 && out.is_contiguous() && rows % 256 == 0
                            ? out_activation
                            : nullptr;
        gguf::lt_gated_product(gguf_type(gate_rows.qtype), gate_rows.qdata, row_bytes(gate_rows),
                               gguf_type(up_rows.qtype), up_rows.qdata, row_bytes(up_rows), rows,
                               gate.k, *form, lt_amax(*form, gate_rows, up_rows),
                               fp4_choices(gate_rows), fp4_choices(up_rows), gate_x, up_x, t,
                               out_data, rows, activation, encoded, scratch, scratch_bytes, stream);
        return encoded != nullptr;
    }
    auto* gate_plane =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(rows) * t * sizeof(float)).data);
    GgufProduct products[2];
    products[0].weight   = &gate_rows;
    products[0].f32      = gate_plane;
    products[0].f32_rows = rows;
    products[1].weight   = &up_rows;
    products[1].out      = &out;
    products[1].epilogue   = GgufEpilogue::GateProduct;
    products[1].gate       = gate_plane;
    products[1].activation = activation;
    gguf_project(x, products, policy, workspace, stream, x_activation);
    return false;
}

} // namespace

void gguf_swiglu(const Tensor& x, const Weight& gate, const Weight* up, Tensor& out,
                 LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream,
                 gguf::GateActivation activation) {
    (void)swiglu(x, gate, up, out, policy, workspace, stream, activation, nullptr, nullptr);
}

void gguf_swiglu_add(const Tensor& x, const Weight& gate, const Weight* up, const Weight& down,
                     Tensor& residual, LinearPolicy gate_policy, LinearPolicy down_policy,
                     WorkspaceArena& workspace, cudaStream_t stream, const void* x_activation) {
    require_gguf(down, "gguf swiglu add down");
    const std::int32_t t    = x.ne[1];
    const std::int32_t rows = up == nullptr ? gate.n / 2 : gate.n;
    if (down.k != rows || residual.dtype != DType::BF16 || !residual.is_contiguous() ||
        residual.ne[0] != down.n || residual.ne[1] != t) {
        throw std::invalid_argument("gguf swiglu add: down must be [N, M] over a BF16 residual [N,T]");
    }
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {rows, t});
    const gguf::VectorOutput out{static_cast<__nv_bfloat16*>(residual.data), nullptr, down.n,
                                 gguf::Epilogue::Accumulate};
    const QType up_type = up == nullptr ? gate.qtype : up->qtype;
    if (!fused_swiglu_add(gate.qtype, up_type, rows, gate.k, down.qtype, t, gate_policy,
                          down_policy) ||
        down.input_columns != nullptr || !gguf::lt_takes_output(out) ||
        reinterpret_cast<std::uintptr_t>(activation.data) % 16 != 0) {
        {
            auto call = workspace.scope();
            (void)swiglu(x, gate, up, activation, gate_policy, workspace, stream,
                         gguf::GateActivation::Silu, nullptr, x_activation);
        }
        gguf_linear_add(activation, down, residual, down_policy, workspace, stream);
        return;
    }
    void* encoded =
        workspace.alloc_bytes(gguf::lt_activation_bytes(rows, t, gguf::LtForm::Nvfp4)).data;
    {
        auto call = workspace.scope();
        if (!swiglu(x, gate, up, activation, gate_policy, workspace, stream,
                    gguf::GateActivation::Silu, encoded, x_activation)) {
            throw std::logic_error("gguf swiglu add: the gated product did not encode down's input");
        }
    }
    // gguf_project's two-pass route over that activation, with its scratch.
    const std::size_t scratch_bytes =
        gguf::lt_planned_scratch_bytes(rows, down.n, t, false, gguf::LtForm::Nvfp4);
    void* scratch = workspace.alloc_bytes(scratch_bytes).data;
    gguf::lt_product(gguf_type(down.qtype), down.qdata, row_bytes(down), down.n, rows,
                     gguf::LtForm::Nvfp4, lt_amax(gguf::LtForm::Nvfp4, down, down),
                     fp4_choices(down), encoded, t, out, scratch, scratch_bytes, stream);
}

std::size_t gguf_swiglu_add_workspace_bytes(const GgufShape& gate, const GgufShape* up,
                                            const GgufShape& down, LinearPolicy gate_policy,
                                            LinearPolicy down_policy, std::int32_t min_tokens,
                                            std::int32_t max_tokens, bool x_activation) {
    const std::int32_t rows = up == nullptr ? gate.rows / 2 : gate.rows;
    if (down.k != rows) { throw std::invalid_argument("gguf swiglu add workspace: down is not [N, M]"); }
    const std::size_t activation = std::size_t(rows) * max_tokens * sizeof(__nv_bfloat16) + 256;
    const std::size_t gated =
        gguf_swiglu_workspace_bytes(gate, up, gate_policy, min_tokens, max_tokens, x_activation);
    // Composed: gguf_swiglu, then gguf_linear_add over the activation.
    std::size_t bytes = std::max(
        gated, gguf_project_workspace_bytes({&down, 1}, down_policy, min_tokens, max_tokens));
    // Fused (sizes grow with the width, so the widest fused width bounds every narrower one).
    const QType up_type = up == nullptr ? gate.qtype : up->qtype;
    if (fused_swiglu_add(gate.qtype, up_type, rows, gate.k, down.qtype, max_tokens, gate_policy,
                         down_policy)) {
        const std::size_t scratch = gguf::lt_planned_scratch_bytes(rows, down.rows, max_tokens,
                                                                   false, gguf::LtForm::Nvfp4);
        bytes = std::max(bytes, gguf::lt_activation_bytes(rows, max_tokens, gguf::LtForm::Nvfp4) +
                                    256 + std::max(gated, scratch + 256));
    }
    return activation + bytes;
}

void gguf_max_abs(const Weight& w, unsigned* amax, cudaStream_t stream) {
    require_gguf(w, "gguf max abs");
    if (amax == nullptr) { throw std::invalid_argument("gguf max abs: null result word"); }
    const auto type = gguf_type(w.qtype);
    if (!gguf::has_lt_product(type, w.k, gguf::LtForm::Nvfp4)) {
        if (cudaMemsetAsync(amax, 0, sizeof(unsigned), stream) != cudaSuccess) {
            throw std::runtime_error("gguf max abs: reset failed");
        }
        return;
    }
    gguf::matrix_amax(type, w.qdata, row_bytes(w), w.n, w.k, amax, stream);
}

std::size_t gguf_fp4_choice_bytes(const Weight& w, LinearPolicy policy) {
    if (!allows_a4(policy) || !is_gguf(w.qtype) || w.layout != QuantLayout::GgufBlocks ||
        w.qdata == nullptr || w.n <= 0 || w.k <= 0 ||
        !gguf::has_lt_product(gguf_type(w.qtype), w.k, gguf::LtForm::Nvfp4)) {
        return 0;
    }
    return gguf::fp4_choice_bytes(w.n, w.k);
}

void gguf_fp4_choices(const Weight& w, float amax, std::uint8_t* table, cudaStream_t stream) {
    require_gguf(w, "gguf NVFP4 choice table");
    if (!gguf::has_lt_product(gguf_type(w.qtype), w.k, gguf::LtForm::Nvfp4)) {
        throw std::invalid_argument("gguf NVFP4 choice table: the matrix has no NVFP4 form");
    }
    gguf::fp4_choices(gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, w.k, amax, table, stream);
}

void gguf_embedding(const Weight& table, const Tensor& ids, Tensor& out, cudaStream_t stream) {
    require_gguf(table, "gguf embedding");
    if (ids.dtype != DType::I32 || !ids.is_contiguous() || out.dtype != DType::BF16 ||
        !out.is_contiguous() || out.ne[0] != table.k || out.numel() != std::int64_t(table.k) * ids.numel()) {
        throw std::invalid_argument("gguf embedding: expected I32 ids and BF16 [K, ids]");
    }
    gguf::dequantize_rows(gguf_type(table.qtype), table.qdata, row_bytes(table), table.k,
                          static_cast<const std::int32_t*>(ids.data),
                          static_cast<int>(ids.numel()), static_cast<__nv_bfloat16*>(out.data),
                          table.k, stream);
}

} // namespace ninfer::ops::detail
