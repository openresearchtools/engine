#include "sortformer-schema.h"

#include "gguf.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace llama::realtime {

namespace {

void add_linear_pair(std::vector<std::string> & out, const std::string & prefix) {
    out.push_back(prefix + ".b");
    out.push_back(prefix + ".w");
}

void add_norm_pair(std::vector<std::string> & out, const std::string & prefix) {
    out.push_back(prefix + ".b");
    out.push_back(prefix + ".w");
}

} // namespace

std::vector<std::string> sortformer_expected_tensor_names(const sortformer_model_metadata & meta) {
    std::vector<std::string> out;
    out.reserve(1024);

    out.push_back("prep.feat.fb");
    out.push_back("prep.feat.win");

    for (const int idx : {0, 2, 3, 5, 6}) {
        add_linear_pair(out, "enc.pre.conv." + std::to_string(idx));
    }
    add_linear_pair(out, "enc.pre.out");

    for (uint32_t i = 0; i < meta.encoder_layers; ++i) {
        const std::string base = "enc.l" + std::to_string(i);
        out.push_back(base + ".conv.bn.b");
        out.push_back(base + ".conv.bn.rm");
        out.push_back(base + ".conv.bn.rv");
        out.push_back(base + ".conv.bn.sc");
        out.push_back(base + ".conv.bn.sh");
        out.push_back(base + ".conv.bn.w");
        add_linear_pair(out, base + ".conv.dw");
        add_linear_pair(out, base + ".conv.pw1");
        add_linear_pair(out, base + ".conv.pw2");
        add_linear_pair(out, base + ".ff1.l1");
        add_linear_pair(out, base + ".ff1.l2");
        add_linear_pair(out, base + ".ff2.l1");
        add_linear_pair(out, base + ".ff2.l2");
        add_norm_pair(out, base + ".nc");
        add_norm_pair(out, base + ".nff1");
        add_norm_pair(out, base + ".nff2");
        add_norm_pair(out, base + ".no");
        add_norm_pair(out, base + ".nsa");
        add_linear_pair(out, base + ".att.k");
        add_linear_pair(out, base + ".att.o");
        out.push_back(base + ".att.p.w");
        add_linear_pair(out, base + ".att.q");
        add_linear_pair(out, base + ".att.v");
        out.push_back(base + ".att.pbu");
        out.push_back(base + ".att.pbv");
    }

    add_linear_pair(out, "mods.ep");
    add_linear_pair(out, "mods.fh2h");
    add_linear_pair(out, "mods.h2s");
    add_linear_pair(out, "mods.sh2s");

    for (uint32_t i = 0; i < meta.transformer_layers; ++i) {
        const std::string base = "te.l" + std::to_string(i);
        add_linear_pair(out, base + ".sa.k");
        add_linear_pair(out, base + ".sa.o");
        add_linear_pair(out, base + ".sa.q");
        add_linear_pair(out, base + ".sa.v");
        add_norm_pair(out, base + ".ln1");
        add_norm_pair(out, base + ".ln2");
        add_linear_pair(out, base + ".ff.di");
        add_linear_pair(out, base + ".ff.do");
    }

    return out;
}

sortformer_tensor_validation validate_sortformer_gguf_tensors(const std::string & path, const sortformer_model_metadata & meta) {
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to open Sortformer GGUF for validation: " + path);
    }

    sortformer_tensor_validation result;
    try {
        result.actual_tensor_count = gguf_get_n_tensors(ctx);
        const auto expected = sortformer_expected_tensor_names(meta);
        result.expected_tensor_count = expected.size();

        std::set<std::string> expected_set(expected.begin(), expected.end());
        std::set<std::string> actual_set;

        for (int64_t i = 0; i < result.actual_tensor_count; ++i) {
            actual_set.insert(gguf_get_tensor_name(ctx, i));
        }

        for (const auto & name : expected_set) {
            if (actual_set.find(name) == actual_set.end()) {
                result.missing.push_back(name);
            }
        }

        for (const auto & name : actual_set) {
            if (expected_set.find(name) == expected_set.end()) {
                result.unexpected.push_back(name);
            }
        }
    } catch (...) {
        gguf_free(ctx);
        throw;
    }

    gguf_free(ctx);
    return result;
}

std::string sortformer_validation_summary(const sortformer_tensor_validation & validation) {
    std::ostringstream oss;
    oss
        << "actual_tensor_count=" << validation.actual_tensor_count
        << ", expected_tensor_count=" << validation.expected_tensor_count
        << ", missing=" << validation.missing.size()
        << ", unexpected=" << validation.unexpected.size();
    return oss.str();
}

} // namespace llama::realtime
