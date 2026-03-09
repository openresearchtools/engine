#pragma once

#include "sortformer-gguf.h"

#include <cstddef>
#include <string>
#include <vector>

struct gguf_context;

namespace llama::realtime {

struct sortformer_tensor_validation {
    int64_t actual_tensor_count = 0;
    size_t expected_tensor_count = 0;
    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
};

std::vector<std::string> sortformer_expected_tensor_names(const sortformer_model_metadata & meta);
sortformer_tensor_validation validate_sortformer_gguf_tensors(const std::string & path, const sortformer_model_metadata & meta);
std::string sortformer_validation_summary(const sortformer_tensor_validation & validation);

} // namespace llama::realtime
