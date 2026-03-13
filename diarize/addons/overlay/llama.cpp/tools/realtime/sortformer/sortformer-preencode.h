#pragma once

#include "sortformer-model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llama::realtime {

struct sortformer_matrix_f32 {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<float> data;
};

sortformer_matrix_f32 load_matrix_f32_bin(const std::string & path);
void save_matrix_f32_bin(const std::string & path, const sortformer_matrix_f32 & matrix);

sortformer_matrix_f32 sortformer_run_preencode(
    const sortformer_model & model,
    const sortformer_matrix_f32 & features);

double sortformer_max_abs_diff(const sortformer_matrix_f32 & a, const sortformer_matrix_f32 & b);
double sortformer_rmse(const sortformer_matrix_f32 & a, const sortformer_matrix_f32 & b);

} // namespace llama::realtime
