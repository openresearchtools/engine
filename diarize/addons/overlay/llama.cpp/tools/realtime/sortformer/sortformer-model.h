#pragma once

#include "sortformer-gguf.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llama::realtime {

class sortformer_model {
public:
    sortformer_model() = default;
    ~sortformer_model();

    sortformer_model(const sortformer_model &) = delete;
    sortformer_model & operator=(const sortformer_model &) = delete;
    sortformer_model(sortformer_model && other) noexcept;
    sortformer_model & operator=(sortformer_model && other) noexcept;

    static sortformer_model load_from_gguf(const std::string & path, const std::string & backend_name);

    const sortformer_model_metadata & metadata() const { return meta_; }
    ggml_backend_t backend() const { return backend_; }
    ggml_tensor * tensor(const std::string & name) const;
    bool has_tensor(const std::string & name) const;

private:
    void release();

    sortformer_model_metadata meta_;
    ggml_backend_t backend_ = nullptr;
    ggml_context * tensor_ctx_ = nullptr;
    ggml_backend_buffer_t tensor_buf_ = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors_;
};

class sortformer_loaded_model {
public:
    static std::shared_ptr<sortformer_loaded_model> load_from_gguf(const std::string & path, const std::string & backend_name);

    const sortformer_model & model() const;
    std::mutex & mutex() const;

private:
    explicit sortformer_loaded_model(sortformer_model && model);

    sortformer_model model_;
    mutable std::mutex mutex_;
};

} // namespace llama::realtime
