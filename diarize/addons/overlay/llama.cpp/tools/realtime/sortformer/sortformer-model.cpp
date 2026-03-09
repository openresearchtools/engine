#include "sortformer-model.h"

#include "gguf.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama::realtime {

namespace {

struct source_gguf_tensors {
    gguf_context * gguf = nullptr;
    ggml_context * ctx = nullptr;

    ~source_gguf_tensors() {
        if (gguf != nullptr) {
            gguf_free(gguf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct pending_tensor_copy {
    ggml_tensor * dst = nullptr;
    const void * src = nullptr;
    size_t nbytes = 0;
};

source_gguf_tensors load_source_tensors(const std::string & path) {
    source_gguf_tensors src;
    gguf_init_params params = {};
    params.no_alloc = false;
    params.ctx = &src.ctx;
    src.gguf = gguf_init_from_file(path.c_str(), params);
    if (src.gguf == nullptr || src.ctx == nullptr) {
        throw std::runtime_error("failed to load Sortformer GGUF tensor data: " + path);
    }
    return src;
}

ggml_init_params make_tensor_ctx_params() {
    ggml_init_params params = {};
    params.mem_size = 64ull * 1024ull * 1024ull;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    return params;
}

void configure_safe_vulkan_sortformer_mode(const std::string & backend_name) {
    if (backend_name.rfind("Vulkan", 0) != 0) {
        return;
    }

    struct env_pair {
        const char * key;
        const char * value;
    };
    const env_pair required[] = {
        {"GGML_VK_DISABLE_COOPMAT", "1"},
        {"GGML_VK_DISABLE_COOPMAT2", "1"},
        {"GGML_VK_DISABLE_F16", "1"},
    };

    for (const auto & item : required) {
        if (std::getenv(item.key) == nullptr) {
#if defined(_WIN32)
            _putenv_s(item.key, item.value);
#else
            setenv(item.key, item.value, 0);
#endif
        }
    }
}

} // namespace

sortformer_model::~sortformer_model() {
    release();
}

sortformer_model::sortformer_model(sortformer_model && other) noexcept {
    *this = std::move(other);
}

sortformer_model & sortformer_model::operator=(sortformer_model && other) noexcept {
    if (this != &other) {
        release();
        meta_ = std::move(other.meta_);
        backend_ = other.backend_;
        tensor_ctx_ = other.tensor_ctx_;
        tensor_buf_ = other.tensor_buf_;
        tensors_ = std::move(other.tensors_);

        other.backend_ = nullptr;
        other.tensor_ctx_ = nullptr;
        other.tensor_buf_ = nullptr;
        other.tensors_.clear();
    }
    return *this;
}

void sortformer_model::release() {
    tensors_.clear();
    if (tensor_buf_ != nullptr) {
        ggml_backend_buffer_free(tensor_buf_);
        tensor_buf_ = nullptr;
    }
    if (tensor_ctx_ != nullptr) {
        ggml_free(tensor_ctx_);
        tensor_ctx_ = nullptr;
    }
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
}

sortformer_model sortformer_model::load_from_gguf(const std::string & path, const std::string & backend_name) {
    sortformer_model model;
    model.meta_ = load_sortformer_gguf(path);

    configure_safe_vulkan_sortformer_mode(backend_name);
    model.backend_ = ggml_backend_init_by_name(backend_name.c_str(), nullptr);
    if (model.backend_ == nullptr) {
        throw std::runtime_error("failed to initialize backend: " + backend_name);
    }

    source_gguf_tensors src = load_source_tensors(path);
    model.tensor_ctx_ = ggml_init(make_tensor_ctx_params());
    if (model.tensor_ctx_ == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer tensor context");
    }

    std::vector<pending_tensor_copy> pending;
    pending.reserve((size_t) gguf_get_n_tensors(src.gguf));

    const int64_t n_tensors = gguf_get_n_tensors(src.gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(src.gguf, i);
        ggml_tensor * src_tensor = ggml_get_tensor(src.ctx, name);
        if (src_tensor == nullptr) {
            throw std::runtime_error(std::string("source GGUF tensor missing from context: ") + name);
        }

        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
        for (int d = 0; d < ggml_n_dims(src_tensor); ++d) {
            ne[d] = src_tensor->ne[d];
        }

        ggml_tensor * dst_tensor = ggml_new_tensor(model.tensor_ctx_, src_tensor->type, ggml_n_dims(src_tensor), ne);
        if (dst_tensor == nullptr) {
            throw std::runtime_error(std::string("failed to create Sortformer tensor: ") + name);
        }

        ggml_set_name(dst_tensor, name);
        model.tensors_.emplace(name, dst_tensor);
        pending.push_back({dst_tensor, src_tensor->data, ggml_nbytes(src_tensor)});
    }

    model.tensor_buf_ = ggml_backend_alloc_ctx_tensors(model.tensor_ctx_, model.backend_);
    if (model.tensor_buf_ == nullptr) {
        throw std::runtime_error("failed to allocate Sortformer tensors on backend");
    }

    for (const auto & item : pending) {
        ggml_backend_tensor_set(item.dst, item.src, 0, item.nbytes);
    }

    return model;
}

ggml_tensor * sortformer_model::tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("missing Sortformer tensor: " + name);
    }
    return it->second;
}

bool sortformer_model::has_tensor(const std::string & name) const {
    return tensors_.find(name) != tensors_.end();
}

sortformer_loaded_model::sortformer_loaded_model(sortformer_model && model)
    : model_(std::move(model)) {
}

std::shared_ptr<sortformer_loaded_model> sortformer_loaded_model::load_from_gguf(const std::string & path, const std::string & backend_name) {
    return std::shared_ptr<sortformer_loaded_model>(
        new sortformer_loaded_model(sortformer_model::load_from_gguf(path, backend_name)));
}

const sortformer_model & sortformer_loaded_model::model() const {
    return model_;
}

std::mutex & sortformer_loaded_model::mutex() const {
    return mutex_;
}

} // namespace llama::realtime
