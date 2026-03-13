#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llama::realtime {

class audio_ring_buffer {
public:
    explicit audio_ring_buffer(size_t capacity_samples = 16000 * 30);

    void clear();
    void push(const float * samples, size_t n_samples);

    size_t size() const;
    size_t capacity() const;
    uint64_t total_samples_pushed() const;

    std::vector<float> snapshot() const;
    std::vector<float> tail(size_t max_samples) const;

private:
    size_t capacity_ = 0;
    std::vector<float> storage_;
    size_t head_ = 0;
    size_t size_ = 0;
    uint64_t total_pushed_ = 0;
};

} // namespace llama::realtime
