#include "audio-ring.h"

#include <algorithm>
#include <stdexcept>

namespace llama::realtime {

audio_ring_buffer::audio_ring_buffer(size_t capacity_samples)
    : capacity_(capacity_samples), storage_(capacity_samples, 0.0f) {
    if (capacity_samples == 0) {
        throw std::invalid_argument("audio ring capacity must be > 0");
    }
}

void audio_ring_buffer::clear() {
    head_ = 0;
    size_ = 0;
    total_pushed_ = 0;
    std::fill(storage_.begin(), storage_.end(), 0.0f);
}

void audio_ring_buffer::push(const float * samples, size_t n_samples) {
    if (samples == nullptr && n_samples != 0) {
        throw std::invalid_argument("audio ring push received null samples");
    }
    for (size_t i = 0; i < n_samples; ++i) {
        const size_t write_index = (head_ + size_) % capacity_;
        storage_[write_index] = samples[i];
        if (size_ == capacity_) {
            head_ = (head_ + 1) % capacity_;
        } else {
            ++size_;
        }
    }
    total_pushed_ += n_samples;
}

size_t audio_ring_buffer::size() const {
    return size_;
}

size_t audio_ring_buffer::capacity() const {
    return capacity_;
}

uint64_t audio_ring_buffer::total_samples_pushed() const {
    return total_pushed_;
}

std::vector<float> audio_ring_buffer::snapshot() const {
    return tail(size_);
}

std::vector<float> audio_ring_buffer::tail(size_t max_samples) const {
    const size_t n = std::min(max_samples, size_);
    std::vector<float> out;
    out.reserve(n);
    const size_t start = (head_ + size_ - n) % capacity_;
    for (size_t i = 0; i < n; ++i) {
        out.push_back(storage_[(start + i) % capacity_]);
    }
    return out;
}

} // namespace llama::realtime
