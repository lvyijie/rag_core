#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <malloc.h>
#endif
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

namespace vrag {

class Arena {
public:
    static constexpr std::size_t DefaultAlignment = 64;

    explicit Arena(std::size_t initial_capacity = 64ULL * 1024 * 1024)
        : capacity_( align_up(std::max<std::size_t>(initial_capacity, 4096), DefaultAlignment) ),
          offset_(0) {
        buffer_ = alloc_aligned(capacity_, DefaultAlignment);
        if (!buffer_) {
            throw std::bad_alloc();
        }
    }

    ~Arena() noexcept {
        free_aligned(buffer_);
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    Arena(Arena&& other) noexcept
        : buffer_(std::exchange(other.buffer_, nullptr)),
          capacity_(std::exchange(other.capacity_, 0)),
          offset_(std::exchange(other.offset_, 0)) {}

    Arena& operator=(Arena&& other) noexcept {
        if (this != &other) {
            free_aligned(buffer_);
            buffer_ = std::exchange(other.buffer_, nullptr);
            capacity_ = std::exchange(other.capacity_, 0);
            offset_ = std::exchange(other.offset_, 0);
        }
        return *this;
    }

    // 确保容量充足；若不足则按 2 倍几何级数平滑扩容并保持连续物理内存布局
    void reserve(std::size_t required_bytes) {
        if (required_bytes <= capacity_) {
            return;
        }
        std::size_t new_cap = capacity_ ? capacity_ : 4096;
        while (new_cap < required_bytes) {
            new_cap *= 2;
        }
        new_cap = align_up(new_cap, DefaultAlignment);

        std::byte* new_buf = alloc_aligned(new_cap, DefaultAlignment);
        if (!new_buf) {
            throw std::bad_alloc();
        }
        if (offset_ > 0 && buffer_) {
            std::memcpy(new_buf, buffer_, offset_);
        }
        free_aligned(buffer_);
        buffer_ = new_buf;
        capacity_ = new_cap;
    }

    template <typename T>
    T* allocate(std::size_t count = 1, std::size_t alignment = alignof(T)) {
        const std::size_t bytes = sizeof(T) * count;
        const std::size_t aligned_offset = align_up(offset_, alignment);

        if (aligned_offset + bytes > capacity_) {
            reserve(aligned_offset + bytes);
        }

        T* ptr = reinterpret_cast<T*>(buffer_ + aligned_offset);
        offset_ = aligned_offset + bytes;
        return ptr;
    }

    std::span<float> allocate_vector(std::size_t dimension) {
        float* ptr = allocate<float>(dimension, 32);
        return std::span<float>(ptr, dimension);
    }

    // 直接将序列化字节流装入 Arena（用于快照极速加载）
    void resize_raw(std::size_t exact_bytes) {
        reserve(exact_bytes);
        offset_ = exact_bytes;
    }

    void reset() noexcept {
        offset_ = 0;
    }

    [[nodiscard]] std::size_t used_bytes() const noexcept { return offset_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] const std::byte* data() const noexcept { return buffer_; }
    [[nodiscard]] std::byte* data() noexcept { return buffer_; }

private:
    static constexpr std::size_t align_up(std::size_t val, std::size_t alignment) noexcept {
        return (val + alignment - 1) & ~(alignment - 1);
    }

    static std::byte* alloc_aligned(std::size_t size, std::size_t alignment) noexcept {
#if defined(_WIN32)
        return static_cast<std::byte*>(_aligned_malloc(size, alignment));
#else
        return static_cast<std::byte*>(std::aligned_alloc(alignment, size));
#endif
    }

    static void free_aligned(std::byte* ptr) noexcept {
        if (!ptr) return;
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }

    std::byte* buffer_{nullptr};
    std::size_t capacity_{0};
    std::size_t offset_{0};
};

} // namespace vrag
