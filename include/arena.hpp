#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <span>
#include <utility>
#include <stdexcept>

namespace vrag {

class Arena {
public:
    static constexpr std::size_t DefaultAlignment = 32;

    // 默认分配 512 MB（足以容纳 12 万篇 1024 维文档），也支持外部自由指定
    explicit Arena(std::size_t capacity = 512ULL * 1024 * 1024)
        : capacity_(capacity), offset_(0) {
        #if defined(_MSC_VER)
        buffer_ = static_cast<std::byte*>(_aligned_malloc(capacity_, DefaultAlignment));
        #else
        // POSIX 环境下使用 posix_memalign 或 std::aligned_alloc
        buffer_ = static_cast<std::byte*>(std::aligned_alloc(DefaultAlignment, capacity_));
        #endif

        if (!buffer_) {
            throw std::bad_alloc();
        }
    }

    ~Arena() noexcept {
        #if defined(_MSC_VER)
        _aligned_free(buffer_);
        #else
        std::free(buffer_);
        #endif
    }

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;

    Arena(Arena&& other) noexcept
        : buffer_(std::exchange(other.buffer_, nullptr)),
          capacity_(std::exchange(other.capacity_, 0)),
          offset_(std::exchange(other.offset_, 0)) {}

    Arena& operator=(Arena&& other) noexcept {
        if (this != &other) {
            #if defined(_MSC_VER)
            _aligned_free(buffer_);
            #else
            std::free(buffer_);
            #endif
            buffer_ = std::exchange(other.buffer_, nullptr);
            capacity_ = std::exchange(other.capacity_, 0);
            offset_ = std::exchange(other.offset_, 0);
        }
        return *this;
    }

    template <typename T>
    T* allocate(std::size_t count = 1, std::size_t alignment = alignof(T)) {
        std::size_t bytes = sizeof(T) * count;
        std::size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        // 如果超出容量，严禁返回 nullptr，直接抛异常阻断，避免后续触发野指针崩溃
        if (aligned_offset + bytes > capacity_) {
            throw std::bad_alloc();
        }

        T* ptr = reinterpret_cast<T*>(buffer_ + aligned_offset);
        offset_ = aligned_offset + bytes;
        return ptr;
    }

    std::span<float> allocate_vector(std::size_t dimension) {
        float* ptr = allocate<float>(dimension, DefaultAlignment);
        return std::span<float>(ptr, dimension);
    }

    void reset() noexcept {
        offset_ = 0;
    }

    [[nodiscard]] std::size_t used_bytes() const noexcept { return offset_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] const std::byte* data() const noexcept { return buffer_; }

private:
    std::byte* buffer_{nullptr};
    std::size_t capacity_{0};
    std::size_t offset_{0};
};

} // namespace vrag
