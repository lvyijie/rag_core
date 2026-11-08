#include <cstdint>
#include <iostream>

#include "../include/arena.hpp"

using namespace vrag;

int main() {
    // 1. 测试初始 1 MiB 内存池与 32/64 字节对齐
    Arena arena(1024 * 1024);

    auto vec1 = arena.allocate_vector(768);
    auto vec2 = arena.allocate_vector(768);
    auto vec3 = arena.allocate_vector(768);

    if (vec1.empty() || vec2.empty() || vec3.empty()) {
        std::cerr << "[FAIL] Failed to allocate vectors\n";
        return 1;
    }

    auto check_align = [](const void* ptr, std::size_t align) {
        return reinterpret_cast<std::uintptr_t>(ptr) % align == 0;
    };

    if (!check_align(vec1.data(), 32) ||
        !check_align(vec2.data(), 32) ||
        !check_align(vec3.data(), 32)) {
        std::cerr << "[FAIL] Pointer is not 32-byte aligned!\n";
        return 1;
    }

    std::cout << "[PASS] Vector allocations are strictly 32-byte aligned.\n";
    std::cout << "Used bytes: " << arena.used_bytes() << " / " << arena.capacity() << '\n';

    // 2. 测试小容量 Arena 自动平滑扩容且数据保持完整
    Arena small_arena(4096);
    int32_t* first_block = small_arena.allocate<int32_t>(512);
    for (int i = 0; i < 512; ++i) {
        first_block[i] = i + 100;
    }
    // 触发超过 4096B 的多次自动扩容
    for (int k = 0; k < 16; ++k) {
        small_arena.allocate<int32_t>(1024);
    }
    const auto* base_i32 = reinterpret_cast<const int32_t*>(small_arena.data());
    for (int i = 0; i < 512; ++i) {
        if (base_i32[i] != i + 100) {
            std::cerr << "[FAIL] Arena auto-growth corrupted existing data!\n";
            return 1;
        }
    }
    std::cout << "[PASS] Arena auto-growth preserved existing data (capacity: "
              << small_arena.capacity() << " B).\n";

    // 3. 测试 O(1) reset
    arena.reset();
    if (arena.used_bytes() != 0) {
        std::cerr << "[FAIL] Reset failed\n";
        return 1;
    }

    std::cout << "[PASS] Arena reset successfully.\n";
    return 0;
}
