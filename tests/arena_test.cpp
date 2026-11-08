#include <cstdint>
#include <iostream>

#include "../include/arena.hpp"

using namespace vrag;

int main() {
    // 分配 1 MiB 内存池
    Arena arena(1024 * 1024);

    // 1. 测试连续分配 3 个 768 维向量
    auto vec1 = arena.allocate_vector(768);
    auto vec2 = arena.allocate_vector(768);
    auto vec3 = arena.allocate_vector(768);

    if (vec1.empty() || vec2.empty() || vec3.empty()) {
        std::cerr << "[FAIL] Failed to allocate vectors\n";
        return 1;
    }

    // 2. 检查首地址是否严格 32 字节对齐（AVX2 对齐加载硬性要求）
    auto check_align = [](void* ptr, std::size_t align) {
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

    // 3. 测试 O(1) reset
    arena.reset();
    if (arena.used_bytes() != 0) {
        std::cerr << "[FAIL] Reset failed\n";
        return 1;
    }

    std::cout << "[PASS] Arena reset successfully.\n";
    return 0;
}
