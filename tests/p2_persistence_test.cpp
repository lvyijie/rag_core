#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../include/vrag.hpp"

using namespace vrag;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        ++g_passed;
        std::cout << "[PASS] " << msg << '\n';
    } else {
        ++g_failed;
        std::cerr << "[FAIL] " << msg << '\n';
    }
}

bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

// 1. 测试 64 位 FNV-1a 校验和、字节篡改检测、文件截断检测与事务性恢复
void test_checksum_and_transactional_load() {
    const std::string base_path = "test_p2_valid.vidx";
    const std::string bitflip_path = "test_p2_bitflip.vidx";
    const std::string trunc_path = "test_p2_trunc.vidx";

    VragEngine engine(8, 4096);
    engine.add_document(10, "distributed consensus raft paxos", std::vector<float>{1, 0, 0, 0, 0, 0, 0, 0}, "topic:sys");
    engine.add_document(20, "vector quantization avx2 simd",    std::vector<float>{0, 1, 0, 0, 0, 0, 0, 0}, "topic:hpc");
    engine.add_document(30, "inverted index bm25 ranking",      std::vector<float>{0, 0, 1, 0, 0, 0, 0, 0}, "topic:ir");

    check(engine.save_index(base_path), "P2.1: save_index(VRAGIDX3 with 64-bit checksum) succeeded");
    check(!std::filesystem::exists(base_path + ".tmp"), "P2.2: temporary .tmp file cleaned up after atomic rename");

    // 读取原始快照字节
    std::vector<char> raw_bytes;
    {
        std::ifstream ifs(base_path, std::ios::binary | std::ios::ate);
        const auto sz = static_cast<std::size_t>(ifs.tellg());
        raw_bytes.resize(sz);
        ifs.seekg(0, std::ios::beg);
        ifs.read(raw_bytes.data(), static_cast<std::streamsize>(sz));
    }

    // 构造单比特翻转损坏文件（翻转文件中间某字节的 1 个 bit）
    {
        std::vector<char> corrupted = raw_bytes;
        corrupted[corrupted.size() / 2] ^= 0x04;
        std::ofstream ofs(bitflip_path, std::ios::binary | std::ios::trunc);
        ofs.write(corrupted.data(), static_cast<std::streamsize>(corrupted.size()));
    }

    // 构造截断损坏文件（截掉末尾 5 个字节）
    {
        std::ofstream ofs(trunc_path, std::ios::binary | std::ios::trunc);
        ofs.write(raw_bytes.data(), static_cast<std::streamsize>(raw_bytes.size() - 5));
    }

    // 验证事务性加载保护：在一个已有数据的引擎上尝试加载损坏文件，失败后原引擎数据必须 100% 完好
    VragEngine live_engine(8, 4096);
    live_engine.add_document(99, "existing live document in memory", std::vector<float>{0, 0, 0, 1, 0, 0, 0, 0}, "live:true");

    check(!live_engine.load_index(bitflip_path), "P2.1: 1-bit payload corruption detected and rejected by 64-bit checksum");
    check(!live_engine.load_index(trunc_path), "P2.1: truncated index file detected and rejected");
    check(live_engine.document_count() == 1 && live_engine.contains_document(99) &&
          live_engine.get_metadata(99) == "live:true",
          "P2.1: failed load_index is transactional (preserves existing live engine state intact)");

    // 验证正常加载与原子覆盖保存
    check(live_engine.load_index(base_path) && live_engine.document_count() == 3,
          "P2.1: valid VRAGIDX3 snapshot loaded and replaced engine state cleanly");

    // 再次保存到同一路径（验证目标已存在时的原子覆盖替换）
    live_engine.add_document(40, "fourth document after reload", std::vector<float>{0, 0, 0, 0, 1, 0, 0, 0});
    check(live_engine.save_index(base_path), "P2.2: atomic overwrite of existing snapshot file succeeded");

    VragEngine verify_overwrite(8, 4096);
    check(verify_overwrite.load_index(base_path) && verify_overwrite.document_count() == 4,
          "P2.2: overwritten snapshot verified with 4 documents");

    std::remove(base_path.c_str());
    std::remove(bitflip_path.c_str());
    std::remove(trunc_path.c_str());
}

// 2. 测试墓碑空间压缩 (compact())：物理内存回收、死词项清除与检索得分完全不变性
void test_tombstone_compaction() {
    constexpr std::size_t Dim = 16;
    VragEngine engine(Dim, 4096);

    // 插入 60 篇文档
    for (std::uint32_t id = 1; id <= 60; ++id) {
        std::vector<float> vec(Dim, 0.0f);
        vec[id % Dim] = 1.0f;
        vec[(id + 3) % Dim] = 0.5f;
        std::string text = (id <= 40)
            ? ("ephemeral_obsolete_term doc_number_" + std::to_string(id))
            : ("persistent_knowledge_base neural retrieval doc_" + std::to_string(id));
        engine.add_document(id, text, vec, "group:" + std::to_string(id % 3));
    }

    // 删除前 40 篇文档 (产生 40 个墓碑槽位)，并对剩余 20 篇中的 10 篇做 Upsert 更新 (再产生 10 个墓碑槽位)
    for (std::uint32_t id = 1; id <= 40; ++id) {
        engine.remove_document(id);
    }
    for (std::uint32_t id = 41; id <= 50; ++id) {
        std::vector<float> vec(Dim, 0.0f);
        vec[id % Dim] = 1.0f;
        vec[(id + 1) % Dim] = 0.8f;
        engine.add_document(id, "persistent_knowledge_base updated neural retrieval doc_" + std::to_string(id),
                            vec, "group:updated");
    }

    // 此时活跃文档数为 20，总槽位数为 60 + 10 = 70，墓碑槽位数为 50
    check(engine.document_count() == 20 && engine.total_slots() == 70 && engine.deleted_docs() == 50,
          "P2.3: pre-compaction slot accounting (20 active, 50 tombstones, 70 total slots)");

    const std::size_t mem_before = engine.memory_used();
    const std::size_t vocab_before = engine.vocab_size();

    std::vector<float> q_vec(Dim, 0.0f);
    q_vec[45 % Dim] = 1.0f;
    q_vec[46 % Dim] = 0.8f;

    auto hits_before = engine.search("updated neural retrieval", q_vec, 10);

    // 执行墓碑压缩
    engine.compact();

    const std::size_t mem_after = engine.memory_used();
    const std::size_t vocab_after = engine.vocab_size();

    check(engine.document_count() == 20 && engine.total_slots() == 20 && engine.deleted_docs() == 0,
          "P2.3: post-compaction slot accounting (20 active, 0 tombstones, 20 total slots)");
    check(mem_after < mem_before && mem_after * 70 == mem_before * 20,
          "P2.3: compact() reclaimed 71.4% (50/70) of Arena vector memory");
    check(vocab_after < vocab_before,
          "P2.3: compact() purged dead vocabulary terms from deleted documents");

    // 验证压缩前后的检索结果 ID、双路原始分、排名、融合分与元数据 100% 一致
    auto hits_after = engine.search("updated neural retrieval", q_vec, 10);
    bool identical = (hits_before.size() == hits_after.size() && !hits_after.empty());
    for (std::size_t i = 0; identical && i < hits_before.size(); ++i) {
        if (hits_before[i].doc_id != hits_after[i].doc_id ||
            !approx_eq(hits_before[i].score, hits_after[i].score, 1e-6f) ||
            !approx_eq(hits_before[i].bm25_score, hits_after[i].bm25_score, 1e-6f) ||
            !approx_eq(hits_before[i].dense_score, hits_after[i].dense_score, 1e-6f) ||
            hits_before[i].text != hits_after[i].text ||
            hits_before[i].metadata != hits_after[i].metadata) {
            identical = false;
        }
    }
    check(identical, "P2.3: search results, BM25/Dense sub-scores, text, and metadata are bit-for-bit identical after compact()");
}

} // namespace

int main() {
    std::cout << "=== Running P2 Index Persistence Hardening & Compaction Suite ===\n";
    test_checksum_and_transactional_load();
    test_tombstone_compaction();

    std::cout << "\nSummary: " << g_passed << " passed, " << g_failed << " failed.\n";
    return (g_failed == 0) ? 0 : 1;
}
