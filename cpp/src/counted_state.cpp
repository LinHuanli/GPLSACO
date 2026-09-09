// 固定版本快照封装；长度检查先于分配，校验覆盖元数据和每份缓冲。
#include "gp_faco/counted_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace gp_faco {
namespace {
using Bytes = std::vector<std::uint8_t>;
constexpr std::uint64_t format_magic = 0x3154534f43414647ULL;
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
auto settings_tuple(const FixedFacoSettings& c) {
    return std::make_tuple(c.ants, c.primary_width, c.backup_width, c.ls_width,
        c.beta, c.retention, c.p_best, c.epoch_source_probability,
        c.ls_evaluation_limit, c.initial_ls_evaluation_limit);
}
void integer(Bytes& out, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void real(Bytes& out, double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits; std::memcpy(&bits, &value, sizeof(bits)); integer(out, bits);
}
std::uint64_t hash_bytes(const Bytes& bytes, std::size_t size) {
    std::uint64_t value = 14695981039346656037ULL;
    for (std::size_t i = 0; i < size; ++i) value = (value ^ bytes[i]) * 1099511628211ULL;
    return value;
}
Bytes payload(const CountedState& s) {
    Bytes out;
    std::size_t capacity = 256;
    for (const auto& b : s.buffers) capacity += b.bytes.size() + b.name.size() + 32;
    out.reserve(capacity);
    integer(out, format_magic); integer(out, s.dimension); integer(out, s.colonies);
    const auto& c = s.settings;
    for (auto v : {c.ants, c.primary_width, c.backup_width, c.ls_width}) integer(out, v);
    for (auto v : {c.beta, c.retention, c.p_best, c.epoch_source_probability}) real(out, v);
    integer(out, c.ls_evaluation_limit); integer(out, c.initial_ls_evaluation_limit);
    integer(out, s.experiment_mask); integer(out, s.completed_batches);
    integer(out, s.progress_evaluation_limit); integer(out, s.tasks.size());
    for (const auto& t : s.tasks) { integer(out, t.instance_key); integer(out, t.seed); }
    integer(out, s.incumbents.size());
    for (const auto& item : s.incumbents) {
        real(out, item.cost); integer(out, item.tour.size());
        for (Node node : item.tour) integer(out, node);
    }
    integer(out, s.buffers.size());
    for (const auto& b : s.buffers) {
        integer(out, b.name.size()); out.insert(out.end(), b.name.begin(), b.name.end());
        integer(out, b.element_bytes); integer(out, b.planes); integer(out, b.bytes.size());
        out.insert(out.end(), b.bytes.begin(), b.bytes.end());
    }
    return out;
}
void structure(const CountedState& s) {
    const auto& c = s.settings;
    require(s.dimension >= 3 && s.colonies >= 1 && s.colonies <= 128 &&
        c.ants >= 1 && c.ants <= 1024 && s.tasks.size() == s.colonies && s.incumbents.size() == s.colonies,
        "快照形状或任务无效");
    for (const auto& item : s.incumbents) {
        require(item.tour.size() == s.dimension && std::isfinite(item.cost) && item.cost > 0,
                "快照主机incumbent无效");
        auto nodes = item.tour; std::sort(nodes.begin(), nodes.end());
        for (Node i = 0; i < s.dimension; ++i) require(nodes[i] == i, "快照主机tour不是完整排列");
    }
    require(s.progress_evaluation_limit > 0 && s.progress_evaluation_limit % c.ants == 0 &&
        s.progress_evaluation_limit / c.ants <= UINT32_MAX &&
        s.completed_batches <= s.progress_evaluation_limit / c.ants &&
        (s.experiment_mask & 0xffffu), "快照批次、progress预算或mask无效");
    require(!s.buffers.empty() && s.buffers.size() <= 64, "快照缓冲集合无效");
    std::set<std::string> names;
    for (const auto& b : s.buffers) {
        require(!b.name.empty() && b.name.size() <= 128 && names.insert(b.name).second &&
            b.element_bytes > 0 && b.element_bytes <= 65536 && b.planes >= 1 && b.planes <= 12 &&
            !b.bytes.empty() && b.bytes.size() % (b.element_bytes * b.planes * s.colonies) == 0,
            "快照缓冲名称、ABI或分层布局无效");
    }
}
struct Reader {
    const Bytes& data;
    std::size_t cursor = 0, end;
    explicit Reader(const Bytes& bytes) : data(bytes), end(bytes.size() >= 8 ? bytes.size() - 8 : 0) {}
    std::uint64_t integer() {
        require(cursor <= end && end - cursor >= 8, "快照元数据截断");
        std::uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(data[cursor++]) << (8 * i);
        return value;
    }
    Node node() {
        const auto value = integer(); require(value <= UINT32_MAX, "快照整数溢出");
        return static_cast<Node>(value);
    }
    double real() {
        const auto bits = integer(); double value; std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    Bytes bytes(std::uint64_t size) {
        require(cursor <= end && size <= end - cursor, "快照缓冲截断");
        Bytes out(data.begin() + cursor, data.begin() + cursor + static_cast<std::size_t>(size));
        cursor += static_cast<std::size_t>(size); return out;
    }
};
}

bool same_settings(const FixedFacoSettings& a, const FixedFacoSettings& b) {
    return settings_tuple(a) == settings_tuple(b);
}
void CountedState::seal() {
    structure(*this); const auto bytes = payload(*this); checksum = hash_bytes(bytes, bytes.size());
}
void CountedState::validate() const {
    structure(*this); const auto bytes = payload(*this);
    require(checksum == hash_bytes(bytes, bytes.size()), "快照内容校验失败");
}
std::vector<std::uint8_t> CountedState::serialize() const {
    validate(); auto out = payload(*this); integer(out, checksum); return out;
}
CountedState CountedState::deserialize(const Bytes& bytes) {
    require(bytes.size() >= 16, "快照文件截断");
    std::uint64_t expected = 0;
    for (unsigned i = 0; i < 8; ++i)
        expected |= static_cast<std::uint64_t>(bytes[bytes.size() - 8 + i]) << (8 * i);
    require(expected == hash_bytes(bytes, bytes.size() - 8), "快照文件校验失败");
    Reader r(bytes); require(r.integer() == format_magic, "未知快照格式");
    CountedState s; s.dimension = r.node(); s.colonies = r.node();
    auto& c = s.settings;
    c.ants = r.node(); c.primary_width = r.node(); c.backup_width = r.node(); c.ls_width = r.node();
    c.beta = r.real(); c.retention = r.real(); c.p_best = r.real(); c.epoch_source_probability = r.real();
    c.ls_evaluation_limit = r.integer(); c.initial_ls_evaluation_limit = r.integer();
    s.experiment_mask = r.node(); s.completed_batches = r.integer(); s.progress_evaluation_limit = r.integer();
    const auto tasks = r.integer(); require(tasks <= 128, "快照任务数量越界");
    for (std::uint64_t i = 0; i < tasks; ++i) {
        const auto key = r.integer(), seed = r.integer(); s.tasks.push_back({key, seed});
    }
    const auto incumbents = r.integer(); require(incumbents <= 128, "快照主机incumbent数量越界");
    for (std::uint64_t i = 0; i < incumbents; ++i) {
        CountedStateIncumbent item; item.cost = r.real();
        const auto size = r.integer();
        require(size == s.dimension && size <= (r.end - r.cursor) / 8, "快照主机tour形状或字节数无效");
        for (std::uint64_t j = 0; j < size; ++j) item.tour.push_back(r.node());
        s.incumbents.push_back(std::move(item));
    }
    const auto buffers = r.integer(); require(buffers <= 64, "快照缓冲数量越界");
    for (std::uint64_t i = 0; i < buffers; ++i) {
        const auto length = r.integer(); require(length <= 128, "快照缓冲名称越界");
        const auto name = r.bytes(length); CountedStateBuffer buffer;
        buffer.name.assign(name.begin(), name.end()); buffer.element_bytes = r.integer();
        buffer.planes = r.integer(); const auto size = r.integer(); buffer.bytes = r.bytes(size);
        s.buffers.push_back(std::move(buffer));
    }
    require(r.cursor == r.end, "快照尾部存在额外数据"); s.checksum = expected; s.validate(); return s;
}
CountedState CountedState::select_colony(Node colony) const {
    validate(); require(colony < colonies, "快照colony越界");
    CountedState out = *this; out.colonies = 1; out.tasks = {tasks[colony]}; out.incumbents = {incumbents[colony]};
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto& source = buffers[i]; auto& target = out.buffers[i]; target.bytes.clear();
        const auto stride = source.bytes.size() / source.planes / colonies;
        for (std::uint64_t plane = 0; plane < source.planes; ++plane) {
            const auto first = source.bytes.begin() + (plane * colonies + colony) * stride;
            target.bytes.insert(target.bytes.end(), first, first + stride);
        }
    }
    out.seal(); return out;
}

}  // namespace gp_faco
