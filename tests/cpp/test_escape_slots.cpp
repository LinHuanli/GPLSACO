// 有效数量从0到20，穷举随机取整的四种余数；不以实现自身的计数作为oracle。
#include "gp_faco/escape_slots.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace gp_faco;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Draws {
    Node rounding;
    Node draw() { return rounding; }
    Node below(Node count) { check(count != 0, "对空集合抽样"); return count - 1; }
};
}
int main() {
    try {
        for (Node m = 0; m <= 20; ++m) {
            std::vector<Node> original(20, 100), backup{0, 1, 51, 51, 52, 53, 54, 55, 56, 100};
            for (Node j = 0; j < m; ++j) original[j] = j + 1;
            Node total = 0;
            for (Node rounding = 0; rounding < 4; ++rounding) {
                Draws random{rounding};
                auto row = escape_slot_view(original.data(), 20, backup.data(), backup.size(), 100, 0, random);
                std::set<Node> actual;
                Node count = 0;
                for (Node j = 0; j < 20; ++j) {
                    if (j >= m) check(row.members[j] == 100 && !row.replaced[j], "填充槽变成额外有效槽");
                    else {
                        check(row.members[j] > 0 && row.members[j] < 100 && actual.insert(row.members[j]).second,
                              "视图出现自身/重复/无效成员");
                        const bool replacement = std::find(original.begin(), original.end(), row.members[j]) == original.end();
                        check(row.replaced[j] == replacement, "逐槽来源标记不正确"); count += replacement;
                    }
                }
                total += count;
                const auto distance = [](Node, Node node) { return static_cast<double>(node % 7); };
                sort_escape_ls_row(row, 0, distance);
                for (Node j = 1; j < m; ++j) check(
                    std::make_pair(distance(0, row.members[j - 1]), row.members[j - 1]) <
                    std::make_pair(distance(0, row.members[j]), row.members[j]), "LS真实距离/node ID次序错误");
                for (Node j = 0; j < m; ++j) check(row.replaced[j] ==
                    (std::find(original.begin(), original.end(), row.members[j]) == original.end()),
                    "LS排序丢失来源标记");
                const Node insufficient[]{0, 1, 1, 100};
                row = escape_slot_view(original.data(), 20, insufficient, 4, 100, 0, random);
                check(row.replacements <= 1 && (m == 0 || row.replacements == 0), "备用不足时伪造候选");
            }
            check(total == m, "四种等概率取整的替换总量不是m，期望比例偏离1/4");
        }
        std::cout << "84 exact slot-rounding cases passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
