#pragma once

#include "gp_faco/fixed_faco_gpu.hpp"
#include "gp_faco/profiling.hpp"
#include "gp_faco/sparse_graph.hpp"

#include <functional>
#include <optional>

namespace gp_faco {

// 图身份和枚举行是独立输入；共同初解必须由同一CPU准备流程重建。
struct CandidateGraphSpec {
    std::vector<Node> common_initial_tour;
    std::vector<Edge> edges;
    CandidateRows primary, backup, ls;
    bool operator==(const CandidateGraphSpec& other) const {
        return common_initial_tour == other.common_initial_tour && edges == other.edges &&
            primary == other.primary && backup == other.backup && ls == other.ls;
    }
};

struct PreparedProblem {
    std::vector<double> coordinates;
    FixedFacoSettings settings;
    CandidateRows primary, backup, ls;
    std::vector<double> local_scale;
    double scale_epsilon = 1e-12;
    std::vector<Node> cheap_tour, initial_tour;
    double cheap_cost = 0, initial_cost = 0;
    double cheap_seconds = 0, preparation_seconds = 0;
    PreparationProfile preparation_profile;
    std::optional<CandidateGraphSpec> graph_spec;
    std::optional<SparseUndirectedGraph> graph;
    double graph_preparation_seconds = 0;
    bool ready = false;
    Node size() const { return coordinates.size() / 2; }
    double distance(Node a, Node b) const;
};

FixedFacoSettings normalized_settings(FixedFacoSettings settings, Node n);
PreparedProblem make_cheap_problem(std::vector<double> coordinates, FixedFacoSettings settings);
// false表示在准备阶段收到截止请求，部分结果不能发布为ready。
bool prepare_problem(PreparedProblem& problem, const std::function<bool()>& stop_requested = {});
// 在普通无标签准备完成后应用固定图；不以图先验重新优化共同初始tour。
void apply_candidate_graph(PreparedProblem& problem, CandidateGraphSpec spec);
std::vector<Node> flattened(const CandidateRows& rows);

}  // namespace gp_faco
