#pragma once

#include "gp_faco/fixed_faco_gpu.hpp"

#include <functional>

namespace gp_faco {

struct PreparedProblem {
    std::vector<double> coordinates;
    FixedFacoSettings settings;
    CandidateRows primary, backup, ls;
    std::vector<double> local_scale;
    double scale_epsilon = 1e-12;
    std::vector<Node> cheap_tour, initial_tour;
    double cheap_cost = 0, initial_cost = 0;
    double cheap_seconds = 0, preparation_seconds = 0;
    bool ready = false;
    Node size() const { return coordinates.size() / 2; }
    double distance(Node a, Node b) const;
};

FixedFacoSettings normalized_settings(FixedFacoSettings settings, Node n);
PreparedProblem make_cheap_problem(std::vector<double> coordinates, FixedFacoSettings settings);
// false表示在准备阶段收到截止请求，部分结果不能发布为ready。
bool prepare_problem(PreparedProblem& problem, const std::function<bool()>& stop_requested = {});
std::vector<Node> flattened(const CandidateRows& rows);

}  // namespace gp_faco
