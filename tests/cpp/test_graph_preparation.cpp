#include "gp_faco/prepared_problem.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace gp_faco;
namespace {
unsigned checks = 0;
void check(bool condition, const char* message) {
    ++checks; if (!condition) throw std::runtime_error(message);
}
CandidateGraphSpec graph_spec(const PreparedProblem& p, bool star) {
    const Node n = p.size();
    std::set<Edge> edges;
    for (Node i = 0; i < n; ++i) {
        const Node a = p.initial_tour[i], b = p.initial_tour[(i+1)%n];
        edges.emplace(std::min(a,b), std::max(a,b));
    }
    if (star) for (Node i = 1; i < n; ++i) edges.emplace(0,i);
    CandidateGraphSpec spec;
    spec.common_initial_tour = p.initial_tour;
    spec.edges.assign(edges.begin(), edges.end());
    for (Node a = 0; a < n; ++a) {
        std::vector<Node> row, backup;
        for (Node b = 0; b < n; ++b) if (a != b &&
            edges.count({std::min(a,b),std::max(a,b)})) row.push_back(b);
        auto primary = row;
        primary.resize(p.settings.primary_width, n);
        for (Node b = 0; b < n; ++b) if (a != b &&
            std::find(primary.begin(), primary.end(), b) == primary.end()) backup.push_back(b);
        backup.resize(p.settings.backup_width, n);
        std::sort(row.begin(), row.end(), [&](Node x, Node y) {
            return std::make_pair(p.distance(a,x),x) < std::make_pair(p.distance(a,y),y);
        });
        row.resize(p.settings.ls_width, n);
        spec.primary.push_back(primary); spec.backup.push_back(backup); spec.ls.push_back(row);
    }
    return spec;
}
}
int main(int argc, char** argv) {
    try {
        for (Node n : {7u,31u}) for (bool star : {false,true}) {
            FixedFacoSettings settings;
            settings.primary_width = 4; settings.backup_width = 6; settings.ls_width = 6;
            std::vector<double> xy;
            for (Node i = 0; i < n; ++i) {
                xy.push_back(((i*17)%37) / 37.0); xy.push_back(((i*11+3)%41) / 41.0);
            }
            auto common = make_cheap_problem(xy, settings); prepare_problem(common);
            const auto spec = graph_spec(common, star);
            auto result = common; apply_candidate_graph(result, spec);
            check(result.initial_tour == common.initial_tour && result.initial_cost == common.initial_cost,
                  "graph changed common initialization");
            check(result.graph->contains_tour(common.initial_tour), "initial tour outside E0");
            check(result.graph->edges() == spec.edges.size(), "actual graph size changed");
            check(result.local_scale == common.local_scale, "graph changed scale normalization");
            check(!star || result.graph->maximum_degree() == n-1, "CSR truncated star degree");
            check(star || result.graph->maximum_degree() == 2, "cycle graph degree wrong");
            check(result.preparation_seconds >= common.preparation_seconds, "graph time missing");
            for (Node i = 0; i < n; ++i) {
                const auto view = result.graph->view();
                check(view.offsets[i+1]-view.offsets[i] >= 2, "cycle edges lost");
                for (Node j : spec.primary[i]) if (j<n) check(view(i,j), "primary outside graph");
            }
            for (int mode = 0; mode < 6; ++mode) {
                auto changed = spec;
                if (mode == 0) std::rotate(changed.common_initial_tour.begin(),
                    changed.common_initial_tour.begin()+1, changed.common_initial_tour.end());
                if (mode == 1) changed.edges.push_back(changed.edges.front());
                if (mode == 2) changed.primary[0][0] = n+1;
                if (mode == 3) changed.primary[0][0] = n;
                if (mode == 4) changed.backup[0][0] = changed.primary[0][0];
                if (mode == 5) std::swap(changed.ls[0][0], changed.ls[0][1]);
                auto invalid = common;
                bool rejected = false;
                try { apply_candidate_graph(invalid, changed); } catch (const std::invalid_argument&) { rejected=true; }
                check(rejected, "invalid graph accepted");
                check(!invalid.graph && invalid.primary == common.primary, "failed graph partially published");
            }
        }
        if (argc == 2) { std::ofstream out(argv[1]); out << "{\"checks\":" << checks << ",\"status\":\"passed\"}\n"; }
        std::cout << checks << " graph preparation checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
