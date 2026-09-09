// 完整Hard Engine：所有蚂蚁/参考/档案逐边核验，含GP、基线、重启和完整图等价。
#include "gp_faco/batch_engine.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace gp_faco;
namespace {
std::uint64_t verified_tours = 0, verified_edges = 0, restarts = 0, paired_batches = 0;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
PreparedProblem problem(Node n, FixedFacoSettings settings, Node variant, bool degenerate) {
    std::vector<double> xy(n*2,0);
    if (degenerate) xy[0] = 1;
    else for (Node i = 0; i < n; ++i) {
        xy[i*2] = ((i*7919+variant*313)%104729) / 104729.0;
        xy[i*2+1] = ((i*i*13+variant*101+17)%130363) / 130363.0;
    }
    auto p = make_cheap_problem(xy, settings); prepare_problem(p); return p;
}
CandidateGraphSpec graph_spec(const PreparedProblem& p, Node kind) {
    const Node n = p.size();
    std::set<Edge> edges;
    const auto add = [&](Node a, Node b) { edges.emplace(std::min(a,b),std::max(a,b)); };
    for (Node i = 0; i<n; ++i) add(p.initial_tour[i],p.initial_tour[(i+1)%n]);
    if (kind == 0) {
        for (Node a=0;a<n;++a) for (Node b=a+1;b<n;++b) add(a,b);
    } else if (kind == 1) {
        for (Node a=0;a<n;++a) for (Node j=0;j<std::min<Node>(8,p.primary[a].size());++j) add(a,p.primary[a][j]);
    }
    CandidateGraphSpec spec;
    spec.common_initial_tour = p.initial_tour; spec.edges.assign(edges.begin(),edges.end());
    if (kind == 0) { spec.primary=p.primary; spec.backup=p.backup; spec.ls=p.ls; return spec; }
    for (Node a=0;a<n;++a) {
        std::vector<Node> row, backup;
        for (Node b=0;b<n;++b) if (b!=a && edges.count({std::min(a,b),std::max(a,b)})) row.push_back(b);
        std::sort(row.begin(),row.end(),[&](Node x, Node y) {
            return std::make_pair(p.distance(a,x),x)<std::make_pair(p.distance(a,y),y);
        });
        auto primary=row, ls=row;
        primary.resize(p.settings.primary_width,n); ls.resize(p.settings.ls_width,n);
        for (Node b=0;b<n;++b) if (b!=a &&
            std::find(primary.begin(),primary.end(),b)==primary.end()) backup.push_back(b);
        backup.resize(p.settings.backup_width,n);
        spec.primary.push_back(primary); spec.backup.push_back(backup); spec.ls.push_back(ls);
    }
    return spec;
}
void verify_tour(const std::vector<Node>& values, std::size_t offset, const PreparedProblem& p) {
    const Node n=p.size();
    std::vector<Node> tour(values.begin()+offset,values.begin()+offset+n);
    check(p.graph->contains_tour(tour),"Hard tour or archive left E0");
    CpuTour valid(tour,[&](Node a,Node b){return p.distance(a,b);});
    check(std::isfinite(valid.recomputed_cost()),"nonfinite Hard tour");
    ++verified_tours; verified_edges+=n;
}
void verify(const BatchEvaluation& result, const std::vector<BatchTask>& tasks,
            const std::vector<PreparedProblem>& problems, Node ants, std::uint64_t fe) {
    check(result.constraint_mode==ConstraintMode::Hard && result.count_limited &&
        result.total_tour_evaluations==fe*tasks.size() && result.completed_batches==fe/ants &&
        result.discarded_batches==0 && result.overrun_seconds==0 && result.charged_seconds==0,
        "Hard FE budget or constraint identity wrong");
    for (Node c=0;c<tasks.size();++c) {
        const auto& p=problems[tasks[c].instance_key-1];
        check(result.graph_edges_per_colony[c]==p.graph->edges(),"batched CSR identity wrong");
        verify_tour(result.incumbents[c].tour,0,p);
    }
    for (const auto& trace:result.control_trace) {
        for (Node c=0;c<tasks.size();++c) {
            const auto& p=problems[tasks[c].instance_key-1]; const auto n=p.size();
            restarts+=trace.actions[c]>=16;
            for (const auto* snapshot:{&trace.before,&trace.after_restart,&trace.after_batch}) {
                for (const auto* values:{&snapshot->parent,&snapshot->epoch,&snapshot->global})
                    verify_tour(*values,static_cast<std::size_t>(c)*n,p);
                for (Node a=0;a<snapshot->controls[c].archive_size;++a)
                    verify_tour(snapshot->archive,(static_cast<std::size_t>(c)*archive_capacity+a)*n,p);
            }
            for (Node a=0;a<ants;++a)
                verify_tour(trace.after_batch.tours,(static_cast<std::size_t>(c)*ants+a)*n,p);
        }
    }
}
void same(const BatchEvaluation& a,const BatchEvaluation& b) {
    check(a.completed_construction_steps==b.completed_construction_steps &&
        a.completed_ls_evaluations==b.completed_ls_evaluations,"Hard replay work mismatch");
    for (std::size_t c=0;c<a.incumbents.size();++c)
        check(a.incumbents[c].tour==b.incumbents[c].tour && a.incumbents[c].cost==b.incumbents[c].cost,
              "Hard replay incumbent mismatch");
    check(a.control_trace.size()==b.control_trace.size(),"trace shape mismatch");
    for (std::size_t i=0;i<a.control_trace.size();++i) {
        const auto& x=a.control_trace[i]; const auto& y=b.control_trace[i];
        check(x.actions==y.actions && x.features==y.features && x.masks==y.masks &&
            x.after_batch.tours==y.after_batch.tours && x.after_batch.archive==y.after_batch.archive &&
            x.after_batch.trails==y.after_batch.trails && x.after_batch.products==y.after_batch.products,
            "Hard complete graph/replay trajectory differs");
        for (Node c=0;c<x.after_batch.colonies.size();++c)
            check(x.after_batch.colonies[c].source_uniform==y.after_batch.colonies[c].source_uniform,
                  "Hard changed source random stream");
        ++paired_batches;
    }
}
}
int main(int argc,char**argv) {
    try {
        FixedFacoSettings settings; settings.ants=8;
        Program program; program.feature_spec_id=2; program.length=1; program.operand[0]=4;
        BaselinePolicy baseline; baseline.mne_level=baseline.max_mne_level=3;
        baseline.restart_mode=StaticRestart::Bernoulli; baseline.restart_probability=1;
        BatchDiagnosticControls diagnostic; diagnostic.capture_control=true;
        // 完整图必须逐批等价于普通底座；等成本多样tour稳定触发实际重启。
        auto common=problem(31,settings,0,true); const auto full=graph_spec(common,0);
        auto constrained=common; apply_candidate_graph(constrained,full);
        FacoBatchEngine ordinary(31,4,settings), hard(31,4,settings,ConstraintMode::Hard);
        ordinary.register_problem(1,common.coordinates); hard.register_graph_problem(1,common.coordinates,full);
        const std::vector<BatchTask> tasks{{1,17},{1,29},{1,41},{1,53}};
        const auto reference=ordinary.evaluate_baseline_evaluations(tasks,64,baseline,PreparationMode::CachedCharged,UINT32_MAX,diagnostic);
        const auto actual=hard.evaluate_baseline_evaluations(tasks,64,baseline,PreparationMode::CachedCharged,UINT32_MAX,diagnostic);
        same(reference,actual); verify(actual,tasks,{constrained},settings.ants,64);
        check(restarts>0,"Hard complete graph did not exercise actual restart");
        hard.evaluate_program_evaluations(tasks,32,program,PreparationMode::CachedCharged);
        auto delayed=diagnostic; delayed.delay_batch=1; delayed.completion_delay_ms=50;
        delayed.profile=true;
        const auto replay=hard.evaluate_baseline_evaluations(tasks,64,baseline,PreparationMode::EndToEnd,UINT32_MAX,delayed);
        same(actual,replay); verify(replay,tasks,{constrained},settings.ants,64);
        bool rejected=false;
        try { hard.evaluate_program(tasks,1,Program{},PreparationMode::CachedCharged); }
        catch (const std::invalid_argument&) {rejected=true;}
        check(rejected,"Hard accepted old wall-clock entrypoint");
        // 同一批含两张不同CSR；图身份不能随别的colony或控制器泄漏。
        for (Node n:{31u,500u,1000u}) {
            std::vector<PreparedProblem> problems;
            FacoBatchEngine engine(n,4,settings,ConstraintMode::Hard);
            for (Node key=1;key<=2;++key) {
                auto p=problem(n,settings,key,false); const auto spec=graph_spec(p,key);
                engine.register_graph_problem(key,p.coordinates,spec); apply_candidate_graph(p,spec);
                problems.push_back(std::move(p));
            }
            const std::vector<BatchTask> panel{{1,17},{2,17},{1,29},{2,29}};
            const auto result=engine.evaluate_program_evaluations(panel,32,program,PreparationMode::CachedCharged,UINT32_MAX,diagnostic);
            verify(result,panel,problems,settings.ants,32);
            engine.evaluate_baseline_evaluations(panel,16,baseline,PreparationMode::CachedCharged);
            const auto repeated=engine.evaluate_program_evaluations(panel,32,program,PreparationMode::EndToEnd,UINT32_MAX,diagnostic);
            same(result,repeated); verify(repeated,panel,problems,settings.ants,32);
            const auto zero=engine.evaluate_program_evaluations(panel,0,program,PreparationMode::CachedCharged);
            verify(zero,panel,problems,settings.ants,0);
            for (Node c=0;c<panel.size();++c)
                check(zero.incumbents[c].tour==problems[panel[c].instance_key-1].initial_tour,"Hard zero FE changed common initial");
        }
        if (argc==2) { std::ofstream out(argv[1]); out<<"{\"status\":\"passed\",\"verified_tours\":"<<verified_tours
            <<",\"verified_edges\":"<<verified_edges<<",\"restart_checks\":"<<restarts
            <<",\"paired_batches\":"<<paired_batches<<"}\n"; }
        std::cout<<verified_tours<<" complete Hard tours checked\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
