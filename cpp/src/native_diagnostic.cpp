// 原生算法诊断封装；MIT 源码版权与来源见 provenance/THIRD_PARTY_NOTICES.md。
// 仅替换入口和结果导出，不改作者 Route、转移、LS 或信息素实现。
#define main gpfaco_unused_upstream_main
#include "faco.cpp"
#undef main

int main(int argc, char** argv) {
    try {
        auto options = parse_program_options(argc, argv);
        if (options.seed_ == 0 || options.ants_count_ == 0 || options.threads_ <= 0) {
            throw std::invalid_argument("诊断必须显式指定非零 seed/ants/threads");
        }
        omp_set_num_threads(options.threads_);
        init_random_number_generators(options.seed_);
        Timer total_timer;
        auto problem = load_tsplib_instance(options.problem_path_.c_str());
        // 不读取 best-known.json；标签只由外部 Python evaluator 使用。
        problem.compute_nn_lists(std::max(options.cand_list_size_ + options.backup_list_size_,
                                          options.ls_cand_list_size_));
        nlohmann::json report;
        ComputationsLog<nlohmann::json> log(report, std::cout);
        Timer solver_timer;
        std::unique_ptr<Solution> result;
        if (options.algorithm_ == "mfaco") {
            result = run_mfaco(problem, options, log);
        } else if (options.algorithm_ == "faco_apt") {
            result = run_faco_apt(problem, options, log);
        } else {
            throw std::invalid_argument("诊断仅允许 mfaco 或 faco_apt");
        }
        report["solver_seconds"] = solver_timer();
        if (!problem.is_route_valid(result->route_)) {
            throw std::runtime_error("原生输出不是合法 tour");
        }
        report["native_recomputed_cost"] = problem.calculate_route_length(result->route_);
        report["cost"] = result->cost_;
        report["tour"] = result->route_;
        report["total_seconds"] = total_timer();
        report["label_input"] = false;
        report["local_search_source_sha256"] = GPFACO_REFERENCE_LS_HASH;
        report["initial_route_count"] = omp_get_num_procs() / 2;
        report["scope"] = "native iteration-budget diagnostic, not wall-clock comparison";
        dump(options, report["args"]);
        const auto output = fs::path(options.results_dir_) / "diagnostic.json";
        fs::create_directories(output.parent_path());
        std::ofstream stream(output);
        if (!stream) throw std::runtime_error("无法写入诊断结果");
        stream << report.dump(2) << '\n';
        if (!stream) throw std::runtime_error("诊断结果写入失败");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
