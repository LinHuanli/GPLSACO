// 原始 2022 FACO：只替换入口和结果导出，算法、初始化和 LS 使用作者源码。
// 不向求解器提供标签。论文的 ceil 蚂蚁数显式覆盖源码的 lround 默认值。
#define main gpfaco_unused_original_main
#include "faco.cpp"
#undef main

int main(int argc, char** argv) {
    try {
        auto options = parse_program_options(argc, argv);
        if (!options.seed_ || options.threads_ <= 0 || options.iterations_ < 0)
            throw std::invalid_argument("需要显式非零 seed、正 threads 和非负 iterations");
        omp_set_dynamic(0);
        omp_set_num_threads(options.threads_);
        init_random_number_generators(options.seed_);
        Timer total_timer;
        auto problem = load_tsplib_instance(options.problem_path_.c_str());
#ifdef GPFACO_CONTINUOUS_ADAPTATION
        // 明确命名的连续距离适配；原始 TSPLIB 复现目标不编译此分支。
        if (problem.coords_.size() != problem.dimension_)
            throw std::invalid_argument("连续适配需要原始二维坐标");
        problem.distance_matrix_.resize(static_cast<std::size_t>(problem.dimension_) * problem.dimension_);
        for (uint32_t a = 0; a < problem.dimension_; ++a) {
            for (uint32_t b = 0; b < problem.dimension_; ++b) {
                const double x = problem.coords_[a].x_ - problem.coords_[b].x_;
                const double y = problem.coords_[a].y_ - problem.coords_[b].y_;
                problem.distance_matrix_[static_cast<std::size_t>(a) * problem.dimension_ + b] = std::sqrt(x*x+y*y);
            }
        }
        // 作者 KD-tree 内部仍使用 lround 欧氏距离；连续问题必须按同一个
        // double 距离矩阵排序 NN，并用作者的 NN-list 初始化分支。
        problem.kdtree_.reset();
#endif
        const auto paper_ants = 64u * static_cast<uint32_t>(std::ceil(std::sqrt(problem.dimension_) / 16.0));
        options.ants_count_ = paper_ants;
        problem.compute_nn_lists(std::max(options.cand_list_size_ + options.backup_list_size_,
                                         options.ls_cand_list_size_));
        nlohmann::json report;
        ComputationsLog<nlohmann::json> log(report, std::cout);
        Timer solver_timer;
        auto result = run_focused_aco(problem, options, log);
        report["solver_seconds"] = solver_timer();
        report["cost"] = result->cost_;
        report["tour"] = result->route_;
        report["total_seconds"] = total_timer();
        report["initial_route_count"] = omp_get_num_procs();
        report["threads"] = options.threads_;
        report["search_tour_evaluations"] = uint64_t(paper_ants) * options.iterations_;
        report["method"] = "original_faco_2022";
#ifdef GPFACO_CONTINUOUS_ADAPTATION
        report["method"] = "faco_2022_continuous_identity_guard";
        report["adaptation_revision"] = 2;
        report["distance_spec"] = "continuous_euclidean_fp64";
        report["candidate_distance_spec"] = "continuous_euclidean_fp64";
#endif
        report["label_input"] = false;
        dump(options, report["args"]);
        const auto output = fs::path(options.results_dir_) / "result.json";
        fs::create_directories(output.parent_path());
        std::ofstream stream(output);
        if (!stream) throw std::runtime_error("无法写入结果");
        stream << report.dump() << '\n';
        if (!stream) throw std::runtime_error("结果写入失败");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
