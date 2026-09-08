#include "gp_faco/deadline.hpp"
#include "gp_faco/prepared_problem.hpp"

#include <iostream>
#include <stdexcept>

void check(bool result, const char* message) { if (!result) throw std::runtime_error(message); }

int main() {
    try {
        double now = 0;
        gp_faco::DeadlineLedger ledger(1.0, [&]() { return now; });
        gp_faco::TimedIncumbent best;
        check(best.offer({0, 1, 2, 3}, 10, ledger.elapsed(), ledger), "廉价解登记失败");
        ledger.charge(0.6); now = 0.2;
        check(best.offer({0, 2, 1, 3}, 8, ledger.elapsed(), ledger), "按时结果未提交");
        now = 0.4;
        check(best.offer({0, 2, 3, 1}, 7, ledger.elapsed(), ledger), "等于deadline的完整结果被拒绝");
        now = 0.5;
        check(!best.offer({0, 3, 2, 1}, 1, ledger.elapsed(), ledger), "迟到改进污染incumbent");
        check(best.cost == 7 && best.completed_seconds == 1.0, "返回了deadline之后的结果");
        check(!ledger.can_start() && ledger.expired(), "预算耗尽仍允许启动工作");
        now = 0.1;
        bool reversed = false;
        try { ledger.elapsed(); } catch (const std::runtime_error&) { reversed = true; }
        check(reversed, "没有拒绝倒退时钟");
        gp_faco::DeadlineLedger empty(0, []() { return 0.0; });
        check(!empty.can_start(), "零预算启动工作");
        gp_faco::DeadlineLedger resource_only(std::nullopt, []() { return 1e100; });
        check(resource_only.can_start() && !resource_only.expired() &&
              resource_only.completed_on_time(resource_only.elapsed()), "无截止入口存在隐藏秒数上限");
        gp_faco::TimedIncumbent untimed;
        check(untimed.offer({0, 1, 2, 3}, 10, resource_only.elapsed(), resource_only),
              "次数入口按耗时拒绝了合法结果");
        bool charged = false;
        try { resource_only.charge(0.1); } catch (const std::logic_error&) { charged = true; }
        check(charged, "次数入口接受了墙钟扣费");

        auto p = gp_faco::make_cheap_problem({0, 0, 1, 0, 1, 1, 0, 1}, {});
        check(!p.ready && p.cheap_cost == 4, "昂贵准备前没有正确廉价解");
        int checks = 0;
        check(!gp_faco::prepare_problem(p, [&]() { return ++checks == 3; }), "未响应准备截止");
        check(!p.ready && p.cheap_cost == 4 && p.cheap_tour.size() == 4, "部分准备破坏廉价解");
        check(gp_faco::prepare_problem(p) && p.initial_cost <= p.cheap_cost, "完整准备丢弃廉价incumbent");
        std::cout << "deadline ledger and staged preparation: passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
