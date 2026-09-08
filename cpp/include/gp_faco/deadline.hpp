#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gp_faco {

// Clock返回调用开始后的单调秒数；生产使用steady_clock，测试使用可控时钟。
class DeadlineLedger {
public:
    using Clock = std::function<double()>;
    DeadlineLedger(double budget, Clock clock) : budget_(budget), clock_(std::move(clock)) {
        if (!std::isfinite(budget_) || budget_ < 0 || !clock_) throw std::invalid_argument("预算或时钟无效");
    }
    // 次数入口只记录单调时间，不以一个很大的秒数伪装为无限截止。
    DeadlineLedger(std::nullopt_t, Clock clock) : budget_(0), clock_(std::move(clock)), enforced_(false) {
        if (!clock_) throw std::invalid_argument("记录资源仍需要有效时钟");
    }
    double elapsed() const {
        const double now = clock_();
        if (!std::isfinite(now) || now < last_) throw std::runtime_error("时钟必须非负且单调");
        last_ = now;
        return now + charged_;
    }
    bool expired() const { return enforced_ && elapsed() > budget_; }
    bool can_start() const { return !enforced_ || elapsed() < budget_; }
    void charge(double seconds) {
        if (!enforced_) throw std::logic_error("次数入口不能扣除墙钟费用");
        if (!std::isfinite(seconds) || seconds < 0) throw std::invalid_argument("缓存费用无效");
        charged_ += seconds;
    }
    bool completed_on_time(double completed) const {
        return std::isfinite(completed) && completed >= 0 && (!enforced_ || completed <= budget_);
    }
    double charged() const { return charged_; }
    double budget() const { return budget_; }
private:
    double budget_, charged_ = 0;
    mutable double last_ = 0;
    Clock clock_;
    bool enforced_ = true;
};

struct TimedIncumbent {
    std::vector<std::uint32_t> tour;
    double cost = std::numeric_limits<double>::infinity();
    double completed_seconds = 0;
    bool present = false;

    // 调用方已在host完成候选构造/下载及排列、成本核验，再取完成时间戳。
    // 此处只负责预算与严格改进判断；迟到更优解也不改变当前记录。
    bool offer(std::vector<std::uint32_t> candidate, double value, double completed,
               const DeadlineLedger& budget) {
        if (!std::isfinite(value) || value <= 0 || candidate.size() < 3)
            throw std::invalid_argument("incumbent候选无效");
        if (!budget.completed_on_time(completed) || (present && !(value < cost))) return false;
        tour = std::move(candidate); cost = value; completed_seconds = completed; present = true;
        return true;
    }
};

}  // namespace gp_faco
