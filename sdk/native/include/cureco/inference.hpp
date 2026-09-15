#pragma once
#include "inference.h"
#include <memory>
#include <stdexcept>
#include <string>
namespace cureco {
using Result = std::unique_ptr<ci_result, decltype(&ci_result_destroy)>;
class Engine {
    std::unique_ptr<ci_engine, decltype(&ci_destroy)> handle_{nullptr, ci_destroy};

  public:
    explicit Engine(const std::string &path, const std::string &options = "{}") {
        ci_engine *p = nullptr;
        if (ci_create(path.c_str(), options.c_str(), &p))
            throw std::runtime_error(ci_last_error());
        handle_.reset(p);
    }
    std::string info() const { return ci_model_info(handle_.get()); }
    Result infer(const uint8_t *pixels, size_t bytes, int width, int height, size_t stride, int format = 0,
                 const float *params = nullptr, size_t count = 0, float threshold = 0.25f) {
        ci_result *result = nullptr;
        if (ci_run(handle_.get(), pixels, bytes, width, height, stride, format, params, count, threshold,
                   &result))
            throw std::runtime_error(ci_last_error());
        return Result(result, ci_result_destroy);
    }
};
} // namespace cureco
