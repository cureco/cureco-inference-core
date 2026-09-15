#include "cureco/inference.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;
static thread_local std::string last_error;
static void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
static Ort::Env &environment() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Cureco");
    return env;
}
static size_t product(size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max() / b, "Image size overflow");
    return a * b;
}
struct ci_result {
    std::string text;
    std::vector<int32_t> mask;
};
struct ci_engine {
    std::unique_ptr<Ort::Session> session;
    json metadata, contract, classes;
    std::string task, info;
    std::vector<std::string> inputs, outputs;
    std::vector<int64_t> image_shape;
    int64_t param_dim = 0;
    bool has_params = false;
    std::array<float, 3> mean{0, 0, 0}, stddev{1, 1, 1};
    std::mutex mutex;
};
static json signature(Ort::Session &s, bool input) {
    Ort::AllocatorWithDefaultOptions allocator;
    json items = json::array();
    size_t count = input ? s.GetInputCount() : s.GetOutputCount();
    for (size_t i = 0; i < count; ++i) {
        auto name = input ? s.GetInputNameAllocated(i, allocator) : s.GetOutputNameAllocated(i, allocator);
        auto type = input ? s.GetInputTypeInfo(i) : s.GetOutputTypeInfo(i);
        auto tensor = type.GetTensorTypeAndShapeInfo();
        auto shape = tensor.GetShape();
        std::vector<const char *> symbols(shape.size());
        tensor.GetSymbolicDimensions(symbols.data(), symbols.size());
        json dims = json::array();
        for (size_t d = 0; d < shape.size(); ++d) {
            if (shape[d] >= 0)
                dims.push_back(shape[d]);
            else if (symbols[d] && *symbols[d])
                dims.push_back(symbols[d]);
            else
                dims.push_back(nullptr);
        }
        auto dt = tensor.GetElementType();
        std::string dtype = dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT   ? "float"
                            : dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? "int64"
                                                                        : "unsupported";
        items.push_back({{"name", name.get()}, {"shape", dims}, {"dtype", dtype}});
    }
    return items;
}
static void load(ci_engine &e, const char *path, const char *options) {
    auto config = json::parse(options ? options : "{}");
    auto providers = config.value("providers", std::vector<std::string>{"CPUExecutionProvider"});
    require(!providers.empty(), "At least one execution provider is required");
    auto available = Ort::GetAvailableProviders();
    Ort::SessionOptions opts;
    bool allow_cpu = config.value("allow_cpu_fallback", true);
    if (!allow_cpu)
        opts.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    for (auto &provider : providers) {
        require(std::find(available.begin(), available.end(), provider) != available.end(),
                "Unavailable provider: " + provider);
        if (provider == "CPUExecutionProvider")
            require(allow_cpu, "CPU provider conflicts with disabled fallback");
        else if (provider == "CUDAExecutionProvider") {
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = config.value("device_id", 0);
            cuda.gpu_mem_limit = std::numeric_limits<size_t>::max();
            cuda.do_copy_in_default_stream = 1;
            opts.AppendExecutionProvider_CUDA(cuda);
        } else if (provider == "TensorrtExecutionProvider") {
            OrtTensorRTProviderOptions trt{};
            trt.device_id = config.value("device_id", 0);
            trt.trt_max_workspace_size = config.value("trt_max_workspace_size", size_t(1) << 30);
            trt.trt_max_partition_iterations = 1000;
            trt.trt_min_subgraph_size = 1;
            trt.trt_fp16_enable = config.value("fp16", false);
            std::string cache = config.value("engine_cache_path", std::string());
            trt.trt_engine_cache_enable = !cache.empty();
            trt.trt_engine_cache_path = cache.c_str();
            opts.AppendExecutionProvider_TensorRT(trt);
        } else
            throw std::runtime_error("Provider not yet supported by native SDK: " + provider);
    }
    e.session = std::make_unique<Ort::Session>(environment(), std::filesystem::u8path(path).c_str(), opts);
    Ort::AllocatorWithDefaultOptions allocator;
    auto meta = e.session->GetModelMetadata();
    e.metadata = json::object();
    for (auto &key : meta.GetCustomMetadataMapKeysAllocated(allocator)) {
        auto raw = meta.LookupCustomMetadataMapAllocated(key.get(), allocator);
        auto value = json::parse(raw.get(), nullptr, false);
        if (value.is_discarded())
            value = std::string(raw.get());
        e.metadata[key.get()] = value;
    }
    auto ins = signature(*e.session, true), outs = signature(*e.session, false);
    e.contract = e.metadata.value("cureco_contract", json());
    e.task = e.metadata.value("dataset_type", std::string("classification"));
    if (!e.contract.is_null()) {
        auto &c = e.contract;
        require(c.is_object() && c.contains("schema_version") && c["schema_version"].is_number_integer() &&
                    c["schema_version"] == 1,
                "Unsupported Cureco contract version");
        e.task = c.at("task").get<std::string>();
        require(!e.metadata.contains("dataset_type") || e.metadata["dataset_type"] == e.task,
                "Task metadata mismatch");
        require(c.value("image_layout", "") == "NCHW", "Only NCHW is supported");
        require(c.at("inputs") == ins && c.at("outputs") == outs, "ONNX signature mismatch");
    } else
        require(e.task == "classification" || e.task.empty(), "Detection/segmentation require a v1 contract");
    if (e.task.empty())
        e.task = "classification";
    require(e.task == "classification" || e.task == "detection" || e.task == "segmentation",
            "Unsupported task");
    bool image_found = false;
    for (size_t i = 0; i < ins.size(); ++i) {
        auto name = ins[i]["name"].get<std::string>();
        e.inputs.push_back(name);
        auto shape = e.session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        require(ins[i]["dtype"] == "float", "Inputs must be float32");
        if (name == "img_tensor") {
            require(shape.size() == 4 && (shape[0] == 1 || shape[0] < 0) && shape[1] == 3 && shape[2] > 0 &&
                        shape[3] > 0,
                    "Expected single-image RGB NCHW input");
            require(shape[2] <= 32768 && shape[3] <= 32768, "Model image dimensions exceed SDK limit");
            e.image_shape = {1, 3, shape[2], shape[3]};
            image_found = true;
        } else {
            require(e.task == "classification" && name == "param_tensor" && shape.size() == 2 &&
                        (shape[0] == 1 || shape[0] < 0) && shape[1] >= 0,
                    "Unsupported auxiliary input");
            e.has_params = true;
            e.param_dim = shape[1];
        }
    }
    require(image_found, "Missing img_tensor");
    for (auto &o : outs)
        e.outputs.push_back(o["name"]);
    json normalization = nullptr;
    auto legacy = e.metadata.value("preprocessing", json::object());
    if (legacy.is_object() && !legacy.value("normalize_mean", json()).is_null() &&
        !legacy.value("normalize_std", json()).is_null())
        normalization = {{"mean", legacy["normalize_mean"]}, {"std", legacy["normalize_std"]}};
    if (!e.contract.is_null() && !e.contract.value("preprocessing", json()).is_null()) {
        auto p = e.contract["preprocessing"], r = p.at("resize");
        require(p.value("location", "") == "outside_graph" && p.value("color", "") == "RGB" &&
                    p.at("scale") == 1.0 / 255 && r.value("mode", "") == "stretch" &&
                    r.value("interpolation", "") == "bicubic" && r.value("antialias", false),
                "Unsupported preprocessing");
        require(r.at("size") == json::array({e.image_shape[2], e.image_shape[3]}), "Resize size mismatch");
        require(p.value("normalization", json()) == normalization, "Normalization metadata mismatch");
    }
    if (!normalization.is_null()) {
        require(normalization["mean"].size() == 3 && normalization["std"].size() == 3,
                "Expected three normalization channels");
        for (size_t i = 0; i < 3; ++i) {
            e.mean[i] = normalization["mean"][i];
            e.stddev[i] = normalization["std"][i];
            require(std::isfinite(e.mean[i]) && std::isfinite(e.stddev[i]) && e.stddev[i] > 0,
                    "Invalid normalization");
        }
    }
    if (e.metadata.contains("parameter_names")) {
        auto names = e.metadata["parameter_names"];
        require(names.is_array(), "parameter_names must be an array");
        std::vector<std::string> unique;
        for (auto &n : names) {
            require(n.is_string() && !n.get<std::string>().empty(), "Invalid parameter name");
            unique.push_back(n);
        }
        std::sort(unique.begin(), unique.end());
        require(std::adjacent_find(unique.begin(), unique.end()) == unique.end(),
                "Duplicate parameter names");
        require(!e.has_params || names.size() == static_cast<size_t>(e.param_dim),
                "Parameter names/dimension mismatch");
        require(!e.metadata.contains("num_params") || e.metadata["num_params"] == names.size(),
                "num_params mismatch");
    }
    auto semantics = e.contract.is_null() ? json() : e.contract.value("output_semantics", json());
    if (e.task == "detection") {
        require(e.outputs == std::vector<std::string>{"boxes", "labels", "scores"},
                "Unsupported detection outputs");
        require(outs[0]["dtype"] == "float" && outs[1]["dtype"] == "int64" && outs[2]["dtype"] == "float",
                "Unsupported detection dtypes");
        require(semantics == json({{"boxes", "xyxy"},
                                   {"coordinate_space", "input_image_pixels"},
                                   {"labels", "zero_based_class_id"},
                                   {"scores", "confidence"}}),
                "Unsupported detection semantics");
        auto nms = e.contract.value("nms", std::string());
        require(nms == "inside_graph" || nms == "not_applied", "Unknown NMS policy");
    } else {
        require(e.outputs == std::vector<std::string>{"output"} && outs[0]["dtype"] == "float",
                "Unsupported output");
        require(e.contract.is_null() || semantics == json({{"output", "logits"}, {"class_axis", 1}}),
                "Expected class logits");
        auto shape = e.session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        require(shape.size() == (e.task == "classification" ? 2u : 4u), "Unsupported output rank");
    }
    auto policy = e.contract.is_null() ? json() : e.contract.value("mask_policy", json());
    if (!policy.is_null())
        require(policy.is_object() &&
                    (policy.value("background_mode", "") == "implicit" ||
                     policy.value("background_mode", "") == "explicit") &&
                    policy.value("ignore_index", 0) == 255,
                "Unsupported mask policy");
    e.classes = e.metadata.value("idx_to_class", json::object());
    require(e.classes.is_object(), "Invalid class mapping");
    e.info = json({{"task", e.task},
                   {"inputs", ins},
                   {"outputs", outs},
                   {"contract", e.contract},
                   {"classes", e.classes},
                   {"parameter_names", e.metadata.value("parameter_names", json::array())},
                   {"configured_providers", providers},
                   {"allow_cpu_fallback", allow_cpu},
                   {"onnxruntime_version", Ort::GetVersionString()}})
                 .dump();
}
// Separable cubic convolution with scale-aware antialiasing. Quantized 22-bit
// coefficients and uint8 intermediate rounding preserve Pillow RGB resize behavior.
struct Filter {
    int start;
    std::vector<int32_t> weights;
};
static double cubic(double x) {
    x = std::abs(x);
    if (x < 1)
        return ((1.5 * x - 2.5) * x) * x + 1;
    if (x < 2)
        return ((-0.5 * x + 2.5) * x - 4) * x + 2;
    return 0;
}
static std::vector<Filter> filters(int from, int to) {
    double scale = double(from) / to, widen = std::max(1.0, scale), support = 2 * widen;
    std::vector<Filter> result;
    for (int i = 0; i < to; ++i) {
        double center = (i + 0.5) * scale;
        int begin = std::max(0, int(center - support + 0.5)),
            end = std::min(from, int(center + support + 0.5));
        std::vector<double> ws;
        double sum = 0;
        for (int j = begin; j < end; ++j) {
            double w = cubic((j - center + 0.5) / widen);
            ws.push_back(w);
            sum += w;
        }
        Filter f{begin, {}};
        for (double w : ws)
            f.weights.push_back(static_cast<int32_t>(std::round(w / sum * (1 << 22))));
        result.push_back(std::move(f));
    }
    return result;
}
static uint8_t rounded(int64_t sum) {
    return static_cast<uint8_t>(std::clamp<int64_t>((sum + (1 << 21)) >> 22, 0, 255));
}
static std::vector<float> preprocess(ci_engine &e, const uint8_t *src, int w, int h, size_t stride,
                                     int format) {
    int tw = static_cast<int>(e.image_shape[3]), th = static_cast<int>(e.image_shape[2]);
    auto fx = filters(w, tw), fy = filters(h, th);
    std::vector<uint8_t> horizontal(product(product(tw, h), 3));
    int channels = format == 2 ? 1 : 3;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < tw; ++x)
            for (int c = 0; c < 3; ++c) {
                int sc = format == 2 ? 0 : format == 1 ? 2 - c : c;
                int64_t sum = 0;
                auto &f = fx[x];
                for (size_t j = 0; j < f.weights.size(); ++j)
                    sum +=
                        int64_t(src[size_t(y) * stride + size_t(f.start + j) * channels + sc]) * f.weights[j];
                horizontal[(size_t(y) * tw + x) * 3 + c] = rounded(sum);
            }
    size_t plane = product(tw, th);
    std::vector<float> tensor(product(plane, 3));
    for (int y = 0; y < th; ++y)
        for (int x = 0; x < tw; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t sum = 0;
                auto &f = fy[y];
                for (size_t j = 0; j < f.weights.size(); ++j)
                    sum += int64_t(horizontal[((f.start + j) * size_t(tw) + x) * 3 + c]) * f.weights[j];
                tensor[c * plane + size_t(y) * tw + x] = (rounded(sum) / 255.0f - e.mean[c]) / e.stddev[c];
            }
    return tensor;
}
static std::string label(ci_engine &e, int64_t id) {
    return e.classes.value(std::to_string(id), std::string("unknown"));
}
static std::unique_ptr<ci_result> run(ci_engine &e, const uint8_t *pixels, size_t bytes, int w, int h,
                                      size_t stride, int format, const float *params, size_t count,
                                      float threshold) {
    require(pixels && w > 0 && h > 0 && w <= 32768 && h <= 32768 && format >= 0 && format <= 2,
            "Invalid image buffer/format");
    size_t row = product(w, format == 2 ? 1 : 3), offset = product(h - 1, stride);
    require(stride >= row && offset <= bytes && row <= bytes - offset, "Image buffer too small");
    require((!e.has_params || count == static_cast<size_t>(e.param_dim)) && (!count || params),
            "Parameter count mismatch");
    for (size_t i = 0; i < count; ++i)
        require(std::isfinite(params[i]), "Non-finite parameter");
    require(std::isfinite(threshold) && threshold >= 0 && threshold <= 1, "Threshold must be in [0,1]");
    auto tensor = preprocess(e, pixels, w, h, stride, format);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> values;
    std::vector<const char *> names, output_names;
    std::array<int64_t, 2> ps{1, e.param_dim};
    float dummy = 0;
    for (auto &name : e.inputs) {
        names.push_back(name.c_str());
        if (name == "img_tensor")
            values.push_back(Ort::Value::CreateTensor<float>(memory, tensor.data(), tensor.size(),
                                                             e.image_shape.data(), 4));
        else
            values.push_back(Ort::Value::CreateTensor<float>(
                memory, count ? const_cast<float *>(params) : &dummy, count, ps.data(), 2));
    }
    for (auto &name : e.outputs)
        output_names.push_back(name.c_str());
    auto output = e.session->Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(),
                                 output_names.data(), output_names.size());
    auto result = std::make_unique<ci_result>();
    json data = {{"task", e.task}};
    if (e.task == "classification") {
        auto shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
        require(shape.size() == 2 && shape[0] == 1 && shape[1] > 0, "Invalid classification output shape");
        auto p = output[0].GetTensorData<float>();
        size_t n = static_cast<size_t>(shape[1]);
        for (size_t i = 0; i < n; ++i)
            require(std::isfinite(p[i]), "Non-finite logits");
        auto best = std::max_element(p, p + n);
        double sum = 0;
        for (size_t i = 0; i < n; ++i)
            sum += std::exp(double(p[i] - *best));
        int64_t id = best - p;
        data["results"] =
            json::array({{{"class_id", id}, {"label", label(e, id)}, {"confidence", 1.0 / sum}}});
    } else if (e.task == "detection") {
        auto shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
        require((shape.size() == 2 && shape[1] == 4) || (shape.size() == 3 && shape[0] == 1 && shape[2] == 4),
                "Invalid boxes shape");
        size_t n = static_cast<size_t>(shape[shape.size() - 2]);
        auto expected = shape.size() == 2 ? std::vector<int64_t>{static_cast<int64_t>(n)}
                                          : std::vector<int64_t>{1, static_cast<int64_t>(n)};
        require(output[1].GetTensorTypeAndShapeInfo().GetShape() == expected &&
                    output[2].GetTensorTypeAndShapeInfo().GetShape() == expected,
                "Detection output shape mismatch");
        auto boxes = output[0].GetTensorData<float>();
        auto labels = output[1].GetTensorData<int64_t>();
        auto scores = output[2].GetTensorData<float>();
        data["detections"] = json::array();
        for (size_t i = 0; i < n; ++i) {
            require(std::isfinite(scores[i]) && scores[i] >= 0 && scores[i] <= 1 && labels[i] >= 0,
                    "Invalid detection result");
            for (int c = 0; c < 4; ++c)
                require(std::isfinite(boxes[i * 4 + c]), "Non-finite box");
            if (scores[i] < threshold)
                continue;
            json box = json::array();
            for (int c = 0; c < 4; ++c) {
                double limit = c % 2 ? h : w;
                double size = static_cast<double>(c % 2 ? e.image_shape[2] : e.image_shape[3]);
                box.push_back(std::clamp(boxes[i * 4 + c] * limit / size, 0.0, limit));
            }
            data["detections"].push_back({{"class_id", labels[i]},
                                          {"label", label(e, labels[i])},
                                          {"confidence", scores[i]},
                                          {"box", box}});
        }
        data["coordinate_space"] = "original_image_pixels";
    } else {
        auto shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
        require(shape.size() == 4 && shape[0] == 1 && shape[1] > 0 && shape[1] <= INT32_MAX && shape[2] > 0 &&
                    shape[3] > 0,
                "Invalid segmentation shape");
        auto logits = output[0].GetTensorData<float>();
        size_t plane = product(static_cast<size_t>(shape[2]), static_cast<size_t>(shape[3]));
        std::vector<int32_t> mask(plane);
        for (size_t i = 0; i < plane; ++i) {
            float best = -std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < shape[1]; ++c) {
                float v = logits[c * plane + i];
                require(std::isfinite(v), "Non-finite logits");
                if (v > best) {
                    best = v;
                    mask[i] = static_cast<int32_t>(c);
                }
            }
        }
        result->mask.resize(product(w, h));
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                size_t sy = std::min<size_t>(
                    shape[2] - 1, static_cast<size_t>(((2 * int64_t(y) + 1) * shape[2]) / (2 * h)));
                size_t sx = std::min<size_t>(
                    shape[3] - 1, static_cast<size_t>(((2 * int64_t(x) + 1) * shape[3]) / (2 * w)));
                result->mask[size_t(y) * w + x] = mask[sy * shape[3] + sx];
            }
        data["width"] = w;
        data["height"] = h;
        data["classes"] = e.classes;
        data["mask_policy"] = e.contract.value("mask_policy", json());
    }
    result->text = data.dump();
    return result;
}
uint32_t ci_abi_version(void) { return 1; }
const char *ci_last_error(void) { return last_error.c_str(); }
const char *ci_available_providers(void) {
    try {
        static const std::string text = json(Ort::GetAvailableProviders()).dump();
        return text.c_str();
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown native error";
        return nullptr;
    }
}
int ci_create(const char *path, const char *options, ci_engine **out) {
    if (out)
        *out = nullptr;
    try {
        last_error.clear();
        require(path && out, "Null create argument");
        auto e = std::make_unique<ci_engine>();
        load(*e, path, options);
        *out = e.release();
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return 1;
    } catch (...) {
        last_error = "Unknown native error";
        return 1;
    }
}
int ci_read_metadata(const char *path, ci_result **out) {
    if (out)
        *out = nullptr;
    try {
        last_error.clear();
        require(path && out, "Null metadata argument");
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        Ort::Session session(environment(), std::filesystem::u8path(path).c_str(), opts);
        Ort::AllocatorWithDefaultOptions allocator;
        auto metadata = session.GetModelMetadata();
        json values = json::object();
        for (auto &key : metadata.GetCustomMetadataMapKeysAllocated(allocator)) {
            auto value = metadata.LookupCustomMetadataMapAllocated(key.get(), allocator);
            values[key.get()] = value.get();
        }
        auto result = std::make_unique<ci_result>();
        result->text = values.dump();
        *out = result.release();
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return 1;
    } catch (...) {
        last_error = "Unknown native error";
        return 1;
    }
}
void ci_destroy(ci_engine *e) { delete e; }
const char *ci_model_info(const ci_engine *e) { return e ? e->info.c_str() : nullptr; }
int ci_run(ci_engine *e, const uint8_t *pixels, size_t bytes, int32_t w, int32_t h, size_t stride,
           int32_t format, const float *params, size_t count, float threshold, ci_result **out) {
    if (out)
        *out = nullptr;
    try {
        last_error.clear();
        require(e && out, "Null run argument");
        std::lock_guard<std::mutex> lock(e->mutex);
        *out = run(*e, pixels, bytes, w, h, stride, format, params, count, threshold).release();
        return 0;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        return 1;
    } catch (...) {
        last_error = "Unknown native error";
        return 1;
    }
}
const char *ci_result_json(const ci_result *r) { return r ? r->text.c_str() : nullptr; }
const int32_t *ci_result_mask(const ci_result *r, size_t *count) {
    if (count)
        *count = r ? r->mask.size() : 0;
    return r && !r->mask.empty() ? r->mask.data() : nullptr;
}
void ci_result_destroy(ci_result *r) { delete r; }
