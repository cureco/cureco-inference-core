#ifndef CURECO_INFERENCE_H
#define CURECO_INFERENCE_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#ifdef CURECO_BUILD
#define CI_API __declspec(dllexport)
#else
#define CI_API __declspec(dllimport)
#endif
#else
#define CI_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ci_engine ci_engine;
typedef struct ci_result ci_result;
/* ABI v1. UTF-8 strings. Status 0 = success, 1 = error; outputs null on failure.
 * Error text belongs to the calling thread, valid until its next API operation.
 * Handles must not be destroyed while another thread uses them.
 * Engine runs are serialized. A result owns its JSON and mask until destroyed. */
CI_API uint32_t ci_abi_version(void);
CI_API const char *ci_last_error(void);
/* JSON provider list owned by library, or null on error. */
CI_API const char *ci_available_providers(void);
CI_API int ci_create(const char *model_path, const char *options_json, ci_engine **out);
/* Raw ONNX metadata as JSON string values. CPU session without inference
   adapter validation. Caller releases *out with ci_result_destroy. */
CI_API int ci_read_metadata(const char *model_path, ci_result **out);
CI_API void ci_destroy(ci_engine *engine);
CI_API const char *ci_model_info(const ci_engine *engine);
/* format: 0 RGB8, 1 BGR8, 2 GRAY8. Positive stride in bytes.
 * Input memory is borrowed only for this synchronous call. No file I/O or camera access.
 * params are float32 in parameter_names order. threshold applies only to detection. */
CI_API int ci_run(ci_engine *engine, const uint8_t *pixels, size_t bytes, int32_t width, int32_t height,
                  size_t stride, int32_t format, const float *params, size_t param_count, float threshold,
                  ci_result **out);
CI_API const char *ci_result_json(const ci_result *result);
/* Segmentation mask is int32, row-major, in original image coordinates; otherwise null. */
CI_API const int32_t *ci_result_mask(const ci_result *result, size_t *count);
CI_API void ci_result_destroy(ci_result *result);
#ifdef __cplusplus
}
#endif
#endif
