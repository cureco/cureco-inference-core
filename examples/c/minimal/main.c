/* Packed RGB8 keeps this example independent of image decoder libraries. */
#include <cureco/inference.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

static int dimension(const char *value) {
    char *end;
    errno = 0;
    long size = strtol(value, &end, 10);
    return errno || *end || size < 1 || size > 16384 ? 0 : (int)size;
}
int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "Usage: infer model.onnx image.rgb width height\n"); return 2; }
    int w = dimension(argv[3]), h = dimension(argv[4]);
    if (!w || !h || (size_t)w * h > 16777216) return 2;
    size_t bytes = (size_t)w * h * 3;
    FILE *file = fopen(argv[2], "rb");
    unsigned char *pixels = (unsigned char *)malloc(bytes);
    if (!file || !pixels) { if(file) fclose(file); free(pixels); return 1; }
    int valid = fread(pixels, 1, bytes, file) == bytes && fgetc(file) == EOF;
    fclose(file);
    ci_engine *engine = NULL;
    ci_result *result = NULL;
    int status = 1;
    if (!valid) fprintf(stderr, "Expected exactly width * height * 3 RGB bytes\n");
    else if (ci_create(argv[1], "{}", &engine) ||
             ci_run(engine, pixels, bytes, w, h, (size_t)w * 3, 0, NULL, 0, .25f, &result))
        fprintf(stderr, "%s\n", ci_last_error());
    else { puts(ci_result_json(result)); status = 0; }
    /* Segmentation masks are available through ci_result_mask until result destruction. */
    ci_result_destroy(result);
    ci_destroy(engine);
    free(pixels);
    return status;
}
