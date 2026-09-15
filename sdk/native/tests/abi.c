#include <cureco/inference.h>
#include <stdio.h>
int main(void) {
    ci_engine *engine = (ci_engine *)1;
    ci_result *result = (ci_result *)1;
    size_t count = 42;
    if (ci_abi_version() != 1) return 1;
    if (ci_create(NULL, "{}", &engine) == 0 || engine != NULL) return 2;
    if (!ci_last_error() || !*ci_last_error()) return 3;
    if (ci_run(NULL, NULL, 0, 0, 0, 0, 0, NULL, 0, .25f, &result) == 0 || result != NULL) return 4;
    if (ci_result_mask(NULL, &count) != NULL || count != 0) return 5;
    ci_result_destroy(NULL); ci_destroy(NULL);
    puts("PASS: ABI version, null arguments, error and ownership contracts");
    return 0;
}
