# Package publishing

The Python package is `cureco-inference-core`; the NuGet package is
`Cureco.Inference.Core`. Both include the shared native inference runtime.

| Registry | Workflow | GitHub environment |
| --- | --- | --- |
| PyPI | `publish-pypi.yml` | `pypi` |
| TestPyPI | `publish-testpypi.yml` | `testpypi` |
| NuGet.org | `publish-nuget.yml` | `nuget` |
| NuGet test site | `publish-testnuget.yml` | `testnuget` |

Configure a Trusted Publisher on each registry with the exact repository,
workflow filename, and environment above. NuGet workflows read the registry
profile name from the `NUGET_USER` and `TESTNUGET_USER` GitHub Actions variables,
respectively. Long-lived API keys are not required.

Stable GitHub Releases trigger production publishing; prereleases trigger the
test registries. Workflows also support manual execution with a release tag.
The `verify_only` option checks the package without uploading it. A published
release must contain the artifacts and `SHA256SUMS.txt` required by the selected
workflow; package versions and the tag must agree.

The TestNuGet workflow verifies the released `.nupkg`, publishes only to
`https://apiint.nugettest.org/v3/index.json`, then downloads the exact version
from that registry and runs the C# inference test with an empty package cache.
Package source mapping prevents the SDK from resolving from production NuGet
during this check. Other dependencies are resolved from NuGet.org.

Registry publication can complete before the package becomes downloadable.
The installation check retries while indexing completes. If indexing outlasts
the retry window, rerun the failed installation job, not the successful publish
job. Existing package versions must not be overwritten.
