# Security

Only load ONNX models from trusted sources. Model parsing and inference execute native code;
a model file is not a sandboxed document.

The sample API binds to IPv4 loopback only, requires bearer authentication and rejects browser origins.
Do not expose it through a proxy or port forward without an appropriate security boundary.
Processes running as the same account are within the same trust boundary.

Tokens are protected with Windows DPAPI for the current user. Do not share settings, models,
images, logs or signing credentials in public issues or repository files.

Report vulnerabilities using private security advisories when enabled, or contact the package
maintainer privately. Do not post credentials or private data in public issues.

Automated public-source checks are an additional safeguard, not a guarantee.
Review source, Git history and exact release archives before publication.
