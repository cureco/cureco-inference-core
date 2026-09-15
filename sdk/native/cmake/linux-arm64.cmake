# Cross-compile with the GNU aarch64 toolchain. ORT_ROOT must be an ARM64 SDK.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
# Supply -DCMAKE_SYSROOT=<target sysroot> when targeting an older JetPack/glibc.
# Jetson production builds should preferably run natively in its matching SDK image.
