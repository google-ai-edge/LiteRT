FROM us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest
RUN apt-get update && apt-get install -y --no-install-recommends libc++-18-dev libc++abi-18-dev llvm-18 clang-18