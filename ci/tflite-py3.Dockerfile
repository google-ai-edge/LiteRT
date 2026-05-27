FROM us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget gnupg && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    # This script adds apt.llvm.org repo and installs clang-19, lld-19 etc.
    ./llvm.sh 19 && \
    apt-get install -y --no-install-recommends libc++-19-dev libc++abi-19-dev