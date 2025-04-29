FROM us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest
RUN apt-get update && apt-get install -y --no-install-recommends libc++abi-dev