FROM us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build-arm64:latest

# Install pyenv and pre-compile Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev build-essential libbz2-dev libncurses5-dev libncursesw5-dev \
    libffi-dev libreadline-dev libsqlite3-dev liblzma-dev zlib1g-dev git curl

ENV PYENV_ROOT /pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    pyenv install -s 3.11 && \
    pyenv global 3.11
