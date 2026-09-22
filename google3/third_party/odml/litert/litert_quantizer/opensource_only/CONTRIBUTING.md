# Development Environment Setup

Every contributor to this repository should develop in a fork.

```bash
# Install uv (see https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ai-edge-quantizer
uv sync
```

## Running Tests

Individual test:
```bash
uv run ai_edge_quantizer/quantizer_test.py
```

All tests:
```bash
uv run pytest
```

## Build PyPi Package at Local

```bash
uv build
```

It will build a PyPi package under the `dist` folder, which you could install at local using command:

```bash
uv pip install dist/ai_edge_quantizer-*.whl
```


# Contributor License Agreement

- Contributions to this project must be accompanied by a [Contributor License
  Agreement](https://cla.developers.google.com/about) (CLA).

- Visit <https://cla.developers.google.com/> to see your current agreements or
  to sign a new one.

# Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

# Code Contribution Guidelines

We recommend that contributors read these tips from the Google Testing Blog:

- [Code Health: Providing Context with Commit Messages and Bug Reports](https://testing.googleblog.com/2017/09/code-health-providing-context-with.html)
- [Code Health: Understanding Code In Review](https://testing.googleblog.com/2018/05/code-health-understanding-code-in-review.html)
- [Code Health: Too Many Comments on Your Code Reviews?](https://testing.googleblog.com/2017/06/code-health-too-many-comments-on-your.html)
- [Code Health: To Comment or Not to Comment?](https://testing.googleblog.com/2017/07/code-health-to-comment-or-not-to-comment.html)

