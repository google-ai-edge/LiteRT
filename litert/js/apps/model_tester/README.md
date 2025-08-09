# LiteRT.js Model Tester

Model Tester runs your LiteRT models in the browser using LiteRT.js on both
WebGPU and Wasm. It helps catch issues with the model before you start writing
code.

## Features

### Test Compatibility

A good first step in running a model on LiteRT.js is to test it in Model Tester.
Model Tester exposes compatibility issues and unsupported operations before you
begin writing code for your model pipeline.

### Test Correctness

Model Tester compares the output of running the model on Wasm CPU to the output
of running on WebGPU to ensure your model has the same outputs on both
accelerators.

### Benchmark Performance

Model Tester benchmarks your model on WebGPU and Wasm CPU so you can see if it
performs well enough for your application.

## Get Started

Install Model Tester with `npm i @litertjs-model-tester`. Then, run it with `npx
model-tester`. It will launch a local server that hosts the model tester site.

Model Tester supports the following CLI options:

```
$ npx model-tester --help
usage: model-tester [-h] [--public] [--port PORT] [--open]

LiteRT.js Model Tester

optional arguments:
  -h, --help   Show this help message and exit
  --public     Host the model tester publicly. By default, only connections from localhost are allowed.
  --port PORT
  --open       Whether to open Chrome. Defaults to true
```

## Development

Model Tester is part of the LiteRT.js monorepo, so you should first build the
monorepo by running `npm run build` in the root directory (`../../` relative to
this directory). Then, you can return to this directory and run `npm run dev` to
run the devserver.
