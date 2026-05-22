# Contributing guidelines

## How to become a contributor and submit your own code

### Contributor License Agreements

We'd love to accept your patches! Before we can take them, we have to jump a
couple of legal hurdles.

Please fill out either the individual or corporate Contributor License Agreement
(CLA).

- If you are an individual writing original source code and you're sure you own
  the intellectual property, then you'll need to sign an
  [individual CLA](https://code.google.com/legal/individual-cla-v1.0.html).
- If you work for a company that wants to allow you to contribute your work,
  then you'll need to sign a
  [corporate CLA](https://code.google.com/legal/corporate-cla-v1.0.html).

Follow either of the two links above to access the appropriate CLA and
instructions for how to sign and return it. Once we receive it, we'll be able to
accept your pull requests.

***NOTE***: Only original source code from you and other people that have signed
the CLA can be accepted into the main repository.

### Contributing code

If you have improvements to LiteRT, send us your pull requests! For those just
getting started, Github has a
[howto](https://help.github.com/articles/using-pull-requests/).

LiteRT team members will be assigned to review your pull requests. Once the
pull requests are approved and pass continuous integration checks, we will merge
the pull requests. For some pull requests, we will apply the patch for each pull
request to our internal version control system first, and export the change out
as a new commit later, at which point the original pull request will be closed.
The commits in the pull request will be squashed into a single commit with the
pull request creator as the author. These pull requests will be labeled as
pending merge internally.

#### C++ coding style

Changes to TensorFlow C++ code should conform to
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Use `clang-tidy` to check your C/C++ changes. To install `clang-tidy` on
ubuntu:16.04, do:

```bash
apt-get install -y clang-tidy
```

You can check a C/C++ file by doing:

```bash
clang-format <my_cc_file> --style=google > /tmp/my_cc_file.cc
diff <my_cc_file> /tmp/my_cc_file.cc
```

#### Python coding style

Changes to TensorFlow Python code should conform to
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

Use `pylint` to check your Python changes. To install `pylint` and check a file
with `pylint` against TensorFlow's custom style definition:

```bash
pip install pylint
pylint --rcfile=tensorflow/tools/ci_build/pylintrc myfile.py
```

Note `pylint --rcfile=tensorflow/tools/ci_build/pylintrc` should run from the
top level tensorflow directory.

#### Coding style for other languages

*   [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)
*   [Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html)
*   [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
*   [Google Objective-C Style Guide](https://google.github.io/styleguide/objcguide.html)

#### License

Include a license at the top of new files.
