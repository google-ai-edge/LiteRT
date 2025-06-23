# Using LiteRT Securely

This document discusses the LiteRT security model. It describes the security
risks to consider when using models, checkpoints or input data for training or
serving. We also provide guidelines on what constitutes a vulnerability in
LiteRT and how to report them.

## LiteRT models are programs

LiteRT
[**models**](https://developers.google.com/machine-learning/glossary/#model) (to
use a term commonly used by machine learning practitioners) are expressed as
programs that LiteRT executes. LiteRT programs are encoded as computation
[**graphs**](https://developers.google.com/machine-learning/glossary/#graph).
Since models are practically programs that LiteRT executes, using untrusted
models or graphs is equivalent to running untrusted code.

If you need to run untrusted models, execute them inside a
[**sandbox**](https://developers.google.com/code-sandboxing). Memory corruptions
in LiteRT ops can be recognized as security issues only if they are
reachable and exploitable through production-grade, benign models.

### Saved graphs and flatbuffer files

When loading untrusted serialized computation graphs (in form of a `.tflite`
model, or equivalent on-disk format), the set of computation primitives
available to LiteRT is powerful enough that you should assume that the
LiteRT process effectively executes arbitrary code.

The risk of loading untrusted files depends on the code or graph that you
are working with. When loading untrusted checkpoints, the values of the traced
variables from your model are also going to be untrusted. That means that if
your code interacts with the filesystem, network, etc. and uses checkpointed
variables as part of those interactions (ex: using a string variable to build a
filesystem path), a maliciously created checkpoint might be able to change the
targets of those operations, which could result in arbitrary
read/write/executions.

## Untrusted inputs during prediction

LiteRT supports a wide range of input data formats. For example it can
process images, audio, videos, and text.

These modifications and conversions are handled by a variety of libraries that
have different security properties and provide different levels of confidence
when dealing with untrusted data. Based on the security history of these
libraries we consider that it is safe to work with untrusted inputs for PNG,
BMP, GIF, WAV, RAW, RAW\_PADDED, CSV and PROTO formats. All other input formats
should be sandboxed if used to process untrusted data.

For example, if an attacker were to provide a malicious image file, they could
potentially exploit a vulnerability in the LiteRT code that handles images,
which could allow them to execute arbitrary code on the system running
LiteRT.

It is important to keep LiteRT up to date with the latest security patches
and follow the sandboxing guideline above to protect against these types of
vulnerabilities.

## Reporting vulnerabilities

### Vulnerabilities in LiteRT

This document covers different use cases for LiteRT together with comments
whether these uses were recommended or considered safe, or where we recommend
some form of isolation when dealing with untrusted data. As a result, this
document also outlines what issues we consider as LiteRT security
vulnerabilities.

We recognize issues as vulnerabilities only when they occur in scenarios that we
outline as safe; issues that have a security impact only when LiteRT is used
in a discouraged way (e.g. running untrusted models or checkpoints, data parsing
outside of the safe formats, etc.) are not treated as vulnerabilities.

### Reporting process

Please use [Google Bug Hunters reporting form](https://g.co/vulnz) to report
security vulnerabilities. Please include the following information along with
your report:

  - A descriptive title
  - Your name and affiliation (if any).
  - A description of the technical details of the vulnerabilities.
  - A minimal example of the vulnerability. It is very important to let us know
    how we can reproduce your findings.
  - An explanation of who can exploit this vulnerability, and what they gain
    when doing so. Write an attack scenario that demonstrates how your issue
    violates the use cases and security assumptions defined in the threat model.
    This will help us evaluate your report quickly, especially if the issue is
    complex.
  - Whether this vulnerability is public or known to third parties. If it is,
    please provide details.

We will try to fix the problems as soon as possible. Vulnerabilities will, in
general, be batched to be fixed at the same time as a release. We
credit reporters for identifying security issues, although we keep your name
confidential if you request it. Please see Google Bug Hunters program website
for more info.