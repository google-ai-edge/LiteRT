load("@build_bazel_rules_android//android:rules.bzl", "android_library")

package(
    # copybara:uncomment default_applicable_licenses = ["@org_tensorflow//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

android_library(
    name = "tokenizer",
    srcs = ["Tokenizer.java"],
    custom_package = "com.google.ai.edge.litert.support.text.tokenizers",
)

android_library(
    name = "bert-tokenizer",
    srcs = ["BertTokenizer.java"],
    custom_package = "com.google.ai.edge.litert.support.text.tokenizers",
    deps = [
        ":tokenizer",
        "@org_tensorflow//tensorflow_lite_support/cc/text/tokenizers:bert_tokenizer_jni",  # unuseddeps: keep
    ],
)

android_library(
    name = "sentencepiece-tokenizer",
    srcs = ["SentencePieceTokenizer.java"],
    custom_package = "com.google.ai.edge.litert.support.text.tokenizers",
    deps = [
        ":tokenizer",
        "@org_tensorflow//tensorflow_lite_support/cc/text/tokenizers:sentencepiece_jni",  # unuseddeps: keep
    ],
)
