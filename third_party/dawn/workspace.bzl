"""This file is used to load the dawn library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def repo():
    http_archive(
        name = "dawn",
        build_file = "//third_party/dawn:dawn.BUILD",
        patch_cmds = [
            # 1. C++17 fallback for std::span -> absl::Span
            "python3 -c 'p=\"generator/templates/api_cpp.h\"; c=open(p).read().replace(\"#include <span>\", \"#if __cplusplus >= 202002L\\n#include <span>\\n#else\\n#include \\\"absl/types/span.h\\\"\\nnamespace std { template <typename T> using span = absl::Span<T>; }\\n#endif\"); open(p,\"w\").write(c)'",
            # 2. C++17 fallback for C++20 unevaluated lambdas in BaseArgsTuple -> take_tuple_t
            "python3 -c 'p=\"generator/templates/api_cpp.h\"; c=open(p).read().replace(\"template <typename CppFT, typename CppFPtr, typename T>\\nstruct CppFTraitsImpl;\", \"template <size_t N, typename Tuple, typename Seq = std::make_index_sequence<N>>\\nstruct take_tuple;\\ntemplate <size_t N, typename... Args, size_t... Is>\\nstruct take_tuple<N, std::tuple<Args...>, std::index_sequence<Is...>> {\\n    using type = std::tuple<std::tuple_element_t<Is, std::tuple<Args...>>...>;\\n};\\ntemplate <size_t N, typename... Args>\\nusing take_tuple_t = typename take_tuple<N, std::tuple<Args...>>::type;\\n\\ntemplate <typename CppFT, typename CppFPtr, typename T>\\nstruct CppFTraitsImpl;\").replace(\"    using BaseArgsTuple = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {\\n        return std::type_identity<std::tuple<std::tuple_element_t<Is, std::tuple<CppArgs...>>...>>{};\\n    }(std::make_index_sequence<std::is_same_v<T, Untyped> ? NumCppArgs : NumCppArgs - 1>{})\\n    )::type;\", \"    using BaseArgsTuple = take_tuple_t<std::is_same_v<T, Untyped> ? NumCppArgs : NumCppArgs - 1, CppArgs...>;\"); open(p,\"w\").write(c)'",
            # 3. C++17 SFINAE fix for CallbackInfoHelper::Create capturing lambda constraints
            "python3 -c 'p=\"generator/templates/api_cpp.h\"; c=open(p).read().replace(\"if constexpr (requires(CInfoT x) { x.mode; }) {\", \"if constexpr (has_mode<CInfoT>::value) {\").replace(\"template <typename CInfoT, typename F>\\nstruct CallbackInfoHelper {\", \"template <typename T, typename = void>\\nstruct has_mode : std::false_type {};\\ntemplate <typename T>\\nstruct has_mode<T, std::void_t<decltype(std::declval<T>().mode)>> : std::true_type {};\\ntemplate <typename CInfoT, typename F>\\nstruct CallbackInfoHelper {\").replace(\"    template <typename T>\\n    requires (!CppFTraits<F>::capturing)\\n    static CInfoT Create(F lambda, T userdata) {\", \"    template <typename T, typename F_ = F, typename = std::enable_if_t<!CppFTraits<F_>::capturing>>\\n    static CInfoT Create(F lambda, T userdata) {\"); open(p,\"w\").write(c)'",
        ],
        strip_prefix = "dawn-20260720.160313",
        urls = [
            "https://github.com/google/dawn/archive/v20260720.160313.tar.gz",
        ],
    )
