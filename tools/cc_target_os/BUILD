package(
    # copybara:uncomment default_applicable_licenses = ["//third_party/odml:license"],
    default_visibility = ["//visibility:public"],
)

config_setting(
    name = "emscripten",
    values = {"define": "dummy_emscripten=true"},
)

config_setting(
    name = "android",
    constraint_values = ["@platforms//os:android"],
)

config_setting(
    name = "apple",
    constraint_values = ["@platforms//os:ios"],
)

config_setting(
    name = "chromiumos",
    values = {"define": "dummy_chromiumos=true"},
)

config_setting(
    name = "linux-google",
    values = {"define": "dummy_linux-google=true"},
)
