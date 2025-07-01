# LiteRT Tools

# Add subdirectory for flags first
add_subdirectory(flags)

# Tool utility libraries
set(LITERT_TOOLS_UTIL_SOURCES
    apply_plugin.cc
    dump.cc
    tool_display.cc
    benchmark_litert_model.cc
    benchmark_stripped_litert_model.cc
)

set(LITERT_TOOLS_UTIL_HEADERS
    apply_plugin.h
    dump.h
    outstream.h
    tool_display.h
    tensor_utils.h
    benchmark_litert_model.h
    benchmark_stripped_litert_model.h
)

# Create utility library
add_library(litert_tools_utils STATIC ${LITERT_TOOLS_UTIL_SOURCES} ${LITERT_TOOLS_UTIL_HEADERS})

target_include_directories(litert_tools_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

target_link_libraries(litert_tools_utils
    PUBLIC
        litert_c
        litert_cc
        litert_core
        litert_core_model
        litert_core_util
        litert_runtime
        litert_tools_flags
        ${LITERT_THIRD_PARTY_LIBS}
        Threads::Threads
)

# Platform-specific configurations
if(ANDROID)
    target_link_libraries(litert_tools_utils PRIVATE log android)
elseif(WIN32)
    target_compile_definitions(litert_tools_utils PRIVATE WIN32_LEAN_AND_MEAN NOGDI)
    target_link_libraries(litert_tools_utils PRIVATE ws2_32)
elseif(UNIX)
    target_link_libraries(litert_tools_utils PRIVATE ${CMAKE_DL_LIBS})
endif()

# Main run_model executable - This is our primary target!
add_executable(run_model 
    run_model.cc
    ${CMAKE_SOURCE_DIR}/tflite_stub.cc
)

target_include_directories(run_model
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

target_link_libraries(run_model
    PRIVATE
        litert_c
        litert_cc
        litert_core
        litert_core_model
        litert_core_util
        litert_runtime
        litert_tools_flags_vendors
        ${LITERT_THIRD_PARTY_LIBS}
        Threads::Threads
)

# Add vendor flag libraries if enabled
if(LITERT_ENABLE_QUALCOMM)
    target_link_libraries(run_model PRIVATE litert_tools_flags_qualcomm)
endif()

if(LITERT_ENABLE_MEDIATEK)
    target_link_libraries(run_model PRIVATE litert_tools_flags_mediatek)
endif()

if(LITERT_ENABLE_GOOGLE_TENSOR)
    target_link_libraries(run_model PRIVATE litert_tools_flags_google_tensor)
endif()

# Platform-specific configurations for run_model
if(ANDROID)
    target_link_libraries(run_model PRIVATE log android)
    # Make it static for easier deployment on Android
    target_link_options(run_model PRIVATE -static-libgcc -static-libstdc++)
elseif(WIN32)
    target_compile_definitions(run_model PRIVATE WIN32_LEAN_AND_MEAN NOGDI)
    target_link_libraries(run_model PRIVATE ws2_32)
elseif(UNIX)
    target_link_libraries(run_model PRIVATE ${CMAKE_DL_LIBS})
endif()

# Other tool executables
add_executable(analyze_model_main analyze_model_main.cc)
target_link_libraries(analyze_model_main
    PRIVATE
        litert_tools_utils
        ${LITERT_THIRD_PARTY_LIBS}
)

add_executable(benchmark_model benchmark_litert_model_main.cc)
target_link_libraries(benchmark_model
    PRIVATE
        litert_tools_utils
        ${LITERT_THIRD_PARTY_LIBS}
)

add_executable(npu_numerics_check npu_numerics_check.cc)
target_link_libraries(npu_numerics_check
    PRIVATE
        litert_c
        litert_cc
        litert_core
        litert_runtime
        litert_tools_flags_vendors
        ${LITERT_THIRD_PARTY_LIBS}
)

# Apply plugin tools (only if vendor support is enabled)
if(LITERT_ENABLE_QUALCOMM OR LITERT_ENABLE_MEDIATEK OR LITERT_ENABLE_GOOGLE_TENSOR)
    add_executable(apply_plugin_main apply_plugin_main.cc)
    target_link_libraries(apply_plugin_main
        PRIVATE
            litert_tools_utils
            ${LITERT_THIRD_PARTY_LIBS}
    )
endif()

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_TOOLS_TEST_SOURCES
        apply_plugin_test.cc
        tool_display_test.cc
        dump_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_TOOLS_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_tools_utils
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # Hardware-specific tests
    if(LITERT_ENABLE_GPU)
        add_executable(benchmark_litert_model_test benchmark_litert_model_test.cc)
        target_link_libraries(benchmark_litert_model_test
            PRIVATE
                litert_tools_utils
                gtest_main
                gtest
        )
        add_test(NAME benchmark_litert_model_test COMMAND benchmark_litert_model_test)
    endif()

    if(LITERT_ENABLE_QUALCOMM)
        add_executable(benchmark_litert_model_qualcomm_test benchmark_litert_model_qualcomm_test.cc)
        target_link_libraries(benchmark_litert_model_qualcomm_test
            PRIVATE
                litert_tools_utils
                gtest_main
                gtest
        )
        add_test(NAME benchmark_litert_model_qualcomm_test COMMAND benchmark_litert_model_qualcomm_test)
    endif()
endif()

# Create aliases for consistency
add_library(LiteRT::Tools::Utils ALIAS litert_tools_utils)

# Installation
install(TARGETS run_model analyze_model_main benchmark_model npu_numerics_check
    RUNTIME DESTINATION bin
)

if(TARGET apply_plugin_main)
    install(TARGETS apply_plugin_main
        RUNTIME DESTINATION bin
    )
endif()

# Export for installation
install(TARGETS litert_tools_utils
    EXPORT LiteRTToolsTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/litert/tools
)