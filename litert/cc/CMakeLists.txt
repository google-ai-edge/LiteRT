# LiteRT C++ API Library

# Collect all source files (excluding test files)
set(LITERT_CC_SOURCES
    litert_compiled_model.cc
    litert_custom_op_kernel.cc
    litert_macros.cc
    litert_model.cc
    litert_model_predicates.cc
    litert_op_options.cc
    litert_opaque_options.cc
    litert_rng.cc
    litert_shared_library.cc
    litert_tensor_buffer.cc
    litert_tensor_buffer_utils.cc
)

# Collect all header files
set(LITERT_CC_HEADERS
    litert_any.h
    litert_buffer_ref.h
    litert_c_types_printing.h
    litert_compiled_model.h
    litert_consts.h
    litert_custom_op_kernel.h
    litert_detail.h
    litert_dispatch_delegate.h
    litert_element_type.h
    litert_environment.h
    litert_environment_options.h
    litert_event.h
    litert_expected.h
    litert_handle.h
    litert_layout.h
    litert_logging.h
    litert_macros.h
    litert_model.h
    litert_model_predicates.h
    litert_numerics.h
    litert_op_options.h
    litert_opaque_options.h
    litert_options.h
    litert_platform_support.h
    litert_profiler.h
    litert_rng.h
    litert_shared_library.h
    litert_source_location.h
    litert_tensor_buffer.h
    litert_tensor_buffer_requirements.h
    litert_tensor_buffer_utils.h
    litert_tflite_error_status_builder.h
)

# Create the C++ API library
add_library(litert_cc STATIC ${LITERT_CC_SOURCES} ${LITERT_CC_HEADERS})

# Set target properties
set_target_properties(litert_cc PROPERTIES
    OUTPUT_NAME "litert_cc"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${LITERT_CC_HEADERS}"
)

# Include directories
target_include_directories(litert_cc
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_cc
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
        Threads::Threads
    PRIVATE
        ${CMAKE_DL_LIBS}
)

# Compile definitions
target_compile_definitions(litert_cc
    PRIVATE
        LITERT_EXPORTS
    PUBLIC
        $<$<PLATFORM_ID:Windows>:LITERT_STATIC_DEFINE>
)

# Platform-specific configurations
if(ANDROID)
    target_link_libraries(litert_cc PRIVATE log android)
elseif(WIN32)
    target_compile_definitions(litert_cc PRIVATE WIN32_LEAN_AND_MEAN NOGDI)
    target_link_libraries(litert_cc PRIVATE ws2_32)
elseif(UNIX)
    target_link_libraries(litert_cc PRIVATE ${CMAKE_DL_LIBS})
endif()

# Internal subdirectories (these provide internal implementation details)
add_subdirectory(internal)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_CC_TEST_SOURCES
        litert_any_test.cc
        litert_buffer_ref_test.cc
        litert_c_types_printing_test.cc
        litert_compiled_model_test.cc
        litert_custom_op_test.cc
        litert_detail_test.cc
        litert_element_type_test.cc
        litert_environment_options_test.cc
        litert_environment_test.cc
        litert_event_test.cc
        litert_expected_test.cc
        litert_handle_test.cc
        litert_layout_test.cc
        litert_logging_test.cc
        litert_macros_test.cc
        litert_model_predicates_test.cc
        litert_model_test.cc
        litert_op_options_test.cc
        litert_opaque_options_test.cc
        litert_rng_test.cc
        litert_shared_library_test.cc
        litert_source_location_test.cc
        litert_tensor_buffer_requirements_test.cc
        litert_tensor_buffer_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_CC_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_cc
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # Hardware-specific tests (only build if support is enabled)
    if(LITERT_ENABLE_GPU)
        add_executable(litert_compiled_model_gpu_test litert_compiled_model_gpu_test.cc)
        target_link_libraries(litert_compiled_model_gpu_test
            PRIVATE
                litert_cc
                gtest_main
                gtest
        )
        add_test(NAME litert_compiled_model_gpu_test COMMAND litert_compiled_model_gpu_test)
    endif()

    if(LITERT_ENABLE_QUALCOMM)
        add_executable(litert_compiled_model_qualcomm_test litert_compiled_model_qualcomm_test.cc)
        target_link_libraries(litert_compiled_model_qualcomm_test
            PRIVATE
                litert_cc
                gtest_main
                gtest
        )
        add_test(NAME litert_compiled_model_qualcomm_test COMMAND litert_compiled_model_qualcomm_test)

        add_executable(litert_compiled_model_npu_test litert_compiled_model_npu_test.cc)
        target_link_libraries(litert_compiled_model_npu_test
            PRIVATE
                litert_cc
                gtest_main
                gtest
        )
        add_test(NAME litert_compiled_model_npu_test COMMAND litert_compiled_model_npu_test)
    endif()

    # Shared library for testing
    add_library(test_shared_library SHARED test_shared_library.cc)
    set_target_properties(test_shared_library PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )
endif()

# Create alias for consistency
add_library(LiteRT::CC ALIAS litert_cc)

# Export for installation
install(TARGETS litert_cc
    EXPORT LiteRTCCTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/litert/cc
)