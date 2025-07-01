# LiteRT C API Library

# Collect all source files (excluding test files)
set(LITERT_C_SOURCES
    litert_accelerator.cc
    litert_accelerator_registration.cc
    litert_common.cc
    litert_compiled_model.cc
    litert_environment.cc
    litert_environment_options.cc
    litert_event.cc
    litert_layout.cc
    litert_logging.cc
    litert_metrics.cc
    litert_model.cc
    litert_op_options_stub.cc
    litert_opaque_options.cc
    litert_options.cc
    litert_profiler.cc
    litert_tensor_buffer.cc
    litert_tensor_buffer_requirements.cc
)

# Collect all header files
set(LITERT_C_HEADERS
    litert_accelerator.h
    litert_accelerator_registration.h
    litert_any.h
    litert_common.h
    litert_compiled_model.h
    litert_custom_op_kernel.h
    litert_dispatch_delegate.h
    litert_environment.h
    litert_environment_options.h
    litert_event.h
    litert_event_type.h
    litert_gl_types.h
    litert_layout.h
    litert_logging.h
    litert_metrics.h
    litert_model.h
    litert_op_code.h
    litert_op_options.h
    litert_opaque_options.h
    litert_options.h
    litert_platform_support.h
    litert_profiler.h
    litert_profiler_event.h
    litert_tensor_buffer.h
    litert_tensor_buffer_requirements.h
    litert_tensor_buffer_types.h
)

# Create the C API library
add_library(litert_c STATIC ${LITERT_C_SOURCES} ${LITERT_C_HEADERS})

# Set target properties
set_target_properties(litert_c PROPERTIES
    OUTPUT_NAME "litert_c"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${LITERT_C_HEADERS}"
)

# Include directories
target_include_directories(litert_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_c
    PUBLIC
        ${LITERT_THIRD_PARTY_LIBS}
        Threads::Threads
)

# Compile definitions
target_compile_definitions(litert_c
    PRIVATE
        LITERT_EXPORTS
    PUBLIC
        $<$<PLATFORM_ID:Windows>:LITERT_STATIC_DEFINE>
)

# Platform-specific configurations
if(ANDROID)
    target_link_libraries(litert_c PRIVATE log android)
elseif(WIN32)
    target_compile_definitions(litert_c PRIVATE WIN32_LEAN_AND_MEAN NOGDI)
    target_link_libraries(litert_c PRIVATE ws2_32)
endif()

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_C_TEST_SOURCES
        litert_accelerator_registration_test.cc
        litert_accelerator_test.cc
        litert_common_test.cc
        litert_compiled_model_test.cc
        litert_custom_op_test.cc
        litert_environment_options_test.cc
        litert_logging_test.cc
        litert_model_test.cc
        litert_op_options_test.cc
        litert_opaque_options_test.cc
        litert_options_test.cc
        litert_profiler_test.cc
        litert_tensor_buffer_requirements_test.cc
        litert_tensor_buffer_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_C_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_c
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Add options subdirectory
add_subdirectory(options)

# Create alias for consistency
add_library(LiteRT::C ALIAS litert_c)

# Export for installation
install(TARGETS litert_c
    EXPORT LiteRTCTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/litert/c
)