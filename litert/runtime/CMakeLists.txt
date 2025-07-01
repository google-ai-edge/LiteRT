# LiteRT Runtime Library

# Base runtime sources (always included)
set(LITERT_RUNTIME_BASE_SOURCES
    accelerator_registry.cc
    compiled_model.cc
    custom_op_dispatcher.cc
    event.cc
    external_litert_buffer_context.cc
    profiler.cc
    tensor_buffer.cc
    tensor_buffer_conversion.cc
    tensor_buffer_requirements.cc
    tfl_utils.cc
)

# Platform-specific sources
set(LITERT_RUNTIME_PLATFORM_SOURCES)

# Android-specific sources
if(ANDROID)
    list(APPEND LITERT_RUNTIME_PLATFORM_SOURCES
        ahwb_buffer.cc
        dmabuf_buffer.cc
        ion_buffer.cc
    )
endif()

# GPU-specific sources (if GPU support is enabled)
if(LITERT_ENABLE_GPU)
    list(APPEND LITERT_RUNTIME_PLATFORM_SOURCES
        gl_buffer.cc
        gl_texture.cc
        gpu_environment.cc
        open_cl_memory.cc
        open_cl_sync.cc
    )
endif()

# Qualcomm-specific sources (if Qualcomm support is enabled)
if(LITERT_ENABLE_QUALCOMM)
    list(APPEND LITERT_RUNTIME_PLATFORM_SOURCES
        fastrpc_buffer.cc
    )
endif()

# Combine all sources
set(LITERT_RUNTIME_SOURCES
    ${LITERT_RUNTIME_BASE_SOURCES}
    ${LITERT_RUNTIME_PLATFORM_SOURCES}
)

# Collect all header files
set(LITERT_RUNTIME_HEADERS
    accelerator.h
    accelerator_registry.h
    ahwb_buffer.h
    compiled_model.h
    custom_op_dispatcher.h
    dmabuf_buffer.h
    event.h
    external_litert_buffer_context.h
    fastrpc_buffer.h
    gl_buffer.h
    gl_texture.h
    gpu_environment.h
    ion_buffer.h
    litert_cpu_options.h
    litert_google_tensor.h
    litert_runtime_options.h
    metrics.h
    open_cl_memory.h
    open_cl_sync.h
    profiler.h
    tensor_buffer.h
    tensor_buffer_conversion.h
    tensor_buffer_requirements.h
    tfl_utils.h
)

# Create the runtime library
add_library(litert_runtime STATIC ${LITERT_RUNTIME_SOURCES} ${LITERT_RUNTIME_HEADERS})

# Set target properties
set_target_properties(litert_runtime PROPERTIES
    OUTPUT_NAME "litert_runtime"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${LITERT_RUNTIME_HEADERS}"
)

# Include directories
target_include_directories(litert_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_runtime
    PUBLIC
        litert_c
        litert_cc
        litert_core
        litert_core_model
        litert_core_util
        ${LITERT_THIRD_PARTY_LIBS}
        Threads::Threads
)

# Platform-specific configurations
if(ANDROID)
    target_link_libraries(litert_runtime 
        PRIVATE 
            log 
            android
    )
    # Add Android NDK specific libraries if available
    find_library(ANDROID_EGL_LIB EGL)
    find_library(ANDROID_GLES_LIB GLESv3)
    if(ANDROID_EGL_LIB AND ANDROID_GLES_LIB)
        target_link_libraries(litert_runtime PRIVATE ${ANDROID_EGL_LIB} ${ANDROID_GLES_LIB})
    endif()
elseif(WIN32)
    target_compile_definitions(litert_runtime PRIVATE WIN32_LEAN_AND_MEAN NOGDI)
    target_link_libraries(litert_runtime PRIVATE ws2_32)
elseif(UNIX)
    target_link_libraries(litert_runtime PRIVATE ${CMAKE_DL_LIBS})
endif()

# GPU support
if(LITERT_ENABLE_GPU AND LITERT_HAS_OPENCL)
    target_link_libraries(litert_runtime PRIVATE OpenCL::OpenCL)
    target_compile_definitions(litert_runtime PRIVATE LITERT_HAS_OPENCL=1)
endif()

# Add subdirectories for specialized components
add_subdirectory(accelerators)
add_subdirectory(dispatch)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_RUNTIME_TEST_SOURCES
        accelerator_test.cc
        compiled_model_test.cc
        profiler_test.cc
        tensor_buffer_conversion_test.cc
    )

    # GPU-specific tests
    if(LITERT_ENABLE_GPU)
        list(APPEND LITERT_RUNTIME_TEST_SOURCES
            gl_buffer_test.cc
            gpu_environment_test.cc
        )
    endif()

    # Create test executables
    foreach(test_source ${LITERT_RUNTIME_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_runtime
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Create alias for consistency
add_library(LiteRT::Runtime ALIAS litert_runtime)

# Export for installation
install(TARGETS litert_runtime
    EXPORT LiteRTRuntimeTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/litert/runtime
)