# LiteRT C++ Options Libraries

# Collect source and header files
set(LITERT_CC_OPTIONS_SOURCES
    litert_cpu_options.cc
    litert_gpu_options.cc
    litert_google_tensor_options.cc
    litert_qualcomm_options.cc
    litert_mediatek_options.cc
    litert_runtime_options.cc
)

set(LITERT_CC_OPTIONS_HEADERS
    litert_cpu_options.h
    litert_gpu_options.h
    litert_google_tensor_options.h
    litert_qualcomm_options.h
    litert_mediatek_options.h
    litert_runtime_options.h
)

# CPU Options
add_library(litert_cpu_options STATIC
    litert_cpu_options.cc
    litert_cpu_options.h
)
target_include_directories(litert_cpu_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_cpu_options
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# GPU Options
add_library(litert_gpu_options STATIC
    litert_gpu_options.cc
    litert_gpu_options.h
)
target_include_directories(litert_gpu_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_gpu_options
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# Google Tensor Options
add_library(litert_google_tensor_options STATIC
    litert_google_tensor_options.cc
    litert_google_tensor_options.h
)
target_include_directories(litert_google_tensor_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_google_tensor_options
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# Qualcomm Options
add_library(litert_qualcomm_options STATIC
    litert_qualcomm_options.cc
    litert_qualcomm_options.h
)
target_include_directories(litert_qualcomm_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_qualcomm_options
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# MediaTek Options
add_library(litert_mediatek_options STATIC
    litert_mediatek_options.cc
    litert_mediatek_options.h
)
target_include_directories(litert_mediatek_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_mediatek_options
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# Runtime Options
add_library(litert_runtime_options STATIC
    litert_runtime_options.cc
    litert_runtime_options.h
)
target_include_directories(litert_runtime_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_runtime_options
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_CC_OPTIONS_TEST_SOURCES
        litert_cpu_options_test.cc
        litert_gpu_options_test.cc
        litert_runtime_options_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_CC_OPTIONS_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        
        # Link with the appropriate options library
        if(test_name MATCHES "cpu")
            target_link_libraries(${test_name} PRIVATE litert_cpu_options)
        elseif(test_name MATCHES "gpu")
            target_link_libraries(${test_name} PRIVATE litert_gpu_options)
        elseif(test_name MATCHES "runtime")
            target_link_libraries(${test_name} PRIVATE litert_runtime_options)
        endif()
        
        target_link_libraries(${test_name}
            PRIVATE
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Create aliases for consistency
add_library(LiteRT::Options::CPU ALIAS litert_cpu_options)
add_library(LiteRT::Options::GPU ALIAS litert_gpu_options)
add_library(LiteRT::Options::GoogleTensor ALIAS litert_google_tensor_options)
add_library(LiteRT::Options::Qualcomm ALIAS litert_qualcomm_options)
add_library(LiteRT::Options::MediaTek ALIAS litert_mediatek_options)
add_library(LiteRT::Options::Runtime ALIAS litert_runtime_options)