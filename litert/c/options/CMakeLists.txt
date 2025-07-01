# LiteRT C Options Libraries

# Collect source and header files
set(LITERT_C_OPTIONS_SOURCES
    litert_cpu_options.cc
    litert_gpu_options.cc
    litert_google_tensor_options.cc
    litert_mediatek_options.cc
    litert_qualcomm_options.cc
    litert_runtime_options.cc
)

set(LITERT_C_OPTIONS_HEADERS
    litert_cpu_options.h
    litert_gpu_options.h
    litert_google_tensor_options.h
    litert_google_tensor_options_type.h
    litert_mediatek_options.h
    litert_qualcomm_options.h
    litert_runtime_options.h
)

# Create individual option libraries for selective linking
add_library(litert_c_cpu_options STATIC
    litert_cpu_options.cc
    litert_cpu_options.h
)
target_include_directories(litert_c_cpu_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_c_cpu_options
    PUBLIC
        ${LITERT_THIRD_PARTY_LIBS}
)

add_library(litert_c_gpu_options STATIC
    litert_gpu_options.cc
    litert_gpu_options.h
)
target_include_directories(litert_c_gpu_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_c_gpu_options
    PUBLIC
        ${LITERT_THIRD_PARTY_LIBS}
)

add_library(litert_c_google_tensor_options STATIC
    litert_google_tensor_options.cc
    litert_google_tensor_options.h
    litert_google_tensor_options_type.h
)
target_include_directories(litert_c_google_tensor_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_c_google_tensor_options
    PUBLIC
        ${LITERT_THIRD_PARTY_LIBS}
)

add_library(litert_c_mediatek_options STATIC
    litert_mediatek_options.cc
    litert_mediatek_options.h
)
target_include_directories(litert_c_mediatek_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_c_mediatek_options
    PUBLIC
        ${LITERT_THIRD_PARTY_LIBS}
)

add_library(litert_c_qualcomm_options STATIC
    litert_qualcomm_options.cc
    litert_qualcomm_options.h
)
target_include_directories(litert_c_qualcomm_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_c_qualcomm_options
    PUBLIC
        ${LITERT_THIRD_PARTY_LIBS}
)

add_library(litert_c_runtime_options STATIC
    litert_runtime_options.cc
    litert_runtime_options.h
)
target_include_directories(litert_c_runtime_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_c_runtime_options
    PUBLIC
        ${LITERT_THIRD_PARTY_LIBS}
)

# Combined options library
add_library(litert_c_options STATIC ${LITERT_C_OPTIONS_SOURCES} ${LITERT_C_OPTIONS_HEADERS})
target_include_directories(litert_c_options
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_c_options
    PUBLIC
        litert_c_cpu_options
        litert_c_gpu_options
        litert_c_google_tensor_options
        litert_c_mediatek_options
        litert_c_qualcomm_options
        litert_c_runtime_options
        ${LITERT_THIRD_PARTY_LIBS}
)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_C_OPTIONS_TEST_SOURCES
        litert_cpu_options_test.cc
        litert_gpu_options_test.cc
        litert_google_tensor_options_test.cc
        litert_mediatek_options_test.cc
        litert_qualcomm_options_test.cc
        litert_runtime_options_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_C_OPTIONS_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        string(REPLACE "_test" "" option_name ${test_name})
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_c_${option_name}
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Create aliases for consistency
add_library(LiteRT::C::Options ALIAS litert_c_options)
add_library(LiteRT::C::Options::CPU ALIAS litert_c_cpu_options)
add_library(LiteRT::C::Options::GPU ALIAS litert_c_gpu_options)
add_library(LiteRT::C::Options::GoogleTensor ALIAS litert_c_google_tensor_options)
add_library(LiteRT::C::Options::MediaTek ALIAS litert_c_mediatek_options)
add_library(LiteRT::C::Options::Qualcomm ALIAS litert_c_qualcomm_options)
add_library(LiteRT::C::Options::Runtime ALIAS litert_c_runtime_options)