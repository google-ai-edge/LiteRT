# LiteRT Tools Vendor Flags

# Vendor-specific flag sources
set(LITERT_VENDOR_FLAGS_SOURCES
    qualcomm_flags.cc
    mediatek_flags.cc
    google_tensor_flags.cc
)

set(LITERT_VENDOR_FLAGS_HEADERS
    qualcomm_flags.h
    mediatek_flags.h
    google_tensor_flags.h
)

# Create vendor flags library
add_library(litert_tools_flags_vendors STATIC ${LITERT_VENDOR_FLAGS_SOURCES} ${LITERT_VENDOR_FLAGS_HEADERS})

# Set target properties
set_target_properties(litert_tools_flags_vendors PROPERTIES
    OUTPUT_NAME "litert_tools_flags_vendors"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories
target_include_directories(litert_tools_flags_vendors
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_tools_flags_vendors
    PUBLIC
        litert_c
        litert_tools_flags
        ${LITERT_THIRD_PARTY_LIBS}
)

# Individual vendor flag libraries (for selective linking)
add_library(litert_tools_flags_qualcomm STATIC qualcomm_flags.cc qualcomm_flags.h)
target_include_directories(litert_tools_flags_qualcomm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_tools_flags_qualcomm
    PUBLIC
        litert_c
        litert_tools_flags
        ${LITERT_THIRD_PARTY_LIBS}
)

add_library(litert_tools_flags_mediatek STATIC mediatek_flags.cc mediatek_flags.h)
target_include_directories(litert_tools_flags_mediatek
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_tools_flags_mediatek
    PUBLIC
        litert_c
        litert_tools_flags
        ${LITERT_THIRD_PARTY_LIBS}
)

add_library(litert_tools_flags_google_tensor STATIC google_tensor_flags.cc google_tensor_flags.h)
target_include_directories(litert_tools_flags_google_tensor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_tools_flags_google_tensor
    PUBLIC
        litert_c
        litert_tools_flags
        ${LITERT_THIRD_PARTY_LIBS}
)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_VENDOR_FLAGS_TEST_SOURCES
        qualcomm_flags_test.cc
        mediatek_flags_test.cc
        google_tensor_flags_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_VENDOR_FLAGS_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        string(REPLACE "_test" "" vendor_name ${test_name})
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_tools_flags_${vendor_name}
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Create aliases for consistency
add_library(LiteRT::Tools::Flags::Vendors ALIAS litert_tools_flags_vendors)
add_library(LiteRT::Tools::Flags::Qualcomm ALIAS litert_tools_flags_qualcomm)
add_library(LiteRT::Tools::Flags::MediaTek ALIAS litert_tools_flags_mediatek)
add_library(LiteRT::Tools::Flags::GoogleTensor ALIAS litert_tools_flags_google_tensor)