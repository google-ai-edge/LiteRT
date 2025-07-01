# LiteRT Tools Flags Library

# Base flags sources
set(LITERT_TOOLS_FLAGS_SOURCES
    apply_plugin_flags.cc
    common_flags.cc
    flag_types.cc
)

set(LITERT_TOOLS_FLAGS_HEADERS
    apply_plugin_flags.h
    common_flags.h
    flag_types.h
)

# Create the flags library
add_library(litert_tools_flags STATIC ${LITERT_TOOLS_FLAGS_SOURCES} ${LITERT_TOOLS_FLAGS_HEADERS})

# Set target properties
set_target_properties(litert_tools_flags PROPERTIES
    OUTPUT_NAME "litert_tools_flags"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories
target_include_directories(litert_tools_flags
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_tools_flags
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# Add vendors subdirectory
add_subdirectory(vendors)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_TOOLS_FLAGS_TEST_SOURCES
        flag_types_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_TOOLS_FLAGS_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_tools_flags
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Create alias for consistency
add_library(LiteRT::Tools::Flags ALIAS litert_tools_flags)