# LiteRT Core Util Library

# Collect all source files (excluding test files)
set(LITERT_CORE_UTIL_SOURCES
    flatbuffer_tools.cc
    tensor_type_util.cc
)

# Collect all header files
set(LITERT_CORE_UTIL_HEADERS
    flatbuffer_tools.h
    tensor_type_util.h
    to_string.h
)

# Create the util library
add_library(litert_core_util STATIC ${LITERT_CORE_UTIL_SOURCES} ${LITERT_CORE_UTIL_HEADERS})

# Set target properties
set_target_properties(litert_core_util PROPERTIES
    OUTPUT_NAME "litert_core_util"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${LITERT_CORE_UTIL_HEADERS}"
)

# Include directories
target_include_directories(litert_core_util
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_core_util
    PUBLIC
        litert_c
        litert_core
        ${LITERT_THIRD_PARTY_LIBS}
        Threads::Threads
)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_CORE_UTIL_TEST_SOURCES
        flatbuffer_tools_test.cc
        tensor_type_util_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_CORE_UTIL_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_core_util
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Create alias for consistency
add_library(LiteRT::Core::Util ALIAS litert_core_util)

# Export for installation
install(TARGETS litert_core_util
    EXPORT LiteRTCoreUtilTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/litert/core/util
)