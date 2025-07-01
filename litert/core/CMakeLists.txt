# LiteRT Core Library

# Platform-specific dynamic loading source
if(WIN32)
    set(DYNAMIC_LOADING_SOURCE dynamic_loading_windows.cc)
else()
    set(DYNAMIC_LOADING_SOURCE dynamic_loading.cc)
endif()

# Collect all source files (excluding test files)
set(LITERT_CORE_SOURCES
    build_stamp.cc
    dispatch_op_schema.cc
    ${DYNAMIC_LOADING_SOURCE}
    environment.cc
    environment_options.cc
    filesystem.cc
)

# Collect all header files
set(LITERT_CORE_HEADERS
    build_stamp.h
    dispatch_op_schema.h
    dynamic_loading.h
    environment.h
    environment_options.h
    filesystem.h
    insert_order_map.h
    options.h
    version.h
)

# Create the core library
add_library(litert_core STATIC ${LITERT_CORE_SOURCES} ${LITERT_CORE_HEADERS})

# Set target properties
set_target_properties(litert_core PROPERTIES
    OUTPUT_NAME "litert_core"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${LITERT_CORE_HEADERS}"
)

# Include directories
target_include_directories(litert_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_core
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
        Threads::Threads
    PRIVATE
        ${CMAKE_DL_LIBS}
)

# Platform-specific configurations
if(ANDROID)
    target_link_libraries(litert_core PRIVATE log android)
elseif(WIN32)
    target_compile_definitions(litert_core PRIVATE WIN32_LEAN_AND_MEAN NOGDI)
    target_link_libraries(litert_core PRIVATE ws2_32)
elseif(UNIX)
    target_link_libraries(litert_core PRIVATE ${CMAKE_DL_LIBS})
endif()

# Add subdirectories
add_subdirectory(model)
add_subdirectory(util)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_CORE_TEST_SOURCES
        build_stamp_test.cc
        dispatch_op_schema_test.cc
        dynamic_loading_test.cc
        environment_options_test.cc
        environment_test.cc
        filesystem_test.cc
        insert_order_map_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_CORE_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_core
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Create alias for consistency
add_library(LiteRT::Core ALIAS litert_core)

# Export for installation
install(TARGETS litert_core
    EXPORT LiteRTCoreTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/litert/core
)