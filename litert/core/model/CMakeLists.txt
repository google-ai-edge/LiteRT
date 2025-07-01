# LiteRT Core Model Library

# Collect all source files (excluding test files)
set(LITERT_CORE_MODEL_SOURCES
    flatbuffer_to_litert.cc
    graph_validation.cc
    litert_to_flatbuffer.cc
    model.cc
    model_buffer.cc
    model_graph.cc
    model_load.cc
    model_serialize.cc
)

# Collect all header files
set(LITERT_CORE_MODEL_HEADERS
    buffer_manager.h
    flatbuffer_to_litert.h
    graph_validation.h
    ir_allocator.h
    litert_to_flatbuffer.h
    model.h
    model_buffer.h
    model_graph.h
    model_load.h
    model_serialize.h
)

# Create the model library
add_library(litert_core_model STATIC ${LITERT_CORE_MODEL_SOURCES} ${LITERT_CORE_MODEL_HEADERS})

# Set target properties
set_target_properties(litert_core_model PROPERTIES
    OUTPUT_NAME "litert_core_model"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${LITERT_CORE_MODEL_HEADERS}"
)

# Include directories
target_include_directories(litert_core_model
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_core_model
    PUBLIC
        litert_c
        litert_core
        ${LITERT_THIRD_PARTY_LIBS}
        Threads::Threads
)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files
    set(LITERT_CORE_MODEL_TEST_SOURCES
        buffer_manager_test.cc
        flatbuffer_to_litert_test.cc
        ir_allocator_test.cc
        litert_to_flatbuffer_test.cc
        model_buffer_test.cc
        model_file_test.cc
        model_graph_test.cc
        model_test.cc
    )

    # Create test executables
    foreach(test_source ${LITERT_CORE_MODEL_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_core_model
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()

    # Test utility library
    add_library(model_file_test_util STATIC model_file_test_util.cc model_file_test_util.h)
    target_link_libraries(model_file_test_util
        PUBLIC
            litert_core_model
    )
    target_include_directories(model_file_test_util
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# Create alias for consistency
add_library(LiteRT::Core::Model ALIAS litert_core_model)

# Export for installation
install(TARGETS litert_core_model
    EXPORT LiteRTCoreModelTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/litert/core/model
)