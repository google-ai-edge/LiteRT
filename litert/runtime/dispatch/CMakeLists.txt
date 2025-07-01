# LiteRT Runtime Dispatch

set(LITERT_DISPATCH_SOURCES
    dispatch_delegate.cc
    dispatch_delegate_kernel.cc
    dispatch_opaque_options.cc
    litert_dispatch.cc
    litert_dispatch_dummy.cc
)

set(LITERT_DISPATCH_HEADERS
    dispatch_delegate_options.h
    dispatch_opaque_options.h
)

# Create the dispatch library
add_library(litert_dispatch STATIC ${LITERT_DISPATCH_SOURCES} ${LITERT_DISPATCH_HEADERS})

# Set target properties
set_target_properties(litert_dispatch PROPERTIES
    OUTPUT_NAME "litert_dispatch"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories
target_include_directories(litert_dispatch
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_dispatch
    PUBLIC
        litert_c
        litert_core
        ${LITERT_THIRD_PARTY_LIBS}
)

# Add tests if enabled
if(LITERT_BUILD_TESTS)
    # Test files (vendor-specific tests)
    set(LITERT_DISPATCH_TEST_SOURCES
        dispatch_opaque_options_test.cc
    )

    # Vendor-specific tests
    if(LITERT_ENABLE_GOOGLE_TENSOR)
        list(APPEND LITERT_DISPATCH_TEST_SOURCES
            dispatch_delegate_google_tensor_test.cc
            dispatch_delegate_cascade_google_tensor_test.cc
            dispatch_delegate_mixed_cascade_google_tensor_test.cc
        )
    endif()

    if(LITERT_ENABLE_MEDIATEK)
        list(APPEND LITERT_DISPATCH_TEST_SOURCES
            dispatch_delegate_mediatek_test.cc
            dispatch_delegate_cascade_mediatek_test.cc
            dispatch_delegate_mixed_cascade_mediatek_test.cc
        )
    endif()

    if(LITERT_ENABLE_QUALCOMM)
        list(APPEND LITERT_DISPATCH_TEST_SOURCES
            dispatch_delegate_qualcomm_test.cc
            dispatch_delegate_cascade_qualcomm_test.cc
            dispatch_delegate_mixed_cascade_qualcomm_test.cc
        )
    endif()

    # Create test executables
    foreach(test_source ${LITERT_DISPATCH_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name}
            PRIVATE
                litert_dispatch
                gtest_main
                gtest
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Create alias for consistency
add_library(LiteRT::Runtime::Dispatch ALIAS litert_dispatch)