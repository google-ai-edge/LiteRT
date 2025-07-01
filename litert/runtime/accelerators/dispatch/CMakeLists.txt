# LiteRT Dispatch Accelerator

set(LITERT_DISPATCH_ACCELERATOR_SOURCES
    dispatch_accelerator.cc
)

set(LITERT_DISPATCH_ACCELERATOR_HEADERS
    dispatch_accelerator.h
)

# Create the dispatch accelerator library
add_library(litert_dispatch_accelerator STATIC 
    ${LITERT_DISPATCH_ACCELERATOR_SOURCES} 
    ${LITERT_DISPATCH_ACCELERATOR_HEADERS}
)

# Set target properties
set_target_properties(litert_dispatch_accelerator PROPERTIES
    OUTPUT_NAME "litert_dispatch_accelerator"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories
target_include_directories(litert_dispatch_accelerator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_dispatch_accelerator
    PUBLIC
        litert_c
        litert_accelerators
        ${LITERT_THIRD_PARTY_LIBS}
)

# Create alias for consistency
add_library(LiteRT::Runtime::Accelerators::Dispatch ALIAS litert_dispatch_accelerator)