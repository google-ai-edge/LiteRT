# LiteRT XNNPACK Accelerator

set(LITERT_XNNPACK_ACCELERATOR_SOURCES
    xnnpack_accelerator.cc
)

set(LITERT_XNNPACK_ACCELERATOR_HEADERS
    xnnpack_accelerator.h
)

# Create the XNNPACK accelerator library
add_library(litert_xnnpack_accelerator STATIC 
    ${LITERT_XNNPACK_ACCELERATOR_SOURCES} 
    ${LITERT_XNNPACK_ACCELERATOR_HEADERS}
)

# Set target properties
set_target_properties(litert_xnnpack_accelerator PROPERTIES
    OUTPUT_NAME "litert_xnnpack_accelerator"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories
target_include_directories(litert_xnnpack_accelerator
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_xnnpack_accelerator
    PUBLIC
        litert_c
        litert_accelerators
        XNNPACK
        ${LITERT_THIRD_PARTY_LIBS}
)

# Create alias for consistency
add_library(LiteRT::Runtime::Accelerators::XNNPACK ALIAS litert_xnnpack_accelerator)