# LiteRT Runtime Accelerators

# Base accelerator sources
set(LITERT_ACCELERATORS_SOURCES
    auto_registration.cc
)

set(LITERT_ACCELERATORS_HEADERS
    accelerator_implementation_helper.h
    auto_registration.h
)

# Create the accelerators library
add_library(litert_accelerators STATIC ${LITERT_ACCELERATORS_SOURCES} ${LITERT_ACCELERATORS_HEADERS})

# Set target properties
set_target_properties(litert_accelerators PROPERTIES
    OUTPUT_NAME "litert_accelerators"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Include directories
target_include_directories(litert_accelerators
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LITERT_THIRD_PARTY_INCLUDES}
)

# Link dependencies
target_link_libraries(litert_accelerators
    PUBLIC
        litert_c
        litert_core
        ${LITERT_THIRD_PARTY_LIBS}
)

# Add subdirectories for specific accelerators
add_subdirectory(dispatch)
add_subdirectory(xnnpack)

# Create alias for consistency
add_library(LiteRT::Runtime::Accelerators ALIAS litert_accelerators)