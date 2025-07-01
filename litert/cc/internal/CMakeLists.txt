# LiteRT C++ Internal API Libraries
# These are internal implementation details that provide additional
# functionality on top of the core C++ API

# Options subdirectory
add_subdirectory(../options options)

# Create internal library targets for different components
# These are header-only or implementations that are referenced in the Bazel BUILD files

# litert_dispatch_delegate (header-only)
add_library(litert_dispatch_delegate INTERFACE)
target_include_directories(litert_dispatch_delegate
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(litert_dispatch_delegate
    INTERFACE
        litert_c
)

# litert_environment (header-only)
add_library(litert_environment INTERFACE)
target_include_directories(litert_environment
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(litert_environment
    INTERFACE
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# litert_custom_op_kernel
add_library(litert_custom_op_kernel STATIC
    ../litert_custom_op_kernel.cc
)
target_include_directories(litert_custom_op_kernel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_custom_op_kernel
    PUBLIC
        litert_c
        ${LITERT_THIRD_PARTY_LIBS}
)

# litert_event (header-only)
add_library(litert_event INTERFACE)
target_include_directories(litert_event
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(litert_event
    INTERFACE
        litert_c
)

# litert_tensor_buffer
add_library(litert_tensor_buffer STATIC
    ../litert_tensor_buffer.cc
)
target_include_directories(litert_tensor_buffer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_tensor_buffer
    PUBLIC
        litert_c
        litert_event
        ${LITERT_THIRD_PARTY_LIBS}
)

# Add OpenCL support if available
if(LITERT_HAS_OPENCL)
    target_link_libraries(litert_tensor_buffer PRIVATE OpenCL::OpenCL)
endif()

# litert_tensor_buffer_requirements (header-only)
add_library(litert_tensor_buffer_requirements INTERFACE)
target_include_directories(litert_tensor_buffer_requirements
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(litert_tensor_buffer_requirements
    INTERFACE
        litert_c
)

# litert_compiled_model
add_library(litert_compiled_model STATIC
    ../litert_compiled_model.cc
)
target_include_directories(litert_compiled_model
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${LITERT_THIRD_PARTY_INCLUDES}
)
target_link_libraries(litert_compiled_model
    PUBLIC
        litert_c
        litert_custom_op_kernel
        litert_environment
        litert_options
        litert_tensor_buffer
        ${LITERT_THIRD_PARTY_LIBS}
)

# litert_options (header-only)
add_library(litert_options INTERFACE)
target_include_directories(litert_options
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(litert_options
    INTERFACE
        litert_c
        litert_custom_op_kernel
)

# Create aliases for consistency
add_library(LiteRT::Internal::DispatchDelegate ALIAS litert_dispatch_delegate)
add_library(LiteRT::Internal::Environment ALIAS litert_environment)
add_library(LiteRT::Internal::CustomOpKernel ALIAS litert_custom_op_kernel)
add_library(LiteRT::Internal::Event ALIAS litert_event)
add_library(LiteRT::Internal::TensorBuffer ALIAS litert_tensor_buffer)
add_library(LiteRT::Internal::TensorBufferRequirements ALIAS litert_tensor_buffer_requirements)
add_library(LiteRT::Internal::CompiledModel ALIAS litert_compiled_model)
add_library(LiteRT::Internal::Options ALIAS litert_options)