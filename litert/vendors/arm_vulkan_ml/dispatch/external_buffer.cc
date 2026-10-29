// Copyright 2026 The LiteRT Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
#include "litert/vendors/arm_vulkan_ml/dispatch/external_buffer.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#include "vulkan/vulkan_android.h"
#endif
namespace litert::arm_vulkan_ml {
ExternalBuffer::~ExternalBuffer() {
  if(fd>=0)close(fd);
  if(ahb && api)api->Release(ahb);
}
std::shared_ptr<ExternalBuffer> ExternalBuffer::DmaBuf(int source, size_t size,
    size_t offset, std::string& error) {
  struct stat st{};
  if(source<0 || !size || offset>size || fstat(source,&st)!=0){error="Invalid DMA-BUF registration";return nullptr;}
  // DMA-BUF defines SEEK_END as an allocation-size query.
  auto length=lseek(source,0,SEEK_END);
  if(length<0 || static_cast<uint64_t>(length)<size){error="DMA-BUF allocation is smaller than declared buffer";return nullptr;}
  int owned=fcntl(source,F_DUPFD_CLOEXEC,0);
  if(owned<0){error="Cannot duplicate DMA-BUF descriptor";return nullptr;}
  auto result=std::make_shared<ExternalBuffer>();result->kind=Kind::kDmaBuf;
  result->fd=owned;result->size=size;result->offset=offset;
  result->identity_a=static_cast<uint64_t>(st.st_dev);result->identity_b=static_cast<uint64_t>(st.st_ino);
  return result;
}
std::shared_ptr<ExternalBuffer> ExternalBuffer::Ahb(void* buffer,size_t size,
    size_t offset,std::shared_ptr<HardwareBufferApi> api,std::string& error){
  uint64_t bytes=0,id=0;
  if(!buffer || !api || offset || !size || !api->Describe(buffer,bytes,id) || bytes<size){error="AHardwareBuffer requires a sufficiently sized GPU BLOB with zero offset";return nullptr;}
  auto result=std::make_shared<ExternalBuffer>();result->kind=Kind::kAhb;
  result->ahb=buffer;result->api=std::move(api);result->size=size;
  result->identity_a=0;result->identity_b=id;result->api->Acquire(buffer);return result;
}
VkExternalMemoryHandleTypeFlagBits ExternalBuffer::HandleType()const{
  return kind==Kind::kDmaBuf?VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
}
uint32_t ExternalBuffer::ForeignQueue()const{return kind==Kind::kAhb?VK_QUEUE_FAMILY_FOREIGN_EXT:VK_QUEUE_FAMILY_EXTERNAL;}
bool ExternalBuffer::Overlaps(const ExternalBuffer& other,size_t bytes,size_t other_bytes)const{
  if(kind!=other.kind || identity_a!=other.identity_a || identity_b!=other.identity_b)return false;
  return offset<=other.offset?other.offset-offset<bytes:offset-other.offset<other_bytes;
}
#if defined(__ANDROID__)
class AndroidApi final:public HardwareBufferApi{
 public:
  bool Describe(void* p,uint64_t& bytes,uint64_t& id)override{
    auto* b=static_cast<AHardwareBuffer*>(p);AHardwareBuffer_Desc d{};AHardwareBuffer_describe(b,&d);
    if(d.format!=AHARDWAREBUFFER_FORMAT_BLOB || d.height!=1 || d.layers!=1 || !(d.usage&AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER) || (d.usage&AHARDWAREBUFFER_USAGE_PROTECTED_CONTENT))return false;
    bytes=d.width;return AHardwareBuffer_getId(b,&id)==0;
  }
  void Acquire(void* p)override{AHardwareBuffer_acquire(static_cast<AHardwareBuffer*>(p));}
  void Release(void* p)override{AHardwareBuffer_release(static_cast<AHardwareBuffer*>(p));}
  bool Properties(VkDevice device,PFN_vkGetDeviceProcAddr get,void* p,VkDeviceSize& size,uint32_t& bits)override{
    auto fn=reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(get(device,"vkGetAndroidHardwareBufferPropertiesANDROID"));
    if(!fn)return false;VkAndroidHardwareBufferPropertiesANDROID props{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    if(fn(device,static_cast<AHardwareBuffer*>(p),&props)!=VK_SUCCESS)return false;
    size=props.allocationSize;bits=props.memoryTypeBits;return true;
  }
};
#endif
std::shared_ptr<HardwareBufferApi> AndroidHardwareBufferApi(){
#if defined(__ANDROID__)
  return std::make_shared<AndroidApi>();
#else
  return nullptr;
#endif
}
}
