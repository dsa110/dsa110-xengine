#include <utility>
#include "dsaX_ptr.h"

dsaX_ptr::dsaX_ptr(dsaXMemoryType type, size_t size, bool pool) : type(type), size(size), pool(pool) {
  if (pool && (type != DSA_MEMORY_DEVICE && type != DSA_MEMORY_HOST_PINNED && type != DSA_MEMORY_HOST)) {    
    printf("dsaX ERROR: Memory pool not available for memory type %d", type);
    exit(0);
  }
  
  if (size > 0) {
    switch (type) {
    case DSA_MEMORY_DEVICE: device = pool ? pool_device_malloc(size) : device_malloc(size); break;
    case DSA_MEMORY_DEVICE_PINNED: device = device_pinned_malloc(size); break;
    case DSA_MEMORY_HOST: host = safe_malloc(size); break;
    case DSA_MEMORY_HOST_PINNED: host = pool ? pool_pinned_malloc(size) : pinned_malloc(size); break;
    case DSA_MEMORY_MAPPED:
      host = mapped_malloc(size);
      device = get_mapped_device_pointer(host);
      break;
    case DSA_MEMORY_MANAGED:
      host = managed_malloc(size);
      device = host;
      break;
    default:
      printf("dsaX ERROR: Unknown memory type %d", type);
      exit(0);
    }
  }
}

dsaX_ptr::dsaX_ptr(void *ptr, dsaXMemoryType type) : type(type), reference(true) {
  switch (type) {
  case DSA_MEMORY_DEVICE:
  case DSA_MEMORY_DEVICE_PINNED:
    device = ptr;
    host = nullptr;
    break;
  case DSA_MEMORY_HOST:
  case DSA_MEMORY_HOST_PINNED:
    device = nullptr;
    host = ptr;
    break;
  case DSA_MEMORY_MANAGED:
    device = ptr;
    host = ptr;
    break;
  default:
    printf("dsaX ERROR: Unsupported memory type %d", type);
    exit(0);
  }
}

dsaX_ptr &dsaX_ptr::operator=(dsaX_ptr &&other) {
  if (&other != this) {
    if (size > 0) {
      printf("dsaX ERROR: Cannot move to already initialized dsaX_ptr");
    }
    type = std::exchange(other.type, DSA_MEMORY_INVALID);
    size = std::exchange(other.size, 0);
    pool = std::exchange(other.pool, false);
    device = std::exchange(other.device, nullptr);
    host = std::exchange(other.host, nullptr);
  }
  return *this;
}

void dsaX_ptr::destroy() {
  if (size > 0) {
    switch (type) {
    case DSA_MEMORY_DEVICE: pool ? pool_device_free(device) : device_free(device); break;
    case DSA_MEMORY_DEVICE_PINNED: device_pinned_free(device); break;
    case DSA_MEMORY_HOST: host_free(host); break;
    case DSA_MEMORY_HOST_PINNED: pool ? pool_pinned_free(host) : host_free(host); break;
    case DSA_MEMORY_MAPPED: host_free(host); break;
    default:
      printf("Unknown memory type %d", type);
      exit(0);
    }
  }

  size = 0;
  device = nullptr;
  host = nullptr;
}

dsaX_ptr::~dsaX_ptr() {
  destroy();
}

void dsaX_ptr::exchange(dsaX_ptr &obj, dsaX_ptr &&new_value) {
  destroy();
  *this = std::move(obj);
  obj = std::move(new_value);
}

bool dsaX_ptr::is_device() const {
  switch (type) {
  case DSA_MEMORY_DEVICE:
  case DSA_MEMORY_DEVICE_PINNED:
  case DSA_MEMORY_MAPPED:
  case DSA_MEMORY_MANAGED: return true;
  default: return false;
  }
}

bool dsaX_ptr::is_host() const {
  switch (type) {
  case DSA_MEMORY_HOST:
  case DSA_MEMORY_HOST_PINNED:
  case DSA_MEMORY_MANAGED: return true;
  default: return false;
  }
}

void *dsaX_ptr::data() const {
  void *ptr = nullptr;

  switch (type) {
  case DSA_MEMORY_DEVICE:
  case DSA_MEMORY_DEVICE_PINNED:
  case DSA_MEMORY_MAPPED:
  case DSA_MEMORY_MANAGED: ptr = device; break;
  case DSA_MEMORY_HOST:
  case DSA_MEMORY_HOST_PINNED: ptr = host; break;
  default:
    printf("Unknown memory type %d", type);
    exit(0);
  }

  return ptr;
}

void *dsaX_ptr::data_device() const {
  if (!device) {
    printf("dsaX ERROR: Device view not defined");
    exit(0);
  }
  return device;
}

void *dsaX_ptr::data_host() const {
  if (!host) {
    printf("dsaX ERROR: Host view not defined");
    exit(0);
  }
  return host;
}

bool dsaX_ptr::is_reference() const { return reference; }

std::ostream &operator<<(std::ostream &output, const dsaX_ptr &ptr) {
  output << "{type = " << ptr.type << ", size = " << ptr.size << ", pool = " << ptr.pool
	 << ", device = " << ptr.device << ", host = " << ptr.host << ", reference = " << ptr.reference << "}";
  return output;
}
