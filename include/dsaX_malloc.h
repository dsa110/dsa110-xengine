#pragma once

#include <iostream>
#include <vector>
#include <unistd.h>   // for getpagesize()
#include <execinfo.h> // for backtrace
#include <map>        // for std::map

#include <dsaX.h>
#include <enums.h>


using namespace std;

// strip path from __FILE__
// DMH: Place somewhere more sensible when working
constexpr const char *str_end(const char *str) { return *str ? str_end(str + 1) : str; }
constexpr bool str_slant(const char *str) { return *str == '/' ? true : (*str ? str_slant(str + 1) : false); }
constexpr const char *r_slant(const char *str) { return *str == '/' ? (str + 1) : r_slant(str - 1); }
constexpr const char *file_name(const char *str) { return str_slant(str) ? r_slant(str_end(str)) : str; }

// Define wrappers around function. May wish to place <function>_
// methods in a dsaX namespace later
void *pinned_malloc_(const char *func, const char *file, int line, size_t size);
#define pinned_malloc(size) pinned_malloc_(__func__, file_name(__FILE__), __LINE__, size)

void *device_malloc_(const char *func, const char *file, int line, size_t size);
#define device_malloc(size) device_malloc_(__func__, file_name(__FILE__), __LINE__, size)

void *device_pinned_malloc_(const char *func, const char *file, int line, size_t size);
#define device_pinned_malloc(size) device_pinned_malloc_(__func__, file_name(__FILE__), __LINE__, size)

void *safe_malloc_(const char *func, const char *file, int line, size_t size);
#define safe_malloc(size) safe_malloc_(__func__, file_name(__FILE__), __LINE__, size)

void *mapped_malloc_(const char *func, const char *file, int line, size_t size);
#define mapped_malloc(size) mapped_malloc_(__func__, file_name(__FILE__), __LINE__, size)

void *managed_malloc_(const char *func, const char *file, int line, size_t size);
#define managed_malloc(size) managed_malloc_(__func__, file_name(__FILE__), __LINE__, size)

void managed_free_(const char *func, const char *file, int line, void *ptr);
#define managed_free(ptr) managed_free_(__func__, file_name(__FILE__), __LINE__, ptr)

void device_free_(const char *func, const char *file, int line, void *ptr);
#define device_free(ptr) device_free_(__func__, file_name(__FILE__), __LINE__, ptr)

void device_pinned_free_(const char *func, const char *file, int line, void *ptr);
#define device_pinned_free(ptr) device_pinned_free_(__func__, file_name(__FILE__), __LINE__, ptr)

void host_free_(const char *func, const char *file, int line, void *ptr);
#define host_free(ptr) host_free_(__func__, file_name(__FILE__), __LINE__, ptr)

/*
  @brief Get device view of a host-mapped pointer
*/
void *get_mapped_device_pointer_(const char *func, const char *file, int line, const void *ptr);
#define get_mapped_device_pointer(ptr) get_mapped_device_pointer_(__func__, file_name(__FILE__), __LINE__, ptr)

// Create a mem_pool namespace to differentiate
// bewtween regular memory management methods
// and those utilising memory pooling
namespace mem_pool {

  /**
     @brief Initialize the memory pool allocator
  */
  void init();
  
  /**
     @brief Allocate device-memory.  If free pre-existing allocation exists
     reuse this.
     @param size Size of allocation
     @return Pointer to allocated memory
  */
  void *device_malloc_(const char *func, const char *file, int line, size_t size);
  
  /**
     @brief Virtual free of pinned-memory allocation.
     @param ptr Pointer to be (virtually) freed
  */
  void device_free_(const char *func, const char *file, int line, void *ptr);
  
  /**
     @brief Allocate pinned-memory.
     If a free pre-existing allocation exists, reuse this.
     @param size Size of allocation
     @return Pointer to allocated memory
  */
  void *pinned_malloc_(const char *func, const char *file, int line, size_t size);
  
  /**
     @brief Virtual free of pinned-memory allocation.
     @param ptr Pointer to be (virtually) freed
  */
  void pinned_free_(const char *func, const char *file, int line, void *ptr);

  /**
     @brief Free all outstanding device-memory allocations.
  */
  void flush_device();
  
  /**
     @brief Free all outstanding pinned-memory allocations.
  */
  void flush_pinned();  
}

#define pool_device_malloc(size) mem_pool::device_malloc_(__func__, __FILE__, __LINE__, size)
#define pool_device_free(ptr) mem_pool::device_free_(__func__, __FILE__, __LINE__, ptr)
#define pool_pinned_malloc(size) mem_pool::pinned_malloc_(__func__, __FILE__, __LINE__, size)
#define pool_pinned_free(ptr) mem_pool::pinned_free_(__func__, __FILE__, __LINE__, ptr)

