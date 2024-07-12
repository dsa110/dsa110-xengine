#pragma once

#include <ostream>
#include "dsaX_malloc.h"

/**
   Object that stores a memory allocation with different views for
   host or device.  Depending on the nature of the underlying memory
   type, both views may not be defined

   type                       defined views
   DSAX_MEMORY_DEVICE         device only
   DSAX_MEMORY_DEVICE_PINNED  device only
   DSAX_MEMORY_HOST           host only
   DSAX_MEMORY_HOST_PINNED    both
   DSAX_MEMORY_MAPPED         both (pinned to host)
   DSAX_MEMORY_MANAGED        both
*/
class dsaX_ptr
{
  friend std::ostream &operator<<(std::ostream &output, const dsaX_ptr &ptr);
  dsaXMemoryType type = DSA_MEMORY_INVALID;  /** Memory type of the allocation */
  size_t size = 0;                           /** Size of the allocation */
  bool pool = false;                         /** Is the allocation is pooled */
  void *device = nullptr;                    /** Device-view of the allocation */
  void *host = nullptr;                      /** Host-view of the allocation */
  bool reference = false;                    /** Is this a reference to another allocation */

  /**
     @brief Internal deallocation routine
  */
  void destroy();

public:
  dsaX_ptr() = default;
  dsaX_ptr(dsaX_ptr &&) = default;
  dsaX_ptr &operator=(dsaX_ptr &&);
  dsaX_ptr(const dsaX_ptr &) = delete;
  dsaX_ptr &operator=(const dsaX_ptr &) = delete;

  /**
     @brief Constructor for dsaX_ptr
     @param[in] type The memory type of the allocation
     @param[in] size The size of the allocation
     @param[in] pool Whether the allocation should be in the memory pool (default is true)
  */
  dsaX_ptr(dsaXMemoryType type, size_t size, bool pool = true);

  /**
     @brief Constructor for dsaX_ptr where we are wrapping a non-owned pointer
     @param[in] ptr Raw base pointer
     @param[in] type The memory type of the allocation
  */
  dsaX_ptr(void *ptr, dsaXMemoryType type);

  /**
     @brief Destructor for the dsaX_ptr
  */
  virtual ~dsaX_ptr();

  /**
     @brief Specialized exchange function to use in place of
     std::exchange when exchanging dsaX_ptr objects: moves obj to
     *this, and moves new_value to obj
     @param[in,out] obj
     @param[in] new_value New value for obj to take
  */
  void exchange(dsaX_ptr &obj, dsaX_ptr &&new_value);

  /**
     @return Returns true if allocation is visible to the device
  */
  bool is_device() const;

  /**
     @return Returns true if allocation is visible to the host
  */
  bool is_host() const;

  /**
     Return view of the pointer.  For mapped memory we return the device view.
  */
  void *data() const;

  /**
     Return the device view of the pointer
  */
  void *data_device() const;

  /**
     Return the host view of the pointer
  */
  void *data_host() const;

  /**
     Return if the instance is a reference rather than an allocation
  */
  bool is_reference() const;
};

std::ostream &operator<<(std::ostream &output, const dsaX_ptr &ptr);

