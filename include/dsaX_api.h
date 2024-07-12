#pragma once

#include <string>

#include "enums.h"

#define STRINGIFY__(x) #x
#define __STRINGIFY__(x) STRINGIFY__(x)

/**
   @brief Wrapper around cudaMemcpy or driver API equivalent
   @param[out] dst Destination pointer
   @param[in] src Source pointer
   @param[in] count Size of transfer
   @param[in] kind Type of memory copy
*/
void dsaXMemcpy_(void *dst, const void *src, size_t count, dsaXMemcpyKind kind, const char *func, const char *file,
		 const char *line);

/**
   @brief Wrapper around cudaMemcpyAsync or driver API equivalent
   @param[out] dst Destination pointer
   @param[in] src Source pointer
   @param[in] count Size of transfer
   @param[in] kind Type of memory copy
   @param[in] stream Stream to issue copy
*/
void dsaXMemcpyAsync_(void *dst, const void *src, size_t count, dsaXMemcpyKind kind, const cudaStream_t &stream,
		      const char *func, const char *file, const char *line);


#define dsaXMemcpy(dst, src, count, kind)                                                                              \
  ::dsaXMemcpy_(dst, src, count, kind, __func__, file_name(__FILE__), __STRINGIFY__(__LINE__))

#define dsaXMemcpyAsync(dst, src, count, kind, stream)                                                                 \
  ::dsaXMemcpyAsync_(dst, src, count, kind, stream, __func__, file_name(__FILE__), __STRINGIFY__(__LINE__))
