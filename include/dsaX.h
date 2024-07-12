#pragma once 

// Expose the use to compile time definitions,
// enums, parameters, and classes
#include "dsaX_def.h"
#include "enums.h"
#include "params.h"
#include "fast_time_domain.h"

// Use manual transpose route
// Uncomment to try new pure cuBLAS
//#define OLD_BLAS

/**
 * Initialize the library. This function will initialise
 * a device if using CUDA and any BLAS libraries that are
 * enabled, such as cublas.
 * @param[in] device_ordinal The GPU device to init
 */
void dsaXInit(int device_ordinal = -1);

/**
 * Finalize the library. This function will finalize
 * a device if using CUDA and any BLAS libraries that are
 * enabled, such as cublas. It will also dump any statistics
 * collected, such as performance metrics.
 */
void dsaXEnd();

/**
 * This function will allocate pinned device memory of the 
 * given size in bytes, and return a void pointer to that
 * memory. The user may delete the memory safely in their
 * application code.
 * @param[in] size The byte size of pinned memory to be allocated 
 *                 by dsaX.
 */
void *dsaXHostRegister(size_t size);

/**
 * This function allows the user to inspect the (4b,4b) char sized
 * complex data at byte address i on the host. If 'non-zero' is true
 * then the complex element will print only if either the real
 * or imaginary element is non-zero. Useful for checking if 
 * an array is populated.
 * @param[in] input    The (4b,4b) char input array
 * @param[in] i        The ith element of the array
 * @param[in] non-zero If true, print only elements with non-zero values
 */
void inspectPackedData(char input, int i, bool non_zero = false);
