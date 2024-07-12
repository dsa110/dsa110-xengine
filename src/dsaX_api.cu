


void qudaMemcpy_(void *dst, const void *src, size_t count, qudaMemcpyKind kind, const char *func, const char *file,
                   const char *line)
  {
    if (count == 0) return;
    QudaMem copy(dst, src, count, qudaMemcpyKindToAPI(kind), device::get_default_stream(), false, func, file, line);
  }


void dsaMemcpyAsync_(void *dst, const void *src, size_t count, dsaMemcpyKind kind, const qudaStream_t &stream,
		     const char *func, const char *file, const char *line)
  {
    if (count == 0) return;

    if (kind == qudaMemcpyDeviceToDevice) {
      QudaMem copy(dst, src, count, qudaMemcpyKindToAPI(kind), stream, true, func, file, line);
    } else {
#ifdef USE_DRIVER_API
      switch (kind) {
      case qudaMemcpyDeviceToHost:
        PROFILE(cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, get_stream(stream)), QUDA_PROFILE_MEMCPY_D2H_ASYNC);
        break;
      case qudaMemcpyHostToDevice:
        PROFILE(cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, get_stream(stream)), QUDA_PROFILE_MEMCPY_H2D_ASYNC);
        break;
      case qudaMemcpyDeviceToDevice:
        PROFILE(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, get_stream(stream)),
                QUDA_PROFILE_MEMCPY_D2D_ASYNC);
        break;
      case qudaMemcpyDefault:
        PROFILE(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, get_stream(stream)),
                QUDA_PROFILE_MEMCPY_DEFAULT_ASYNC);
        break;
      default: errorQuda("Unsupported cuMemcpyTypeAsync %d", kind);
      }
#else
      PROFILE(cudaMemcpyAsync(dst, src, count, qudaMemcpyKindToAPI(kind), get_stream(stream)),
              kind == qudaMemcpyDeviceToHost ? QUDA_PROFILE_MEMCPY_D2H_ASYNC : QUDA_PROFILE_MEMCPY_H2D_ASYNC);
#endif
    }
  }
