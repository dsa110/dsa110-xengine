#include "dsaX_psrdada_utils.h"

void dsaX_dbgpu_cleanup(dada_hdu_t * in, dada_hdu_t * out)
{
  if (dada_hdu_unlock_read (in) < 0) syslog(LOG_ERR, "could not unlock read on hdu_in");
  dada_hdu_destroy (in);
  
  if (dada_hdu_unlock_write (out) < 0) syslog(LOG_ERR, "could not unlock write on hdu_out");
  dada_hdu_destroy (out);
  
} 
