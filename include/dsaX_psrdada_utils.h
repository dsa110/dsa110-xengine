#pragma once

#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_def.h"
#include "dsaX_enums.h"

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);

int dada_bind_thread_to_core(int core);
