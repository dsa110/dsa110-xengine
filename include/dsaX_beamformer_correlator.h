#pragma once

// correlator function
// workflow: copy to device, reorder, stridedBatchedGemm, reorder
void dcorrelator(dmem *d);

// beamformer function
void dbeamformer(dmem * d);

