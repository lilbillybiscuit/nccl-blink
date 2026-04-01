/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_EDMONDS_H_
#define NCCL_EDMONDS_H_

#include "blink.h"

// Chu-Liu/Edmonds' algorithm: find the minimum total weight directed
// spanning tree (arborescence) rooted at 'root'.
//
//   bg:          directed graph (vertices + edges)
//   root:        root vertex index for the arborescence
//   edgeWeights: weight per edge (length bg->nEdges)
//   tree:        output arborescence (parent[] array)
//
// Returns ncclInternalError if the graph is not connected from root.
// Time: O(VE) per call — trivial for V <= 16.
ncclResult_t blinkEdmonds(const struct BlinkGraph* bg, int root,
                          const double* edgeWeights,
                          struct BlinkTree* tree);

#endif // NCCL_EDMONDS_H_
