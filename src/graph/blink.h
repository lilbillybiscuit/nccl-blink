/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_BLINK_H_
#define NCCL_BLINK_H_

#include "graph.h"
#include "topo.h"

// Main entry point: compute Blink spanning tree packing for tree channels.
// Replaces ncclTopoCompute for tree graphs when NCCL_BLINK=1.
//   system: NCCL topology system (read-only)
//   graph:  output ncclTopoGraph (will be populated with tree channels)
//   nChannelsNeeded: number of channels to produce (typically ringGraph->nChannels)
ncclResult_t ncclBlinkCompute(struct ncclTopoSystem* system,
                              struct ncclTopoGraph* graph,
                              int nChannelsNeeded);

/*------------------------------------------------------------------------
 * Internal data structures
 *------------------------------------------------------------------------*/

#define BLINK_MAX_GPUS 16
#define BLINK_MAX_EDGES (BLINK_MAX_GPUS * BLINK_MAX_GPUS)
#define BLINK_MAX_TREES MAXCHANNELS

struct BlinkEdge {
  int src;
  int dst;
  float capacity;   // link bandwidth in GB/s
};

struct BlinkGraph {
  int nVertices;
  int nEdges;
  BlinkEdge edges[BLINK_MAX_EDGES];
  int gpuRanks[BLINK_MAX_GPUS];  // vertex index -> NCCL rank
  int edgeIndex[BLINK_MAX_GPUS][BLINK_MAX_GPUS]; // edgeIndex[src][dst] -> edge index, -1 if none
};

struct BlinkTree {
  int root;
  int parent[BLINK_MAX_GPUS];    // parent[v] for arborescence, parent[root] = -1
  float weight;                   // assigned rate for this tree
};

struct BlinkPacking {
  int nTrees;
  BlinkTree trees[BLINK_MAX_TREES];
  float totalRate;
};

#endif // NCCL_BLINK_H_
