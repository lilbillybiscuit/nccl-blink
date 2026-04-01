/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "edmonds.h"
#include "core.h"
#include <float.h>
#include <string.h>

/*========================================================================
 * Chu-Liu/Edmonds' Algorithm (Minimum Weight Arborescence)
 *
 * Given a weighted directed graph and root vertex, find the minimum
 * total weight spanning tree rooted at root that reaches all vertices.
 * Iterative contraction approach. O(VE) per call.
 *========================================================================*/

// Find minimum incoming edge for vertex v (under current representatives)
// Returns edge index or -1 if no incoming edge exists
static int edmondsMinIncoming(int nEdges, const int* edgeSrc, const int* edgeDst,
                              const double* w, const int* repr, int v) {
  int best = -1;
  double bestW = DBL_MAX;
  for (int e = 0; e < nEdges; e++) {
    if (repr[edgeDst[e]] == v && repr[edgeSrc[e]] != v) {
      if (w[e] < bestW) {
        bestW = w[e];
        best = e;
      }
    }
  }
  return best;
}

ncclResult_t blinkEdmonds(const struct BlinkGraph* bg, int root,
                          const double* edgeWeights,
                          struct BlinkTree* tree) {
  int V = bg->nVertices;
  int E = bg->nEdges;
  if (V <= 1) {
    tree->root = root;
    tree->parent[root] = -1;
    return ncclSuccess;
  }

  // Working copies of edge endpoints and weights
  int edgeSrc[BLINK_MAX_EDGES], edgeDst[BLINK_MAX_EDGES];
  double w[BLINK_MAX_EDGES];
  for (int e = 0; e < E; e++) {
    edgeSrc[e] = bg->edges[e].src;
    edgeDst[e] = bg->edges[e].dst;
    w[e] = edgeWeights[e];
  }

  // repr[v] = representative supernode for vertex v
  int repr[BLINK_MAX_GPUS];
  for (int v = 0; v < V; v++) repr[v] = v;

  // Track which vertices are active supernodes
  int active[BLINK_MAX_GPUS];
  for (int v = 0; v < V; v++) active[v] = 1;

  // selectedEdge[v] = edge index of min incoming edge chosen for v
  int selectedEdge[BLINK_MAX_GPUS];

  // Contraction history for expansion
  struct ContractionRecord {
    int supernode;
    int cycleLen;
    int cycleMembers[BLINK_MAX_GPUS];
    int cycleEdges[BLINK_MAX_GPUS]; // selectedEdge for each cycle member
  };
  ContractionRecord history[BLINK_MAX_GPUS];
  int nContractions = 0;

  int curRoot = root;

  for (int iter = 0; iter < V; iter++) {
    // Step 1: For each non-root active vertex, find min incoming edge
    for (int v = 0; v < V; v++) {
      if (!active[v] || v == curRoot) continue;
      selectedEdge[v] = edmondsMinIncoming(E, edgeSrc, edgeDst, w, repr, v);
      if (selectedEdge[v] < 0) {
        return ncclInternalError; // disconnected from root
      }
    }

    // Step 2: Detect cycle by following selected-edge chains
    int visited[BLINK_MAX_GPUS];
    memset(visited, -1, sizeof(int) * V);

    int cycleMembers[BLINK_MAX_GPUS];
    int cycleLen = 0;

    for (int start = 0; start < V; start++) {
      if (!active[start] || start == curRoot) continue;
      int path[BLINK_MAX_GPUS];
      int pathLen = 0;
      int v = start;
      while (v != curRoot && active[v] && visited[v] == -1) {
        visited[v] = start;
        path[pathLen++] = v;
        int e = selectedEdge[v];
        v = repr[edgeSrc[e]];
      }
      if (v != curRoot && active[v] && visited[v] == start) {
        // Found a cycle containing v
        cycleLen = 0;
        int u = v;
        do {
          cycleMembers[cycleLen++] = u;
          int e = selectedEdge[u];
          u = repr[edgeSrc[e]];
        } while (u != v);
        break; // Process one cycle at a time
      }
    }

    if (cycleLen == 0) {
      // No cycles — selected edges form the arborescence
      break;
    }

    // Step 3: Contract the cycle
    int supernode = cycleMembers[0];

    ContractionRecord* rec = &history[nContractions++];
    rec->supernode = supernode;
    rec->cycleLen = cycleLen;
    for (int i = 0; i < cycleLen; i++) {
      rec->cycleMembers[i] = cycleMembers[i];
      rec->cycleEdges[i] = selectedEdge[cycleMembers[i]];
    }

    // Mark cycle members (except supernode) inactive
    for (int i = 1; i < cycleLen; i++) {
      active[cycleMembers[i]] = 0;
    }

    // Update representatives: anything pointing to a cycle member now points to supernode
    for (int v = 0; v < V; v++) {
      for (int i = 1; i < cycleLen; i++) {
        if (repr[v] == cycleMembers[i]) repr[v] = supernode;
      }
    }

    // Adjust weights for edges entering the cycle
    // For edge (u,v) entering cycle member v: w' = w - w[selectedEdge[v]]
    for (int e = 0; e < E; e++) {
      int dst = repr[edgeDst[e]];
      int src = repr[edgeSrc[e]];
      if (dst != supernode || src == supernode) continue;

      // Find which original cycle member this edge targets
      for (int j = 0; j < cycleLen; j++) {
        if (edgeDst[e] == rec->cycleMembers[j]) {
          w[e] -= w[rec->cycleEdges[j]];
          break;
        }
      }
    }
  }

  // Expand contractions to recover arborescence
  int finalParent[BLINK_MAX_GPUS];
  for (int v = 0; v < V; v++) finalParent[v] = -1;

  // Collect selected edges for active vertices
  for (int v = 0; v < V; v++) {
    if (v == curRoot || !active[v]) continue;
    int e = selectedEdge[v];
    if (e >= 0) finalParent[v] = edgeSrc[e];
  }

  // Expand contractions in reverse order
  for (int c = nContractions - 1; c >= 0; c--) {
    ContractionRecord* rec = &history[c];
    int super = rec->supernode;

    int incomingParent = finalParent[super];

    // Restore cycle edges (each cycle member's parent = selected edge source)
    for (int i = 0; i < rec->cycleLen; i++) {
      int v = rec->cycleMembers[i];
      int e = rec->cycleEdges[i];
      finalParent[v] = edgeSrc[e];
    }

    // Break the cycle at the member that receives the external incoming edge
    if (incomingParent >= 0) {
      // Find which cycle member the incoming edge actually targets
      int target = super; // default
      for (int e = 0; e < E; e++) {
        if (edgeSrc[e] == incomingParent) {
          for (int j = 0; j < rec->cycleLen; j++) {
            if (edgeDst[e] == rec->cycleMembers[j]) {
              target = rec->cycleMembers[j];
              goto found_target;
            }
          }
        }
      }
found_target:
      finalParent[target] = incomingParent;
    }
  }

  // Write output
  tree->root = root;
  for (int v = 0; v < V; v++) {
    if (v == root) {
      tree->parent[v] = -1;
    } else {
      tree->parent[v] = finalParent[v];
      if (tree->parent[v] < 0 || tree->parent[v] >= V) {
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}
