/*************************************************************************
 * Alex/Micah/Bill
 *************************************************************************/

#include "blink.h"
#include "edmonds.h"
#include "comm.h"
#include "core.h"
#include "graph.h"
#include "topo.h"
#include <math.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <time.h>

// Return elapsed microseconds between two timespecs
static inline double blinkElapsedUs(struct timespec* start, struct timespec* end) {
  return (double)(end->tv_sec - start->tv_sec) * 1e6 +
         (double)(end->tv_nsec - start->tv_nsec) / 1e3;
}

/*========================================================================
 * Section 1: Graph Extraction
 *========================================================================*/

// Find the index of a GPU node by pointer within the GPU node set
static int blinkGpuIndex(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  return (int)(node - system->nodes[GPU].nodes);
}

// Extract directed GPU interconnect graph from ncclTopoSystem.
// Only NVLink edges are considered (intra-node).
static ncclResult_t blinkExtractGraph(struct ncclTopoSystem* system,
                                      struct BlinkGraph* bg) {
  memset(bg, 0, sizeof(struct BlinkGraph));
  memset(bg->edgeIndex, -1, sizeof(bg->edgeIndex));
  int nGpus = system->nodes[GPU].count;
  if (nGpus > BLINK_MAX_GPUS) {
    WARN("Blink: too many GPUs (%d > %d)", nGpus, BLINK_MAX_GPUS);
    return ncclInternalError;
  }
  bg->nVertices = nGpus;

  for (int i = 0; i < nGpus; i++) {
    bg->gpuRanks[i] = system->nodes[GPU].nodes[i].gpu.rank;
  }

  for (int i = 0; i < nGpus; i++) {
    struct ncclTopoNode* gpu = &system->nodes[GPU].nodes[i];
    for (int l = 0; l < gpu->nlinks; l++) {
      struct ncclTopoLink* link = &gpu->links[l];
      if (link->type != LINK_NVL) continue;
      if (link->remNode->type != GPU) continue;
      int j = blinkGpuIndex(system, link->remNode);
      if (j < 0 || j >= nGpus) continue;

      // Multiple NVLinks between same pair get summed
      int found = bg->edgeIndex[i][j];
      if (found >= 0) {
        bg->edges[found].capacity += link->bw;
      } else {
        if (bg->nEdges >= BLINK_MAX_EDGES) {
          WARN("Blink: too many edges");
          return ncclInternalError;
        }
        bg->edgeIndex[i][j] = bg->nEdges;
        bg->edges[bg->nEdges].src = i;
        bg->edges[bg->nEdges].dst = j;
        bg->edges[bg->nEdges].capacity = link->bw;
        bg->nEdges++;
      }
    }
  }

  INFO(NCCL_GRAPH, "Blink: extracted graph with %d GPUs, %d directed edges",
       bg->nVertices, bg->nEdges);

#ifdef DEBUG
  for (int i = 0; i < nGpus; i++) {
    for (int j = 0; j < nGpus; j++) {
      int e = bg->edgeIndex[i][j];
      if (e >= 0) {
        assert(e < bg->nEdges && bg->edges[e].src == i && bg->edges[e].dst == j);
      }
    }
  }
#endif

  return ncclSuccess;
}

/*========================================================================
 * Section 2: MWU (Multiplicative Weight Update) Tree Packing
 *========================================================================*/

// Compute total weight of a tree under given edge weights
static double blinkTreeWeight(const struct BlinkGraph* bg, const struct BlinkTree* tree,
                              const double* edgeWeights) {
  double total = 0.0;
  for (int v = 0; v < bg->nVertices; v++) {
    if (tree->parent[v] < 0) continue;
    int e = bg->edgeIndex[tree->parent[v]][v];
    if (e >= 0) total += edgeWeights[e];
  }
  return total;
}

// Compute bottleneck capacity of a tree (minimum edge capacity along tree edges)
static float blinkTreeBottleneck(const struct BlinkGraph* bg, const struct BlinkTree* tree) {
  float bottleneck = FLT_MAX;
  for (int v = 0; v < bg->nVertices; v++) {
    if (tree->parent[v] < 0) continue;
    int e = bg->edgeIndex[tree->parent[v]][v];
    if (e >= 0 && bg->edges[e].capacity < bottleneck) {
      bottleneck = bg->edges[e].capacity;
    }
  }
  return bottleneck;
}

static ncclResult_t blinkMWU(const struct BlinkGraph* bg, double epsilon,
                             struct BlinkPacking* packing) {
  int V = bg->nVertices;
  int E = bg->nEdges;

  memset(packing, 0, sizeof(struct BlinkPacking));

  if (V <= 1 || E == 0) return ncclSuccess;

  // Initialize edge weights: w[e] = 1 / capacity[e]
  double w[BLINK_MAX_EDGES];
  for (int e = 0; e < E; e++) {
    w[e] = 1.0 / (double)bg->edges[e].capacity;
  }

  int maxIter = (int)ceil(log((double)E) / (epsilon * epsilon));
  if (maxIter > 1000) maxIter = 1000;
  if (maxIter < 10) maxIter = 10;

  INFO(NCCL_GRAPH, "Blink MWU: %d vertices, %d edges, epsilon=%.2f, maxIter=%d",
       V, E, epsilon, maxIter);

  for (int iter = 0; iter < maxIter; iter++) {
    // Try each vertex as root, find the best min-weight arborescence
    struct BlinkTree bestTree;
    double bestWeight = DBL_MAX;
    int bestRoot = -1;

    for (int r = 0; r < V; r++) {
      struct BlinkTree candidate;
      candidate.root = r;
      candidate.weight = 0;
      ncclResult_t ret = blinkEdmonds(bg, r, w, &candidate);
      if (ret != ncclSuccess) continue; // disconnected from this root

      double tw = blinkTreeWeight(bg, &candidate, w);
      if (tw < bestWeight) {
        bestWeight = tw;
        memcpy(&bestTree, &candidate, sizeof(struct BlinkTree));
        bestRoot = r;
      }
    }

    if (bestRoot < 0) break;

    // Record this tree. Weights are not meaningful here; blinkRefine
    // recomputes them via greedy bottleneck selection.
    if (packing->nTrees < BLINK_MAX_TREES) {
      memcpy(&packing->trees[packing->nTrees], &bestTree, sizeof(struct BlinkTree));
      packing->nTrees++;
    }

    // Update edge weights multiplicatively
    for (int v = 0; v < V; v++) {
      if (bestTree.parent[v] < 0) continue;
      int e = bg->edgeIndex[bestTree.parent[v]][v];
      if (e >= 0) {
        w[e] *= (1.0 + epsilon / (double)bg->edges[e].capacity);
      }
    }
  }

  INFO(NCCL_GRAPH, "Blink MWU: found %d candidate trees", packing->nTrees);
  return ncclSuccess;
}

/*========================================================================
 * Section 3: Greedy Refinement
 *========================================================================*/

static int blinkTreesEqual(const struct BlinkTree* a, const struct BlinkTree* b, int nVerts) {
  return a->root == b->root && memcmp(a->parent, b->parent, sizeof(int) * nVerts) == 0;
}

static ncclResult_t blinkRefine(const struct BlinkGraph* bg,
                                struct BlinkPacking* packing,
                                int targetCount) {
  int V = bg->nVertices;
  int E = bg->nEdges;

  if (packing->nTrees == 0 || targetCount <= 0) return ncclSuccess;

  // Deduplicate trees
  int uniqueCount = 0;
  struct BlinkTree unique[BLINK_MAX_TREES];
  int occurrence[BLINK_MAX_TREES];
  memset(occurrence, 0, sizeof(occurrence));

  for (int t = 0; t < packing->nTrees; t++) {
    int found = -1;
    for (int u = 0; u < uniqueCount; u++) {
      if (blinkTreesEqual(&packing->trees[t], &unique[u], V)) {
        found = u;
        break;
      }
    }
    if (found >= 0) {
      occurrence[found]++;
    } else if (uniqueCount < BLINK_MAX_TREES) {
      memcpy(&unique[uniqueCount], &packing->trees[t], sizeof(struct BlinkTree));
      occurrence[uniqueCount] = 1;
      uniqueCount++;
    }
  }

  INFO(NCCL_GRAPH, "Blink refine: %d unique trees from %d candidates, target=%d channels",
       uniqueCount, packing->nTrees, targetCount);

  // Greedy selection on residual capacities
  float residual[BLINK_MAX_EDGES];
  for (int e = 0; e < E; e++) {
    residual[e] = bg->edges[e].capacity;
  }

  struct BlinkTree selected[BLINK_MAX_TREES];
  int nSelected = 0;

  for (int round = 0; round < targetCount && round < BLINK_MAX_TREES; round++) {
    int bestIdx = -1;
    float bestBw = 0;

    for (int u = 0; u < uniqueCount; u++) {
      float bw = FLT_MAX;
      bool feasible = true;
      for (int v = 0; v < V; v++) {
        if (unique[u].parent[v] < 0) continue;
        int e = bg->edgeIndex[unique[u].parent[v]][v];
        if (e < 0) { feasible = false; break; }
        if (residual[e] < bg->edges[e].capacity * 1e-4f) {
          feasible = false; break;
        } else if (residual[e] < bw) {
          bw = residual[e];
        }
      }
      if (feasible && bw > bestBw) {
        bestBw = bw;
        bestIdx = u;
      }
    }

    if (bestIdx < 0) break;  // no more feasible trees

    memcpy(&selected[nSelected], &unique[bestIdx], sizeof(struct BlinkTree));
    selected[nSelected].weight = bestBw;
    nSelected++;

    // Subtract usage from residual
    for (int v = 0; v < V; v++) {
      if (unique[bestIdx].parent[v] < 0) continue;
      int e = bg->edgeIndex[unique[bestIdx].parent[v]][v];
      if (e >= 0) {
        residual[e] -= bestBw;
        if (residual[e] < 0) residual[e] = 0;
      }
    }
  }

  if (nSelected == 0) {
    WARN("Blink: refine found no feasible trees");
    return ncclInternalError;
  }

  // Fill remaining channels by duplicating with zero weight
  if (nSelected < targetCount) {
    INFO(NCCL_GRAPH, "Blink refine: padding %d zero-weight channels (had %d, need %d)",
         targetCount - nSelected, nSelected, targetCount);
    int base = nSelected;
    while (nSelected < targetCount && nSelected < BLINK_MAX_TREES) {
      memcpy(&selected[nSelected], &selected[nSelected % base], sizeof(struct BlinkTree));
      selected[nSelected].weight = 0;
      nSelected++;
    }
  }

  packing->nTrees = nSelected;
  packing->totalRate = 0;
  for (int t = 0; t < nSelected; t++) {
    memcpy(&packing->trees[t], &selected[t], sizeof(struct BlinkTree));
    packing->totalRate += selected[t].weight;
  }

  INFO(NCCL_GRAPH, "Blink refine: selected %d trees, total rate=%.2f GB/s",
       packing->nTrees, packing->totalRate);
  return ncclSuccess;
}

/*========================================================================
 * Section 4: Tree-to-Chain Conversion
 *========================================================================*/

static ncclResult_t blinkTreeToChain(const struct BlinkTree* tree, int nVerts,
                                     const int* gpuRanks, int* chain) {
  int children[BLINK_MAX_GPUS][BLINK_MAX_GPUS];
  int nChildren[BLINK_MAX_GPUS];
  memset(nChildren, 0, sizeof(int) * nVerts);

  for (int v = 0; v < nVerts; v++) {
    if (tree->parent[v] >= 0 && tree->parent[v] != v) {
      int p = tree->parent[v];
      children[p][nChildren[p]++] = v;
    }
  }

  // DFS from root
  int stack[BLINK_MAX_GPUS];
  int stackTop = 0;
  stack[stackTop++] = tree->root;
  int pos = 0;

  while (stackTop > 0 && pos < nVerts) {
    int v = stack[--stackTop];
    chain[pos++] = gpuRanks[v];
    for (int c = nChildren[v] - 1; c >= 0; c--) {
      stack[stackTop++] = children[v][c];
    }
  }

  if (pos != nVerts) {
    WARN("Blink: DFS produced %d vertices, expected %d", pos, nVerts);
    return ncclInternalError;
  }

  return ncclSuccess;
}

/*========================================================================
 * Section 5: Main Entry Point
 *========================================================================*/

ncclResult_t ncclBlinkCompute(struct ncclTopoSystem* system,
                              struct ncclTopoGraph* graph,
                              int nChannelsNeeded) {
  int nGpus = system->nodes[GPU].count;

  if (nGpus <= 1 || nChannelsNeeded <= 0) {
    INFO(NCCL_GRAPH, "Blink: nGpus=%d nChannels=%d, falling back to default",
         nGpus, nChannelsNeeded);
    graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    graph->minChannels = nChannelsNeeded;
    graph->maxChannels = nChannelsNeeded;
    return ncclTopoCompute(system, graph);
  }

  struct timespec t0, t1, t2, t3, t4;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // Step 1: Extract directed graph
  struct BlinkGraph bg;
  ncclResult_t ret = blinkExtractGraph(system, &bg);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (ret != ncclSuccess || bg.nEdges == 0) {
    INFO(NCCL_GRAPH, "Blink: no NVLink edges, falling back to default");
    graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    graph->minChannels = nChannelsNeeded;
    graph->maxChannels = nChannelsNeeded;
    return ncclTopoCompute(system, graph);
  }

  // Step 2: Run MWU to find candidate trees
  struct BlinkPacking packing;
  double epsilon = (bg.nVertices > 8) ? 0.05 : 0.1;

  ret = blinkMWU(&bg, epsilon, &packing);
  clock_gettime(CLOCK_MONOTONIC, &t2);
  if (ret != ncclSuccess || packing.nTrees == 0) {
    INFO(NCCL_GRAPH, "Blink: MWU failed or found no trees, falling back");
    graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    graph->minChannels = nChannelsNeeded;
    graph->maxChannels = nChannelsNeeded;
    return ncclTopoCompute(system, graph);
  }

  // Step 3: Refine — select targetCount trees
  ret = blinkRefine(&bg, &packing, nChannelsNeeded);
  clock_gettime(CLOCK_MONOTONIC, &t3);
  if (ret != ncclSuccess || packing.nTrees == 0) {
    INFO(NCCL_GRAPH, "Blink: refinement failed, falling back");
    graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
    graph->minChannels = nChannelsNeeded;
    graph->maxChannels = nChannelsNeeded;
    return ncclTopoCompute(system, graph);
  }

  // Step 4: Convert trees to ncclTopoGraph format
  graph->id = 1;
  graph->pattern = NCCL_TOPO_PATTERN_TREE;
  graph->crossNic = 0;
  graph->collNet = 0;
  graph->nChannels = packing.nTrees;
  graph->sameChannels = 0;
  graph->nHops = nGpus - 1;
  graph->typeIntra = PATH_NVL;
  graph->typeInter = PATH_SYS;
  graph->latencyInter = 0;

  // Bandwidth: minimum bottleneck across trees with non-zero weight
  float minBw = FLT_MAX;
  int nonZeroTrees = 0;
  for (int t = 0; t < packing.nTrees; t++) {
    if (packing.trees[t].weight > 0) {
      float bw = blinkTreeBottleneck(&bg, &packing.trees[t]);
      if (bw < minBw) minBw = bw;
      nonZeroTrees++;
    }
  }
  if (nonZeroTrees == 0) minBw = 0;
  graph->bwIntra = minBw;
  graph->bwInter = 0;

  // Fill intra[] with DFS chain orderings
  for (int c = 0; c < packing.nTrees; c++) {
    int chain[BLINK_MAX_GPUS];
    ret = blinkTreeToChain(&packing.trees[c], bg.nVertices, bg.gpuRanks, chain);
    if (ret != ncclSuccess) {
      INFO(NCCL_GRAPH, "Blink: chain conversion failed for tree %d, falling back", c);
      graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
      graph->minChannels = nChannelsNeeded;
      graph->maxChannels = nChannelsNeeded;
      return ncclTopoCompute(system, graph);
    }
    for (int i = 0; i < bg.nVertices; i++) {
      graph->intra[c * nGpus + i] = chain[i];
    }
    graph->inter[c * 2 + 0] = -1;
    graph->inter[c * 2 + 1] = -1;
  }

  clock_gettime(CLOCK_MONOTONIC, &t4);

  INFO(NCCL_GRAPH, "Blink: computed %d tree channels, bwIntra=%.1f GB/s, totalRate=%.1f GB/s",
       graph->nChannels, graph->bwIntra, packing.totalRate);
  INFO(NCCL_GRAPH, "Blink timing: extract=%.1fus MWU=%.1fus refine=%.1fus chain=%.1fus total=%.1fus",
       blinkElapsedUs(&t0, &t1), blinkElapsedUs(&t1, &t2),
       blinkElapsedUs(&t2, &t3), blinkElapsedUs(&t3, &t4),
       blinkElapsedUs(&t0, &t4));

  for (int c = 0; c < graph->nChannels; c++) {
    char buf[256];
    int pos = 0;
    pos += snprintf(buf + pos, sizeof(buf) - pos, "Blink tree %d (root=%d, w=%.1f): ",
                    c, packing.trees[c].root, packing.trees[c].weight);
    for (int i = 0; i < nGpus && pos < (int)sizeof(buf) - 8; i++) {
      pos += snprintf(buf + pos, sizeof(buf) - pos, "%d ", graph->intra[c * nGpus + i]);
    }
    INFO(NCCL_GRAPH, "%s", buf);
  }

  return ncclSuccess;
}
