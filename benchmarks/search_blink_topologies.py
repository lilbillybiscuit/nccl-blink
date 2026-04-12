#!/usr/bin/env python3
"""Search for topology shapes where Blink outperforms default NCCL trees.

This harness takes a base NCCL topology XML, mutates its GPU NVLink structure
into simple candidate families, benchmarks each candidate with `NCCL_BLINK=0`
and `NCCL_BLINK=1`, and ranks the topologies where Blink wins on throughput or
latency.

The search is intentionally simple and stdlib-only:
  - it mutates either direct GPU-to-GPU NVLink counts or switch-backed GPU
    uplink counts in the XML
  - it generates structured graph families plus random connected samples
  - it writes every generated topology XML to disk
  - it benchmarks candidates with nccl-tests `all_reduce_perf`
  - it exports raw, per-size, and per-topology CSV reports

Typical usage:
  python3 benchmarks/search_blink_topologies.py

Dry-run to inspect generated candidate topologies without touching GPUs:
  python3 benchmarks/search_blink_topologies.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import re
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from run_intranode_allreduce import (
    DEFAULT_BINARY,
    DEFAULT_LIB_DIR,
    REPO_ROOT,
    build_env,
    detect_gpu_count,
    ensure_binary_exists,
    parse_extra_env,
    parse_perf_output,
    parse_size_token,
)


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "benchmarks" / "results" / "topology_search"
DEFAULT_RANDOM_SAMPLES = 16
DEFAULT_FAMILIES = "base,uniform,two-islands,hub,ring,random"
BLINK_TIMING_RE = re.compile(
    r"Blink timing: extract=([\d.]+)us MWU=([\d.]+)us refine=([\d.]+)us "
    r"chain=([\d.]+)us total=([\d.]+)us"
)


def default_base_topology() -> Path:
    candidates = (
        REPO_ROOT / "benchmarks" / "topologies" / "h200_real.xml",
        REPO_ROOT / "benchmarks" / "topologies" / "dgx1_8gpu.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "candidate"


def zero_matrix(n: int) -> list[list[int]]:
    return [[0 for _ in range(n)] for _ in range(n)]


def freeze_matrix(matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in matrix)


def symmetrize(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    n = len(matrix)
    out = zero_matrix(n)
    for i in range(n):
        for j in range(i + 1, n):
            count = max(int(matrix[i][j]), int(matrix[j][i]))
            out[i][j] = count
            out[j][i] = count
    return out


def matrix_signature(matrix: Sequence[Sequence[int]]) -> tuple[int, ...]:
    n = len(matrix)
    signature = []
    for i in range(n):
        for j in range(i + 1, n):
            signature.append(int(matrix[i][j]))
    return tuple(signature)


def graph_degrees(matrix: Sequence[Sequence[int]]) -> list[int]:
    return [sum(1 for count in row if count > 0) for row in matrix]


def is_connected(matrix: Sequence[Sequence[int]]) -> bool:
    n = len(matrix)
    if n <= 1:
        return True
    seen = {0}
    stack = [0]
    while stack:
        src = stack.pop()
        for dst, count in enumerate(matrix[src]):
            if dst in seen or count <= 0:
                continue
            seen.add(dst)
            stack.append(dst)
    return len(seen) == n


def undirected_edges(matrix: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    edges = []
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] > 0:
                edges.append((i, j, matrix[i][j]))
    return edges


def matrix_stats(matrix: Sequence[Sequence[int]]) -> dict[str, int]:
    degrees = graph_degrees(matrix)
    edges = undirected_edges(matrix)
    return {
        "gpu_count": len(matrix),
        "undirected_edges": len(edges),
        "total_link_count": sum(edge[2] for edge in edges),
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
    }


def switch_totals(counts: Sequence[Sequence[int]]) -> list[int]:
    return [sum(int(value) for value in row) for row in counts]


def switch_stats(counts: Sequence[Sequence[int]]) -> dict[str, int]:
    totals = switch_totals(counts)
    active_gpu_count = sum(1 for total in totals if total > 0)
    degrees = [active_gpu_count - 1 if total > 0 else 0 for total in totals]
    return {
        "gpu_count": len(counts),
        "undirected_edges": active_gpu_count * (active_gpu_count - 1) // 2,
        "total_link_count": sum(totals),
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
    }


def candidate_stats(
    template: TopologyTemplate, counts: Sequence[Sequence[int]]
) -> dict[str, int]:
    if is_direct_template(template):
        return matrix_stats(counts)
    return switch_stats(counts)


def candidate_signature(
    template: TopologyTemplate, counts: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    if is_direct_template(template):
        return matrix_signature(counts)
    return tuple(int(value) for row in counts for value in row)


def candidate_connected(
    template: TopologyTemplate, counts: Sequence[Sequence[int]]
) -> bool:
    if is_direct_template(template):
        return is_connected(counts)
    return all(total > 0 for total in switch_totals(counts))


def gmean(values: Iterable[float]) -> float | None:
    valid = [value for value in values if value and value > 0]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def format_speedup(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(value)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)}{unit}"
    return f"{size:.1f}{unit}"


@dataclass(frozen=True)
class GpuNode:
    rank: int
    dev: int
    busid: str


@dataclass
class TopologyTemplate:
    source_path: Path
    root: ET.Element
    gpus: list[GpuNode]
    kind: str
    base_counts: list[list[int]]
    positive_levels: list[int]
    default_tclass: str
    allow_new_edges: bool
    switch_targets: list[str]

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def busids(self) -> list[str]:
        return [gpu.busid for gpu in self.gpus]


@dataclass(frozen=True)
class TopologyCandidate:
    name: str
    family: str
    description: str
    params: str
    xml_path: Path
    counts: tuple[tuple[int, ...], ...]
    gpu_count: int
    undirected_edges: int
    total_link_count: int
    min_degree: int
    max_degree: int


def is_direct_template(template: TopologyTemplate) -> bool:
    return template.kind == "direct"


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def load_topology_template(path: Path, allow_new_edges: bool) -> TopologyTemplate:
    tree = ET.parse(path)
    root = tree.getroot()

    gpu_entries: list[GpuNode] = []
    for pci in root.iter("pci"):
        gpu = pci.find("gpu")
        if gpu is None:
            continue
        busid = pci.get("busid")
        if busid is None:
            raise ValueError(f"GPU parent <pci> node is missing busid in {path}")
        rank = int(gpu.get("rank", gpu.get("dev", len(gpu_entries))))
        dev = int(gpu.get("dev", rank))
        gpu_entries.append(GpuNode(rank=rank, dev=dev, busid=busid))

    if not gpu_entries:
        raise ValueError(f"no <gpu> nodes found in topology XML: {path}")

    gpu_entries.sort(key=lambda gpu: (gpu.rank, gpu.dev, gpu.busid))
    bus_to_idx = {gpu.busid: index for index, gpu in enumerate(gpu_entries)}
    counts = zero_matrix(len(gpu_entries))
    switch_target_order: list[str] = []
    switch_rows: list[dict[str, int]] = [{} for _ in gpu_entries]
    default_tclass = None
    saw_gpu_nvlinks = False

    for pci in root.iter("pci"):
        gpu = pci.find("gpu")
        if gpu is None:
            continue
        src_busid = pci.get("busid")
        if src_busid not in bus_to_idx:
            continue
        src = bus_to_idx[src_busid]
        for nvlink in gpu.findall("nvlink"):
            saw_gpu_nvlinks = True
            target = nvlink.get("target")
            count = int(nvlink.get("count", "0"))
            if default_tclass is None:
                default_tclass = nvlink.get("tclass")
            if target in bus_to_idx:
                dst = bus_to_idx[target]
                counts[src][dst] = max(counts[src][dst], count)
            elif target:
                if target not in switch_target_order:
                    switch_target_order.append(target)
                switch_rows[src][target] = max(switch_rows[src].get(target, 0), count)

    if default_tclass is None:
        default_tclass = "0x030200"

    counts = symmetrize(counts)
    positive_levels = sorted({count for row in counts for count in row if count > 0})
    if positive_levels:
        return TopologyTemplate(
            source_path=path,
            root=root,
            gpus=gpu_entries,
            kind="direct",
            base_counts=counts,
            positive_levels=positive_levels,
            default_tclass=default_tclass,
            allow_new_edges=allow_new_edges,
            switch_targets=[],
        )

    switch_counts = [
        [row.get(target, 0) for target in switch_target_order]
        for row in switch_rows
    ]
    switch_levels = sorted(
        {
            count
            for row in switch_counts
            for count in row
            if count > 0
        }
    )
    if switch_levels:
        return TopologyTemplate(
            source_path=path,
            root=root,
            gpus=gpu_entries,
            kind="switch",
            base_counts=switch_counts,
            positive_levels=switch_levels,
            default_tclass=default_tclass,
            allow_new_edges=allow_new_edges,
            switch_targets=switch_target_order,
        )

    if saw_gpu_nvlinks:
        raise ValueError(
            f"topology {path} contains nvlink entries but none carried a positive count"
        )
    raise ValueError(f"topology {path} has no GPU nvlink counts")


def edge_allowed(template: TopologyTemplate, src: int, dst: int) -> bool:
    if not is_direct_template(template):
        return False
    if src == dst:
        return False
    if template.allow_new_edges:
        return True
    return template.base_counts[src][dst] > 0


def switch_link_allowed(template: TopologyTemplate, gpu: int, target_idx: int) -> bool:
    if is_direct_template(template):
        return False
    if template.allow_new_edges:
        return True
    return template.base_counts[gpu][target_idx] > 0


def write_topology_candidate(
    template: TopologyTemplate,
    counts: Sequence[Sequence[int]],
    output_path: Path,
) -> None:
    root = copy.deepcopy(template.root)
    gpu_nodes: dict[str, ET.Element] = {}
    known_busids = set(template.busids)

    for pci in root.iter("pci"):
        gpu = pci.find("gpu")
        if gpu is None:
            continue
        busid = pci.get("busid")
        if busid in known_busids:
            gpu_nodes[busid] = gpu

    for src, gpu in enumerate(template.gpus):
        gpu_node = gpu_nodes[gpu.busid]
        for child in list(gpu_node):
            if child.tag != "nvlink":
                continue
            if not is_direct_template(template) or child.get("target") in known_busids:
                gpu_node.remove(child)

        if is_direct_template(template):
            for dst, target_gpu in enumerate(template.gpus):
                count = int(counts[src][dst])
                if src == dst or count <= 0:
                    continue
                nvlink = ET.Element("nvlink")
                nvlink.set("target", target_gpu.busid)
                nvlink.set("count", str(count))
                nvlink.set("tclass", template.default_tclass)
                gpu_node.append(nvlink)
        else:
            for target_idx, target in enumerate(template.switch_targets):
                count = int(counts[src][target_idx])
                if count <= 0:
                    continue
                nvlink = ET.Element("nvlink")
                nvlink.set("target", target)
                nvlink.set("count", str(count))
                nvlink.set("tclass", template.default_tclass)
                gpu_node.append(nvlink)

    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8")


class CandidateRegistry:
    def __init__(
        self,
        template: TopologyTemplate,
        output_dir: Path,
        max_candidates: int = 0,
    ):
        self.template = template
        self.output_dir = output_dir
        self.max_candidates = max_candidates
        self.seen: set[tuple[int, ...]] = set()
        self.candidates: list[TopologyCandidate] = []

    def add(
        self,
        family: str,
        name: str,
        description: str,
        params: str,
        counts: Sequence[Sequence[int]],
    ) -> bool:
        if self.max_candidates > 0 and len(self.candidates) >= self.max_candidates:
            return False
        if is_direct_template(self.template):
            counts = symmetrize(counts)
            n = len(counts)
            for src in range(n):
                for dst in range(n):
                    if counts[src][dst] > 0 and not edge_allowed(self.template, src, dst):
                        return False
        else:
            counts = [[int(value) for value in row] for row in counts]
            if len(counts) != self.template.gpu_count:
                return False
            if any(len(row) != len(self.template.switch_targets) for row in counts):
                return False
            for gpu, row in enumerate(counts):
                for target_idx, value in enumerate(row):
                    if value < 0:
                        return False
                    if value > 0 and not switch_link_allowed(self.template, gpu, target_idx):
                        return False

        if not candidate_connected(self.template, counts):
            return False

        signature = candidate_signature(self.template, counts)
        if signature in self.seen:
            return False
        self.seen.add(signature)

        safe_name = slugify(name)
        xml_path = self.output_dir / f"{safe_name}.xml"
        write_topology_candidate(self.template, counts, xml_path)
        stats = candidate_stats(self.template, counts)
        self.candidates.append(
            TopologyCandidate(
                name=safe_name,
                family=family,
                description=description,
                params=params,
                xml_path=xml_path,
                counts=freeze_matrix(counts),
                gpu_count=stats["gpu_count"],
                undirected_edges=stats["undirected_edges"],
                total_link_count=stats["total_link_count"],
                min_degree=stats["min_degree"],
                max_degree=stats["max_degree"],
            )
        )
        return True


def pick_levels(levels: Sequence[int]) -> tuple[int, int, int]:
    low = levels[0]
    high = levels[-1]
    mid = levels[len(levels) // 2]
    return low, mid, high


def strong_levels(levels: Sequence[int]) -> list[int]:
    low, mid, high = pick_levels(levels)
    out = [high]
    if mid != high:
        out.append(mid)
    return out


def weak_levels(levels: Sequence[int]) -> list[int]:
    low = levels[0]
    return [0, low]


def search_levels(template: TopologyTemplate) -> list[int]:
    levels = set(int(level) for level in template.positive_levels)
    if levels and min(levels) > 1:
        levels.add(1)
    return sorted(levels)


def build_uniform(template: TopologyTemplate, count: int) -> list[list[int]]:
    n = template.gpu_count
    counts = zero_matrix(n)
    for i in range(n):
        for j in range(i + 1, n):
            if edge_allowed(template, i, j):
                counts[i][j] = count
                counts[j][i] = count
    return counts


def build_two_islands(
    template: TopologyTemplate,
    split: int,
    intra_count: int,
    inter_count: int,
) -> list[list[int]]:
    n = template.gpu_count
    counts = zero_matrix(n)
    for i in range(n):
        for j in range(i + 1, n):
            if not edge_allowed(template, i, j):
                continue
            same_side = (i < split) == (j < split)
            count = intra_count if same_side else inter_count
            counts[i][j] = count
            counts[j][i] = count
    return counts


def build_hub(
    template: TopologyTemplate,
    hub: int,
    strong_count: int,
    weak_count: int,
) -> list[list[int]]:
    n = template.gpu_count
    counts = zero_matrix(n)
    for i in range(n):
        for j in range(i + 1, n):
            if not edge_allowed(template, i, j):
                continue
            count = strong_count if hub in (i, j) else weak_count
            counts[i][j] = count
            counts[j][i] = count
    return counts


def build_ring(
    template: TopologyTemplate,
    strong_count: int,
    weak_count: int,
) -> list[list[int]]:
    n = template.gpu_count
    ring_edges = {
        tuple(sorted((index, (index + 1) % n)))
        for index in range(n)
    }
    counts = zero_matrix(n)
    for i in range(n):
        for j in range(i + 1, n):
            if not edge_allowed(template, i, j):
                continue
            count = strong_count if (i, j) in ring_edges else weak_count
            counts[i][j] = count
            counts[j][i] = count
    return counts


def build_switch_uniform(template: TopologyTemplate, count: int) -> list[list[int]]:
    counts = [[0 for _ in template.switch_targets] for _ in range(template.gpu_count)]
    for gpu in range(template.gpu_count):
        for target_idx in range(len(template.switch_targets)):
            if switch_link_allowed(template, gpu, target_idx):
                counts[gpu][target_idx] = count
    return counts


def build_switch_group_strengths(
    template: TopologyTemplate,
    selector: Sequence[int],
    strong_count: int,
    weak_count: int,
) -> list[list[int]]:
    selected = set(selector)
    counts = [[0 for _ in template.switch_targets] for _ in range(template.gpu_count)]
    for gpu in range(template.gpu_count):
        per_link_count = strong_count if gpu in selected else weak_count
        for target_idx in range(len(template.switch_targets)):
            if switch_link_allowed(template, gpu, target_idx):
                counts[gpu][target_idx] = per_link_count
    return counts


def build_switch_alternating(
    template: TopologyTemplate,
    strong_count: int,
    weak_count: int,
) -> list[list[int]]:
    strong_gpus = [gpu for gpu in range(template.gpu_count) if gpu % 2 == 0]
    return build_switch_group_strengths(template, strong_gpus, strong_count, weak_count)


def build_switch_random_candidate(
    template: TopologyTemplate,
    rng: random.Random,
    count_palette: Sequence[int],
) -> list[list[int]]:
    counts = [[0 for _ in template.switch_targets] for _ in range(template.gpu_count)]
    choices = [0, *count_palette]
    non_zero_choices = [count for count in count_palette if count > 0]
    for gpu in range(template.gpu_count):
        active_targets = [
            target_idx
            for target_idx in range(len(template.switch_targets))
            if switch_link_allowed(template, gpu, target_idx)
        ]
        if not active_targets:
            return None
        row_has_positive = False
        for target_idx in active_targets:
            value = rng.choice(choices)
            counts[gpu][target_idx] = value
            row_has_positive = row_has_positive or value > 0
        if not row_has_positive:
            force_target = rng.choice(active_targets)
            counts[gpu][force_target] = rng.choice(non_zero_choices)
    return counts


def random_spanning_tree(
    n: int,
    allowed_edges: Sequence[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    shuffled = list(allowed_edges)
    rng.shuffle(shuffled)
    dsu = DisjointSet(n)
    tree: list[tuple[int, int]] = []
    for src, dst in shuffled:
        if dsu.union(src, dst):
            tree.append((src, dst))
            if len(tree) == n - 1:
                return tree
    return None


def build_random_candidate(
    template: TopologyTemplate,
    rng: random.Random,
    count_palette: Sequence[int],
) -> list[list[int]] | None:
    n = template.gpu_count
    counts = zero_matrix(n)
    allowed_edges = [
        (src, dst)
        for src in range(n)
        for dst in range(src + 1, n)
        if edge_allowed(template, src, dst)
    ]
    tree_edges = random_spanning_tree(n, allowed_edges, rng)
    if tree_edges is None:
        return None

    high_counts = strong_levels(count_palette)
    for src, dst in tree_edges:
        count = rng.choice(high_counts)
        counts[src][dst] = count
        counts[dst][src] = count

    for src, dst in allowed_edges:
        if counts[src][dst] > 0:
            continue
        if rng.random() < 0.45:
            count = rng.choice([0, count_palette[0], count_palette[-1]])
            counts[src][dst] = count
            counts[dst][src] = count
    return counts


def generate_candidates(
    template: TopologyTemplate,
    families: Sequence[str],
    random_samples: int,
    random_seed: int,
    output_dir: Path,
    max_candidates: int = 0,
) -> list[TopologyCandidate]:
    registry = CandidateRegistry(template, output_dir, max_candidates=max_candidates)
    levels = search_levels(template)
    n = template.gpu_count

    if "base" in families:
        registry.add(
            family="base",
            name="base_topology",
            description="Unmodified base topology",
            params=f"source={template.source_path.name}",
            counts=template.base_counts,
        )

    if is_direct_template(template):
        if "uniform" in families:
            for count in levels:
                registry.add(
                    family="uniform",
                    name=f"uniform_count_{count}",
                    description="All allowed NVLink edges share the same count",
                    params=f"count={count}",
                    counts=build_uniform(template, count),
                )

        if "two-islands" in families and n >= 4:
            split = n // 2
            for intra_count in strong_levels(levels):
                for inter_count in weak_levels(levels):
                    registry.add(
                        family="two-islands",
                        name=f"two_islands_intra_{intra_count}_inter_{inter_count}",
                        description="Dense intra-island links with weaker cross-island bridges",
                        params=f"split={split},intra={intra_count},inter={inter_count}",
                        counts=build_two_islands(template, split, intra_count, inter_count),
                    )

        if "hub" in families and n >= 4:
            hub_indices = [0, n // 2]
            if (n - 1) not in hub_indices:
                hub_indices.append(n - 1)
            seen_hubs = []
            for hub in hub_indices:
                if hub not in seen_hubs:
                    seen_hubs.append(hub)
            for hub in seen_hubs:
                for strong_count in strong_levels(levels):
                    for weak_count in weak_levels(levels):
                        registry.add(
                            family="hub",
                            name=f"hub_{hub}_strong_{strong_count}_weak_{weak_count}",
                            description=f"Hub-and-spoke layout centered on GPU rank index {hub}",
                            params=f"hub={hub},strong={strong_count},weak={weak_count}",
                            counts=build_hub(template, hub, strong_count, weak_count),
                        )

        if "ring" in families and n >= 4:
            for strong_count in strong_levels(levels):
                for weak_count in weak_levels(levels):
                    registry.add(
                        family="ring",
                        name=f"ring_strong_{strong_count}_weak_{weak_count}",
                        description="Rank-ordered ring gets the strong links, all others are weak",
                        params=f"strong={strong_count},weak={weak_count}",
                        counts=build_ring(template, strong_count, weak_count),
                    )

        if "random" in families and random_samples > 0:
            rng = random.Random(random_seed)
            for sample_idx in range(random_samples):
                counts = build_random_candidate(template, rng, levels)
                if counts is None:
                    continue
                registry.add(
                    family="random",
                    name=f"random_sample_{sample_idx:03d}",
                    description="Random connected thinning seeded by a random spanning tree",
                    params=f"seed={random_seed},sample={sample_idx}",
                    counts=counts,
                )
    else:
        if "uniform" in families:
            for count in levels:
                registry.add(
                    family="uniform",
                    name=f"uniform_count_{count}",
                    description="Every GPU gets the same NVSwitch uplink count on each visible target",
                    params=f"count={count}",
                    counts=build_switch_uniform(template, count),
                )

        if "two-islands" in families and n >= 4:
            split = n // 2
            island_a = list(range(split))
            for strong_count in strong_levels(levels):
                for weak_count in weak_levels(levels):
                    registry.add(
                        family="two-islands",
                        name=f"two_islands_strong_{strong_count}_weak_{weak_count}",
                        description="First half of GPUs get stronger NVSwitch uplinks than the second half",
                        params=f"split={split},strong={strong_count},weak={weak_count}",
                        counts=build_switch_group_strengths(
                            template, island_a, strong_count, weak_count
                        ),
                    )

        if "hub" in families and n >= 4:
            hub_indices = [0, n // 2]
            if (n - 1) not in hub_indices:
                hub_indices.append(n - 1)
            seen_hubs = []
            for hub in hub_indices:
                if hub not in seen_hubs:
                    seen_hubs.append(hub)
            for hub in seen_hubs:
                for strong_count in strong_levels(levels):
                    for weak_count in weak_levels(levels):
                        registry.add(
                            family="hub",
                            name=f"hub_{hub}_strong_{strong_count}_weak_{weak_count}",
                            description=f"GPU rank index {hub} gets stronger NVSwitch uplinks than its peers",
                            params=f"hub={hub},strong={strong_count},weak={weak_count}",
                            counts=build_switch_group_strengths(
                                template, [hub], strong_count, weak_count
                            ),
                        )

        if "ring" in families and n >= 4:
            for strong_count in strong_levels(levels):
                for weak_count in weak_levels(levels):
                    registry.add(
                        family="ring",
                        name=f"ring_alternating_strong_{strong_count}_weak_{weak_count}",
                        description="Alternating ranks get stronger versus weaker NVSwitch uplinks",
                        params=f"strong={strong_count},weak={weak_count}",
                        counts=build_switch_alternating(template, strong_count, weak_count),
                    )

        if "random" in families and random_samples > 0:
            rng = random.Random(random_seed)
            for sample_idx in range(random_samples):
                counts = build_switch_random_candidate(template, rng, levels)
                if counts is None:
                    continue
                registry.add(
                    family="random",
                    name=f"random_sample_{sample_idx:03d}",
                    description="Randomized NVSwitch uplink counts per GPU and target",
                    params=f"seed={random_seed},sample={sample_idx}",
                    counts=counts,
                )

    return registry.candidates


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def candidate_manifest_rows(candidates: Sequence[TopologyCandidate]) -> list[dict[str, object]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "candidate": candidate.name,
                "family": candidate.family,
                "description": candidate.description,
                "params": candidate.params,
                "gpu_count": candidate.gpu_count,
                "undirected_edges": candidate.undirected_edges,
                "total_link_count": candidate.total_link_count,
                "min_degree": candidate.min_degree,
                "max_degree": candidate.max_degree,
                "topology_file": str(candidate.xml_path),
            }
        )
    return rows


def parse_blink_timing(output: str) -> dict[str, float] | None:
    match = BLINK_TIMING_RE.search(output)
    if not match:
        return None
    return {
        "extract_us": float(match.group(1)),
        "mwu_us": float(match.group(2)),
        "refine_us": float(match.group(3)),
        "chain_us": float(match.group(4)),
        "total_us": float(match.group(5)),
    }


def run_candidate(
    *,
    binary: Path,
    lib_dir: Path,
    results_dir: Path,
    candidate: TopologyCandidate,
    gpu_count: int,
    visible_devices: str,
    begin_size: str,
    end_size: str,
    factor: str,
    validation: int,
    warmup_iters: int,
    iters: int,
    blink: int,
    repeat: int,
    extra_env: dict[str, str],
    capture_debug: bool,
) -> tuple[list[dict[str, object]], dict[str, object] | None, Path]:
    env_overrides = {
        "NCCL_BLINK": str(blink),
        "NCCL_TOPO_FILE": str(candidate.xml_path),
        "NCCL_IGNORE_DISABLED_P2P": "2",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
    }
    if capture_debug:
        env_overrides["NCCL_DEBUG"] = "GRAPH"
        env_overrides["NCCL_DEBUG_SUBSYS"] = "GRAPH"
    env_overrides.update(extra_env)

    env = build_env(visible_devices, lib_dir, env_overrides)
    command = [
        str(binary),
        "-b",
        begin_size,
        "-e",
        end_size,
        "-f",
        factor,
        "-g",
        str(gpu_count),
        "-c",
        str(validation),
        "-w",
        str(warmup_iters),
        "-n",
        str(iters),
    ]

    proc = subprocess.run(
        command,
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    combined_output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    log_file = (
        results_dir
        / "logs"
        / f"{candidate.name}_blink{blink}_repeat{repeat:02d}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(combined_output)

    if proc.returncode != 0:
        raise RuntimeError(
            f"all_reduce_perf failed for candidate={candidate.name} blink={blink} "
            f"repeat={repeat}. See {log_file}"
        )

    rows = parse_perf_output(combined_output)
    if not rows:
        raise RuntimeError(
            f"no performance rows parsed for candidate={candidate.name} "
            f"blink={blink} repeat={repeat}. See {log_file}"
        )

    timestamp_utc = datetime.now(timezone.utc).isoformat()
    annotated_rows = []
    for row in rows:
        annotated_rows.append(
            {
                "timestamp_utc": timestamp_utc,
                "candidate": candidate.name,
                "family": candidate.family,
                "description": candidate.description,
                "params": candidate.params,
                "topology_file": str(candidate.xml_path),
                "blink": blink,
                "repeat": repeat,
                "gpu_count": gpu_count,
                "visible_devices": visible_devices,
                "binary": str(binary),
                "begin_size": begin_size,
                "end_size": end_size,
                "factor": factor,
                "validation": validation,
                "warmup_iters": warmup_iters,
                "iters": iters,
                "log_file": str(log_file),
                "undirected_edges": candidate.undirected_edges,
                "total_link_count": candidate.total_link_count,
                "min_degree": candidate.min_degree,
                "max_degree": candidate.max_degree,
                **row,
            }
        )

    blink_timing = parse_blink_timing(combined_output)
    timing_row = None
    if blink_timing is not None:
        timing_row = {
            "candidate": candidate.name,
            "family": candidate.family,
            "blink": blink,
            "repeat": repeat,
            "log_file": str(log_file),
            **blink_timing,
        }
    return annotated_rows, timing_row, log_file


def aggregate_rows(
    raw_rows: Sequence[dict[str, object]],
    large_message_min_bytes: int,
    small_message_max_bytes: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["candidate"]), int(row["blink"]), int(row["size_bytes"]))].append(row)

    per_candidate_size: dict[str, dict[int, dict[int, dict[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (candidate, blink, size_bytes), rows in sorted(grouped.items()):
        busbw = median([float(row["busbw_gbps"]) for row in rows])
        algbw = median([float(row["algbw_gbps"]) for row in rows])
        time_us = median([float(row["time_us"]) for row in rows])
        record = {
            "candidate": candidate,
            "family": rows[0]["family"],
            "description": rows[0]["description"],
            "params": rows[0]["params"],
            "topology_file": rows[0]["topology_file"],
            "blink": blink,
            "size_bytes": size_bytes,
            "size_mb": size_bytes / (1024 * 1024),
            "median_busbw_gbps": busbw,
            "median_algbw_gbps": algbw,
            "median_time_us": time_us,
            "runs": len(rows),
            "undirected_edges": rows[0]["undirected_edges"],
            "total_link_count": rows[0]["total_link_count"],
            "min_degree": rows[0]["min_degree"],
            "max_degree": rows[0]["max_degree"],
        }
        per_candidate_size[candidate][size_bytes][blink] = record

    comparison_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for candidate, size_map in sorted(per_candidate_size.items()):
        base_rows = [entry[0] for entry in size_map.values() if 0 in entry]
        blink_rows = [entry[1] for entry in size_map.values() if 1 in entry]
        template_row = base_rows[0] if base_rows else blink_rows[0]
        busbw_speedups_large = []
        latency_speedups_small = []
        busbw_wins = 0
        latency_wins = 0
        best_busbw_speedup = None
        best_busbw_size = None
        best_latency_speedup = None
        best_latency_size = None

        for size_bytes in sorted(size_map):
            paired = size_map[size_bytes]
            if 0 not in paired or 1 not in paired:
                continue
            default_row = paired[0]
            blink_row = paired[1]
            busbw_speedup = (
                float(blink_row["median_busbw_gbps"]) / float(default_row["median_busbw_gbps"])
                if float(default_row["median_busbw_gbps"]) > 0
                else None
            )
            latency_speedup = (
                float(default_row["median_time_us"]) / float(blink_row["median_time_us"])
                if float(blink_row["median_time_us"]) > 0
                else None
            )
            if busbw_speedup and busbw_speedup > 1.01:
                busbw_wins += 1
            if latency_speedup and latency_speedup > 1.01:
                latency_wins += 1
            if size_bytes >= large_message_min_bytes and busbw_speedup:
                busbw_speedups_large.append(busbw_speedup)
            if size_bytes <= small_message_max_bytes and latency_speedup:
                latency_speedups_small.append(latency_speedup)
            if busbw_speedup and (best_busbw_speedup is None or busbw_speedup > best_busbw_speedup):
                best_busbw_speedup = busbw_speedup
                best_busbw_size = size_bytes
            if latency_speedup and (
                best_latency_speedup is None or latency_speedup > best_latency_speedup
            ):
                best_latency_speedup = latency_speedup
                best_latency_size = size_bytes

            comparison_rows.append(
                {
                    "candidate": candidate,
                    "family": template_row["family"],
                    "description": template_row["description"],
                    "params": template_row["params"],
                    "topology_file": template_row["topology_file"],
                    "size_bytes": size_bytes,
                    "size_mb": size_bytes / (1024 * 1024),
                    "default_busbw_gbps": default_row["median_busbw_gbps"],
                    "blink_busbw_gbps": blink_row["median_busbw_gbps"],
                    "busbw_speedup": busbw_speedup,
                    "default_time_us": default_row["median_time_us"],
                    "blink_time_us": blink_row["median_time_us"],
                    "latency_speedup": latency_speedup,
                    "undirected_edges": template_row["undirected_edges"],
                    "total_link_count": template_row["total_link_count"],
                    "min_degree": template_row["min_degree"],
                    "max_degree": template_row["max_degree"],
                }
            )

        summary_rows.append(
            {
                "candidate": candidate,
                "family": template_row["family"],
                "description": template_row["description"],
                "params": template_row["params"],
                "topology_file": template_row["topology_file"],
                "undirected_edges": template_row["undirected_edges"],
                "total_link_count": template_row["total_link_count"],
                "min_degree": template_row["min_degree"],
                "max_degree": template_row["max_degree"],
                "n_sizes_compared": sum(1 for paired in size_map.values() if 0 in paired and 1 in paired),
                "large_message_min_bytes": large_message_min_bytes,
                "small_message_max_bytes": small_message_max_bytes,
                "throughput_large_gmean_speedup": gmean(busbw_speedups_large),
                "latency_small_gmean_speedup": gmean(latency_speedups_small),
                "throughput_win_sizes": busbw_wins,
                "latency_win_sizes": latency_wins,
                "best_throughput_speedup": best_busbw_speedup,
                "best_throughput_size_bytes": best_busbw_size,
                "best_latency_speedup": best_latency_speedup,
                "best_latency_size_bytes": best_latency_size,
            }
        )

    return comparison_rows, summary_rows


def build_report(summary_rows: Sequence[dict[str, object]], top_k: int) -> str:
    throughput_rows = sorted(
        summary_rows,
        key=lambda row: row["throughput_large_gmean_speedup"] or 0.0,
        reverse=True,
    )
    latency_rows = sorted(
        summary_rows,
        key=lambda row: row["latency_small_gmean_speedup"] or 0.0,
        reverse=True,
    )

    lines = []
    lines.append("Top throughput wins:")
    for row in throughput_rows[:top_k]:
        lines.append(
            "  "
            f"{row['candidate']}: large-msg={format_speedup(row['throughput_large_gmean_speedup'])}, "
            f"best={format_speedup(row['best_throughput_speedup'])} @ "
            f"{format_bytes(row['best_throughput_size_bytes'])}, "
            f"edges={row['undirected_edges']}, family={row['family']}"
        )

    lines.append("")
    lines.append("Top latency wins:")
    for row in latency_rows[:top_k]:
        lines.append(
            "  "
            f"{row['candidate']}: small-msg={format_speedup(row['latency_small_gmean_speedup'])}, "
            f"best={format_speedup(row['best_latency_speedup'])} @ "
            f"{format_bytes(row['best_latency_size_bytes'])}, "
            f"edges={row['undirected_edges']}, family={row['family']}"
        )
    return "\n".join(lines)


def parse_families(spec: str) -> list[str]:
    families = []
    for family in spec.split(","):
        family = family.strip()
        if family:
            families.append(family)
    return families


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search simple topology variants where Blink beats default NCCL trees."
    )
    parser.add_argument("--base-topology", type=Path, default=default_base_topology())
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--lib-dir", type=Path, default=DEFAULT_LIB_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--random-samples", type=int, default=DEFAULT_RANDOM_SAMPLES)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--allow-new-edges", action="store_true")
    parser.add_argument("--limit-topologies", type=int, default=0)
    parser.add_argument("--begin-size", default="8")
    parser.add_argument("--end-size", default="1G")
    parser.add_argument("--factor", default="2")
    parser.add_argument("--validation", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--visible-devices", default="")
    parser.add_argument("--small-message-max", default="1M")
    parser.add_argument("--large-message-min", default="64M")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--capture-debug", action="store_true")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variables passed through to all_reduce_perf",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()

    base_topology = args.base_topology.resolve()
    if not base_topology.is_file():
        parser.error(f"base topology XML not found: {base_topology}")

    template = load_topology_template(base_topology, allow_new_edges=args.allow_new_edges)
    families = parse_families(args.families)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = args.output_root.resolve() / timestamp
    generated_topology_dir = run_dir / "topologies"

    candidates = generate_candidates(
        template=template,
        families=families,
        random_samples=args.random_samples,
        random_seed=args.random_seed,
        output_dir=generated_topology_dir,
        max_candidates=args.limit_topologies,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = candidate_manifest_rows(candidates)
    manifest_csv = run_dir / "topology_manifest.csv"
    write_csv(manifest_csv, manifest_rows)

    print(f"Base topology: {base_topology}")
    print(f"Generated candidates: {len(candidates)}")
    print(f"Manifest: {manifest_csv}")

    if args.dry_run:
        print("Dry-run enabled; skipping benchmarks.")
        return 0

    ensure_binary_exists(args.binary)
    extra_env = parse_extra_env(args.env)
    detected_gpu_count = detect_gpu_count()
    gpu_count = template.gpu_count
    if args.visible_devices:
        visible_devices = args.visible_devices
        visible_count = len([token for token in args.visible_devices.split(",") if token.strip()])
        if visible_count != gpu_count:
            parser.error(
                f"--visible-devices specifies {visible_count} GPUs but topology expects {gpu_count}"
            )
    else:
        if detected_gpu_count < gpu_count:
            parser.error(
                f"detected {detected_gpu_count} GPUs but topology expects {gpu_count}; "
                "use a smaller topology or pass --visible-devices"
            )
        visible_devices = ",".join(str(index) for index in range(gpu_count))

    print(f"Detected GPUs: {detected_gpu_count}")
    print(f"Using GPU count: {gpu_count}")
    print(f"CUDA_VISIBLE_DEVICES={visible_devices}")

    raw_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for candidate_index, candidate in enumerate(candidates, start=1):
        print(
            f"[{candidate_index}/{len(candidates)}] {candidate.name} "
            f"({candidate.family}, edges={candidate.undirected_edges})"
        )
        for blink in (0, 1):
            for repeat in range(args.repeats):
                print(f"  blink={blink} repeat={repeat + 1}/{args.repeats}")
                try:
                    rows, timing_row, _ = run_candidate(
                        binary=args.binary.resolve(),
                        lib_dir=args.lib_dir.resolve(),
                        results_dir=run_dir,
                        candidate=candidate,
                        gpu_count=gpu_count,
                        visible_devices=visible_devices,
                        begin_size=args.begin_size,
                        end_size=args.end_size,
                        factor=args.factor,
                        validation=args.validation,
                        warmup_iters=args.warmup_iters,
                        iters=args.iters,
                        blink=blink,
                        repeat=repeat,
                        extra_env=extra_env,
                        capture_debug=args.capture_debug,
                    )
                    raw_rows.extend(rows)
                    if timing_row is not None:
                        timing_rows.append(timing_row)
                except Exception as exc:  # noqa: BLE001
                    failure_rows.append(
                        {
                            "candidate": candidate.name,
                            "family": candidate.family,
                            "blink": blink,
                            "repeat": repeat,
                            "error": str(exc),
                        }
                    )
                    print(f"    failed: {exc}", file=sys.stderr)

    raw_csv = run_dir / "search_raw.csv"
    timing_csv = run_dir / "blink_timings.csv"
    failures_csv = run_dir / "failures.csv"
    comparisons_csv = run_dir / "search_per_size.csv"
    summary_csv = run_dir / "search_summary.csv"
    report_txt = run_dir / "report.txt"

    write_csv(raw_csv, raw_rows)
    if timing_rows:
        write_csv(timing_csv, timing_rows)
    if failure_rows:
        write_csv(failures_csv, failure_rows)

    comparison_rows, summary_rows = aggregate_rows(
        raw_rows=raw_rows,
        large_message_min_bytes=parse_size_token(args.large_message_min),
        small_message_max_bytes=parse_size_token(args.small_message_max),
    )
    write_csv(comparisons_csv, comparison_rows)
    write_csv(summary_csv, summary_rows)

    report = build_report(summary_rows, top_k=args.top_k)
    report_txt.write_text(report + "\n")

    print("")
    print(report)
    print("")
    print(f"Raw rows: {raw_csv}")
    print(f"Per-size comparison: {comparisons_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Report: {report_txt}")
    if timing_rows:
        print(f"Blink timings: {timing_csv}")
    if failure_rows:
        print(f"Failures: {failures_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
