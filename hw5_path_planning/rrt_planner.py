import math
import random
import time

import numpy as np


def _dist(a, b):
    return math.dist(a, b)


def _free(mask, p):
    x, y = round(p[0]), round(p[1])
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])


def _cells(a, b):
    x0, y0 = round(a[0]), round(a[1])
    x1, y1 = round(b[0]), round(b[1])
    out = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return out
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _line_free(mask, a, b):
    for x, y in _cells(a, b):
        if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])):
            return False
    return True


def _steer(a, b, step):
    d = _dist(a, b)
    if d <= step:
        return float(b[0]), float(b[1])
    k = step / d
    return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k


def _path_len(path):
    return sum(_dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def _trace(nodes, i):
    out = []
    while i is not None:
        out.append(nodes[i]["p"])
        i = nodes[i]["parent"]
    out.reverse()
    return out


def _sample_free(points, rng):
    y, x = points[rng.randrange(len(points))]
    return float(x), float(y)


def rrt_search(mask, start, goal, max_iter=3000, step_size=10.0, goal_sample_rate=0.1, goal_tolerance=6.0, rng_seed=0):
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    rng = random.Random(rng_seed)
    pts = np.argwhere(mask)
    nodes = [{"p": start, "parent": None, "cost": 0.0, "children": []}]
    edges, samples = [], []
    t0 = time.perf_counter()

    for _ in range(max_iter):
        sample = goal if rng.random() < goal_sample_rate else _sample_free(pts, rng)
        samples.append(sample)
        j = min(range(len(nodes)), key=lambda i: _dist(nodes[i]["p"], sample))
        p = nodes[j]["p"]
        q = _steer(p, sample, step_size)
        if not _free(mask, q) or not _line_free(mask, p, q):
            continue
        nodes.append({"p": q, "parent": j, "cost": nodes[j]["cost"] + _dist(p, q), "children": []})
        k = len(nodes) - 1
        nodes[j]["children"].append(k)
        edges.append((p, q))
        if _dist(q, goal) <= goal_tolerance and _line_free(mask, q, goal):
            nodes.append({"p": goal, "parent": k, "cost": nodes[k]["cost"] + _dist(q, goal), "children": []})
            g = len(nodes) - 1
            nodes[k]["children"].append(g)
            edges.append((q, goal))
            path = _trace(nodes, g)
            return {
                "planner": "RRT",
                "success": True,
                "path": path,
                "cost": _path_len(path),
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
                "visited_count": len(nodes),
                "expanded_points": samples,
                "tree_edges": edges,
                "smoothed_path": [],
                "smoothed_cost": None,
            }

    return {
        "planner": "RRT",
        "success": False,
        "path": [],
        "cost": None,
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
        "visited_count": len(nodes),
        "expanded_points": samples,
        "tree_edges": edges,
        "smoothed_path": [],
        "smoothed_cost": None,
    }
