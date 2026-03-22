import heapq
import math
import time

import numpy as np


D = math.sqrt(2.0)


def _free(mask, p):
    x, y = round(p[0]), round(p[1])
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x])


def _h(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return D * min(dx, dy) + abs(dx - dy)


def _path(came, cur):
    out = [cur]
    while cur in came:
        cur = came[cur]
        out.append(cur)
    out.reverse()
    return [(float(x), float(y)) for x, y in out]


def _path_len(path):
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def astar_search(mask, start, goal, allow_diagonal=True, allow_corner_cutting=False):
    start = (round(start[0]), round(start[1]))
    goal = (round(goal[0]), round(goal[1]))
    t0 = time.perf_counter()
    dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if allow_diagonal:
        dirs += [(-1, -1, D), (-1, 1, D), (1, -1, D), (1, 1, D)]
    q = [(0.0, 0.0, start)]
    came, g, seen, used = {}, {start: 0.0}, [], set()
    q[0] = (_h(start, goal), 0.0, start)

    while q:
        _, cost, cur = heapq.heappop(q)
        if cur in used:
            continue
        used.add(cur)
        seen.append((float(cur[0]), float(cur[1])))
        if cur == goal:
            path = _path(came, cur)
            return {
                "planner": "A*",
                "success": True,
                "path": path,
                "cost": _path_len(path),
                "elapsed_ms": (time.perf_counter() - t0) * 1000,
                "visited_count": len(used),
                "expanded_points": seen,
                "tree_edges": [],
                "smoothed_path": [],
                "smoothed_cost": None,
            }
        x, y = cur
        for dx, dy, w in dirs:
            nxt = (x + dx, y + dy)
            if not _free(mask, nxt):
                continue
            if dx and dy and not allow_corner_cutting and (not _free(mask, (x + dx, y)) or not _free(mask, (x, y + dy))):
                continue
            nc = cost + w
            if nc >= g.get(nxt, 1e18):
                continue
            g[nxt] = nc
            came[nxt] = cur
            heapq.heappush(q, (nc + _h(nxt, goal), nc, nxt))

    return {
        "planner": "A*",
        "success": False,
        "path": [],
        "cost": None,
        "elapsed_ms": (time.perf_counter() - t0) * 1000,
        "visited_count": len(used),
        "expanded_points": seen,
        "tree_edges": [],
        "smoothed_path": [],
        "smoothed_cost": None,
    }


def downsample_free_mask(mask, factor):
    if factor <= 1:
        return mask.copy()
    h, w = mask.shape
    h -= h % factor
    w -= w % factor
    x = mask[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return x.all(axis=(1, 3))


def scale_point_down(p, factor):
    x, y = round(p[0]), round(p[1])
    return x // factor, y // factor


def astar_resolution_study(mask, start, goal, factors=(1, 2, 4, 8)):
    rows = []
    for factor in factors:
        small = downsample_free_mask(mask, factor)
        res = astar_search(small, scale_point_down(start, factor), scale_point_down(goal, factor))
        rows.append(
            {
                "factor": factor,
                "shape": f"{small.shape[1]}x{small.shape[0]}",
                "success": res["success"],
                "path_length": None if res["cost"] is None else round(res["cost"] * factor, 3),
                "time_ms": round(res["elapsed_ms"], 2),
                "visited_points": res["visited_count"],
            }
        )
    return rows
