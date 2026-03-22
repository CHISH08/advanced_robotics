from rrt_planner import _dist, _free, _line_free, _path_len, _sample_free, _steer, _trace

import random
import time

import numpy as np


def _near(nodes, p, r):
    return [i for i in range(len(nodes)) if _dist(nodes[i]["p"], p) <= r]


def _push_delta(nodes, i, delta):
    stack = [i]
    while stack:
        cur = stack.pop()
        for child in nodes[cur]["children"]:
            nodes[child]["cost"] += delta
            stack.append(child)


def rrt_star_search(mask, start, goal, max_iter=3000, step_size=10.0, goal_sample_rate=0.15, goal_tolerance=6.0, neighbor_radius=16.0, rng_seed=0):
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    rng = random.Random(rng_seed)
    pts = np.argwhere(mask)
    nodes = [{"p": start, "parent": None, "cost": 0.0, "children": []}]
    edges, samples = [], []
    best_parent, best_cost = None, 1e18
    t0 = time.perf_counter()

    for _ in range(max_iter):
        sample = goal if rng.random() < goal_sample_rate else _sample_free(pts, rng)
        samples.append(sample)
        j = min(range(len(nodes)), key=lambda i: _dist(nodes[i]["p"], sample))
        q = _steer(nodes[j]["p"], sample, step_size)
        if not _free(mask, q) or not _line_free(mask, nodes[j]["p"], q):
            continue

        near = _near(nodes, q, neighbor_radius)
        parent, cost = j, nodes[j]["cost"] + _dist(nodes[j]["p"], q)
        for i in near:
            c = nodes[i]["cost"] + _dist(nodes[i]["p"], q)
            if c < cost and _line_free(mask, nodes[i]["p"], q):
                parent, cost = i, c

        nodes.append({"p": q, "parent": parent, "cost": cost, "children": []})
        k = len(nodes) - 1
        nodes[parent]["children"].append(k)
        edges.append((nodes[parent]["p"], q))

        for i in near:
            if i == parent:
                continue
            c = cost + _dist(q, nodes[i]["p"])
            if c >= nodes[i]["cost"] or not _line_free(mask, q, nodes[i]["p"]):
                continue
            old = nodes[i]["parent"]
            if old is None:
                continue
            if i in nodes[old]["children"]:
                nodes[old]["children"].remove(i)
            nodes[k]["children"].append(i)
            delta = c - nodes[i]["cost"]
            nodes[i]["parent"] = k
            nodes[i]["cost"] = c
            _push_delta(nodes, i, delta)
            edges.append((q, nodes[i]["p"]))

        if _dist(q, goal) <= goal_tolerance and _line_free(mask, q, goal):
            c = cost + _dist(q, goal)
            if c < best_cost:
                best_parent, best_cost = k, c

    if best_parent is None:
        return {
            "planner": "RRT*",
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

    nodes.append({"p": goal, "parent": best_parent, "cost": best_cost, "children": []})
    g = len(nodes) - 1
    nodes[best_parent]["children"].append(g)
    edges.append((nodes[best_parent]["p"], goal))
    path = _trace(nodes, g)
    return {
        "planner": "RRT*",
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
