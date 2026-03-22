import math
import random
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from astar_planner import astar_resolution_study, astar_search
from rrt_planner import rrt_search
from rrt_star_planner import rrt_star_search

def _dist(a, b):
    return math.dist(a, b)


def _path_len(path):
    return sum(_dist(path[i], path[i + 1]) for i in range(len(path) - 1))


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


def _turn_sum(path):
    if len(path) < 3:
        return 0.0
    s = 0.0
    for i in range(1, len(path) - 1):
        a = path[i - 1]
        b = path[i]
        c = path[i + 1]
        aa = math.atan2(b[1] - a[1], b[0] - a[0])
        bb = math.atan2(c[1] - b[1], c[0] - b[0])
        s += abs(math.degrees((bb - aa + math.pi) % (2 * math.pi) - math.pi))
    return s


def format_benchmark_table(rows, headers=None):
    if headers is None:
        headers = ["planner", "success", "path_length", "smoothed_length", "time_ms", "visited_points"]
    w = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    out = [
        " | ".join(h.ljust(w[h]) for h in headers),
        "-+-".join("-" * w[h] for h in headers),
    ]
    for r in rows:
        out.append(" | ".join(str(r.get(h, "")).ljust(w[h]) for h in headers))
    return "\n".join(out)


def shortcut_path(mask, path, max_trials=120, rng_seed=0):
    if len(path) <= 2:
        return list(path)
    rng = random.Random(rng_seed)
    path = list(path)
    for _ in range(max_trials):
        if len(path) <= 2:
            break
        i, j = sorted(rng.sample(range(len(path)), 2))
        if j <= i + 1:
            continue
        if _line_free(mask, path[i], path[j]):
            path = path[: i + 1] + path[j:]
    return path


def resample_path(path, spacing=1.0):
    if len(path) <= 1:
        return list(path)
    out = [tuple(path[0])]
    for i in range(len(path) - 1):
        a = np.array(path[i], float)
        b = np.array(path[i + 1], float)
        n = max(int(math.ceil(np.linalg.norm(b - a) / spacing)), 1)
        for k in range(1, n + 1):
            p = a + (b - a) * (k / n)
            out.append((float(p[0]), float(p[1])))
    return out


def gradient_smooth_path(path, data_weight=0.15, smooth_weight=0.35, tolerance=1e-3, max_iterations=250):
    if len(path) <= 2:
        return list(path)
    raw = np.array(path, float)
    cur = np.array(path, float)
    for _ in range(max_iterations):
        change = 0.0
        for i in range(1, len(cur) - 1):
            old = cur[i].copy()
            cur[i] += data_weight * (raw[i] - cur[i])
            cur[i] += smooth_weight * (cur[i - 1] + cur[i + 1] - 2 * cur[i])
            change += float(np.linalg.norm(cur[i] - old))
        if change < tolerance:
            break
    return [(float(x), float(y)) for x, y in cur]


def smooth_path(mask, path, rng_seed=0, shortcut_trials=120, spacing=1.0):
    if len(path) <= 2:
        return list(path)
    short = shortcut_path(mask, path, max_trials=shortcut_trials, rng_seed=rng_seed)
    smooth = gradient_smooth_path(resample_path(short, spacing=spacing))
    if all(_line_free(mask, smooth[i], smooth[i + 1]) for i in range(len(smooth) - 1)):
        return smooth
    return short


def attach_smoothed_path(mask, res, rng_seed=0, shortcut_trials=120, spacing=1.0):
    if res["success"] and res["path"]:
        res["smoothed_path"] = smooth_path(mask, res["path"], rng_seed=rng_seed, shortcut_trials=shortcut_trials, spacing=spacing)
        res["smoothed_cost"] = _path_len(res["smoothed_path"])
    return res


def summarize_planner_result(res, start, goal, benchmark_optimal_length=None, query_index=None, seed=None):
    straight = _dist(start, goal)
    row = {
        "planner": res["planner"],
        "success": res["success"],
        "path_length": None if res["cost"] is None else round(res["cost"], 3),
        "smoothed_length": None if res["smoothed_cost"] is None else round(res["smoothed_cost"], 3),
        "time_ms": round(res["elapsed_ms"], 2),
        "visited_points": res["visited_count"],
        "detour_ratio": None if not res["cost"] else round(res["cost"] / straight, 3),
        "smooth_gain_pct": None if not res["cost"] or res["smoothed_cost"] is None else round((res["cost"] - res["smoothed_cost"]) / res["cost"] * 100, 2),
        "turn_sum_deg": None if not res["path"] else round(_turn_sum(res["path"]), 2),
        "smoothed_turn_sum_deg": None if not res["smoothed_path"] else round(_turn_sum(res["smoothed_path"]), 2),
        "optimality_gap_pct": None if benchmark_optimal_length in (None, 0) or res["cost"] is None else round((res["cost"] - benchmark_optimal_length) / benchmark_optimal_length * 100, 2),
    }
    if benchmark_optimal_length is not None:
        row["benchmark_optimal"] = round(float(benchmark_optimal_length), 3)
    if query_index is not None:
        row["query_index"] = query_index
    if seed is not None:
        row["seed"] = seed
    return row


def default_planner_specs(rng_seed=0):
    return [
        ("A*", astar_search, {"allow_diagonal": True, "allow_corner_cutting": False}),
        ("RRT", rrt_search, {"max_iter": 3500, "step_size": 12.0, "goal_sample_rate": 0.12, "goal_tolerance": 6.0, "rng_seed": rng_seed}),
        ("RRT*", rrt_star_search, {"max_iter": 3500, "step_size": 12.0, "goal_sample_rate": 0.16, "goal_tolerance": 6.0, "neighbor_radius": 18.0, "rng_seed": rng_seed + 1}),
    ]


def benchmark_planners(mask, start, goal, rng_seed=0, benchmark_optimal_length=None, planner_overrides=None):
    rows, results = [], {}
    for i, (name, fn, kwargs) in enumerate(default_planner_specs(rng_seed)):
        kw = dict(kwargs)
        if planner_overrides and name in planner_overrides:
            kw.update(planner_overrides[name])
        res = attach_smoothed_path(mask, fn(mask, start, goal, **kw), rng_seed=rng_seed + 10 + i)
        results[name] = res
        rows.append(summarize_planner_result(res, start, goal, benchmark_optimal_length, seed=kw.get("rng_seed")))
    return rows, results


def plot_free_mask(ax, mask):
    ax.imshow(mask, cmap="gray", origin="upper", interpolation="nearest", vmin=0, vmax=1)


def plot_result(ax, mask, res, start, goal, max_points=5000):
    plot_free_mask(ax, mask)
    if res["tree_edges"]:
        ax.add_collection(LineCollection(res["tree_edges"][:max_points], colors="#4f6db8", linewidths=0.6, alpha=0.35))
    elif res["expanded_points"]:
        pts = np.array(res["expanded_points"][:max_points], float)
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c="#f1c40f", alpha=0.35)
    if res["path"]:
        p = np.array(res["path"], float)
        ax.plot(p[:, 0], p[:, 1], color="#d62728", linewidth=2.0, label="путь")
    if res["smoothed_path"]:
        p = np.array(res["smoothed_path"], float)
        ax.plot(p[:, 0], p[:, 1], color="#17becf", linewidth=2.0, linestyle="--", label="сглаженный")
    ax.scatter([start[0]], [start[1]], c="#2ecc71", s=35, marker="o", label="старт")
    ax.scatter([goal[0]], [goal[1]], c="#ff7f0e", s=45, marker="X", label="цель")
    ax.set_xlim(-0.5, mask.shape[1] - 0.5)
    ax.set_ylim(mask.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(f'{res["planner"]}\nуспех={"да" if res["success"] else "нет"}, длина={"нет" if res["cost"] is None else f"{res["cost"]:.2f}"}\nвремя={res["elapsed_ms"]:.1f} мс, посещено={res["visited_count"]}')
    ax.legend(loc="upper right", fontsize=8)


def plot_planner_comparison(mask, start, goal, results):
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 7), squeeze=False)
    for ax, res in zip(axes.ravel(), results.values()):
        plot_result(ax, mask, res, start, goal)
    fig.tight_layout()
    return fig, axes


def plot_single_query_dashboard(rows, benchmark_optimal_length=None):
    planners = [r["planner"] for r in rows]
    x = np.arange(len(planners))
    w = 0.35
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), squeeze=False)
    ax = axes.ravel()
    raw = [math.nan if r["path_length"] is None else r["path_length"] for r in rows]
    smooth = [math.nan if r["smoothed_length"] is None else r["smoothed_length"] for r in rows]
    ax[0].bar(x - w / 2, raw, width=w, label="исходный")
    ax[0].bar(x + w / 2, smooth, width=w, label="сглаженный")
    if benchmark_optimal_length is not None:
        ax[0].axhline(benchmark_optimal_length, color="#d62728", linestyle=":", label="эталон")
    ax[0].set_title("Длина пути")
    ax[0].set_xticks(x, planners)
    ax[0].grid(True, alpha=0.25, axis="y")
    ax[0].legend()
    ax[1].bar(x, [r["time_ms"] for r in rows], color="#ff7f0e"); ax[1].set_title("Время работы"); ax[1].set_xticks(x, planners); ax[1].grid(True, alpha=0.25, axis="y")
    ax[2].bar(x, [r["visited_points"] for r in rows], color="#2ca02c"); ax[2].set_title("Посещенные узлы / сэмплы"); ax[2].set_xticks(x, planners); ax[2].grid(True, alpha=0.25, axis="y")
    ax[3].bar(x, [math.nan if r["detour_ratio"] is None else r["detour_ratio"] for r in rows], color="#9467bd"); ax[3].set_title("Коэффициент обхода"); ax[3].set_xticks(x, planners); ax[3].grid(True, alpha=0.25, axis="y")
    ax[4].bar(x, [math.nan if r["smooth_gain_pct"] is None else r["smooth_gain_pct"] for r in rows], color="#17becf"); ax[4].set_title("Выигрыш от сглаживания"); ax[4].set_xticks(x, planners); ax[4].grid(True, alpha=0.25, axis="y")
    ax[5].bar(x - w / 2, [math.nan if r["turn_sum_deg"] is None else r["turn_sum_deg"] for r in rows], width=w, label="исходный")
    ax[5].bar(x + w / 2, [math.nan if r["smoothed_turn_sum_deg"] is None else r["smoothed_turn_sum_deg"] for r in rows], width=w, label="сглаженный")
    ax[5].set_title("Суммарный угол поворота"); ax[5].set_xticks(x, planners); ax[5].grid(True, alpha=0.25, axis="y"); ax[5].legend()
    fig.tight_layout()
    return fig, axes


def plot_before_after_smoothing(mask, res, start, goal):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), squeeze=False)
    left, right = axes.ravel()
    for ax in (left, right):
        plot_free_mask(ax, mask)
        ax.set_xlim(-0.5, mask.shape[1] - 0.5)
        ax.set_ylim(mask.shape[0] - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.scatter([start[0]], [start[1]], c="#2ecc71", s=40, marker="o")
        ax.scatter([goal[0]], [goal[1]], c="#ff7f0e", s=45, marker="X")
    if res["tree_edges"]:
        left.add_collection(LineCollection(res["tree_edges"], colors="#4f6db8", linewidths=0.5, alpha=0.2))
        right.add_collection(LineCollection(res["tree_edges"], colors="#4f6db8", linewidths=0.5, alpha=0.15))
    if res["expanded_points"] and not res["tree_edges"]:
        pts = np.array(res["expanded_points"], float)
        left.scatter(pts[:, 0], pts[:, 1], s=2, c="#f1c40f", alpha=0.25)
        right.scatter(pts[:, 0], pts[:, 1], s=2, c="#f1c40f", alpha=0.2)
    if res["path"]:
        p = np.array(res["path"], float)
        left.plot(p[:, 0], p[:, 1], color="#d62728", linewidth=2.0, label="исходный путь")
    if res["smoothed_path"]:
        p = np.array(res["smoothed_path"], float)
        right.plot(p[:, 0], p[:, 1], color="#17becf", linewidth=2.0, label="сглаженный путь")
    left.set_title(f'{res["planner"]}: до сглаживания'); left.legend(loc="upper right", fontsize=8)
    right.set_title(f'{res["planner"]}: после сглаживания'); right.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, axes


def unpack_query(query):
    if hasattr(query, "start") and hasattr(query, "goal"):
        return query.start, query.goal, getattr(query, "optimal_length", None)
    if isinstance(query, dict):
        return query["start"], query["goal"], query.get("optimal_length")
    if len(query) == 2:
        return query[0], query[1], None
    return query[0], query[1], query[2]


def benchmark_query_suite(mask, queries, query_indices, seeds=(0, 1, 2), planner_overrides=None):
    detail = []
    for qid in query_indices:
        start, goal, best = unpack_query(queries[qid])
        akw = dict(default_planner_specs(0)[0][2])
        if planner_overrides and "A*" in planner_overrides:
            akw.update(planner_overrides["A*"])
        res = attach_smoothed_path(mask, astar_search(mask, start, goal, **akw), rng_seed=1000 + qid)
        detail.append(summarize_planner_result(res, start, goal, best, qid, 0))
        for seed in seeds:
            for name, fn, kwargs in default_planner_specs(seed)[1:]:
                kw = dict(kwargs)
                if planner_overrides and name in planner_overrides:
                    kw.update(planner_overrides[name])
                res = attach_smoothed_path(mask, fn(mask, start, goal, **kw), rng_seed=seed + qid * 100)
                detail.append(summarize_planner_result(res, start, goal, best, qid, seed))

    out = []
    for planner in sorted({r["planner"] for r in detail}):
        rows = [r for r in detail if r["planner"] == planner]
        good = [r for r in rows if r["success"]]

        def pick(key):
            return [float(r[key]) for r in good if isinstance(r.get(key), (int, float))]

        out.append(
            {
                "planner": planner,
                "runs": len(rows),
                "success_rate_pct": round(mean([1.0 if r["success"] else 0.0 for r in rows]) * 100, 2),
                "mean_time_ms": round(mean([r["time_ms"] for r in rows]), 2),
                "median_time_ms": round(median([r["time_ms"] for r in rows]), 2),
                "mean_visited_points": round(mean([r["visited_points"] for r in rows]), 2),
                "mean_path_length": None if not pick("path_length") else round(mean(pick("path_length")), 3),
                "mean_smoothed_length": None if not pick("smoothed_length") else round(mean(pick("smoothed_length")), 3),
                "mean_optimality_gap_pct": None if not pick("optimality_gap_pct") else round(mean(pick("optimality_gap_pct")), 2),
                "mean_smoothing_gain_pct": None if not pick("smooth_gain_pct") else round(mean(pick("smooth_gain_pct")), 2),
            }
        )
    return detail, out


def plot_suite_overview(detail_rows, aggregate_rows):
    planners = [r["planner"] for r in aggregate_rows]
    x = np.arange(len(planners))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), squeeze=False)
    ax = axes.ravel()
    ax[0].bar(x, [r["success_rate_pct"] for r in aggregate_rows], color="#2ca02c"); ax[0].set_title("Доля успешных запусков"); ax[0].set_xticks(x, planners); ax[0].set_ylim(0, 105); ax[0].grid(True, alpha=0.25, axis="y")
    ax[1].bar(x, [r["mean_time_ms"] for r in aggregate_rows], color="#ff7f0e"); ax[1].set_title("Среднее время работы"); ax[1].set_xticks(x, planners); ax[1].grid(True, alpha=0.25, axis="y")
    ax[2].bar(x, [r["mean_visited_points"] for r in aggregate_rows], color="#1f77b4"); ax[2].set_title("Среднее число посещений"); ax[2].set_xticks(x, planners); ax[2].grid(True, alpha=0.25, axis="y")
    w = 0.35
    ax[3].bar(x - w / 2, [math.nan if r["mean_path_length"] is None else r["mean_path_length"] for r in aggregate_rows], width=w, label="исходный")
    ax[3].bar(x + w / 2, [math.nan if r["mean_smoothed_length"] is None else r["mean_smoothed_length"] for r in aggregate_rows], width=w, label="сглаженный")
    ax[3].set_title("Средняя длина пути"); ax[3].set_xticks(x, planners); ax[3].grid(True, alpha=0.25, axis="y"); ax[3].legend()
    ax[4].bar(x, [math.nan if r["mean_optimality_gap_pct"] is None else r["mean_optimality_gap_pct"] for r in aggregate_rows], color="#d62728"); ax[4].set_title("Средний проигрыш эталону"); ax[4].set_xticks(x, planners); ax[4].grid(True, alpha=0.25, axis="y")
    ax[5].boxplot([[r["time_ms"] for r in detail_rows if r["planner"] == p] for p in planners], labels=planners); ax[5].set_title("Распределение времени работы"); ax[5].grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    return fig, axes


def make_planner_animation(mask, res, start, goal, max_frames=180, figsize=(7.0, 7.0)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_free_mask(ax, mask)
    ax.set_xlim(-0.5, mask.shape[1] - 0.5)
    ax.set_ylim(mask.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(f'Анимация поиска: {res["planner"]}')
    ax.scatter([start[0]], [start[1]], c="#2ecc71", s=45, marker="o", label="старт")
    ax.scatter([goal[0]], [goal[1]], c="#ff7f0e", s=50, marker="X", label="цель")
    dots = ax.scatter([], [], s=4, c="#f1c40f", alpha=0.45, label="расширение")
    edges = LineCollection([], colors="#4f6db8", linewidths=0.65, alpha=0.35)
    ax.add_collection(edges)
    raw, = ax.plot([], [], color="#d62728", linewidth=2.0, label="путь")
    smooth, = ax.plot([], [], color="#17becf", linewidth=2.0, linestyle="--", label="сглаженный")
    ax.legend(loc="upper right", fontsize=8)
    exp = res["tree_edges"] if res["tree_edges"] else res["expanded_points"]
    f1 = np.unique(np.linspace(0, len(exp), num=max(2, min(max_frames, len(exp) + 1)), dtype=int))
    f2 = np.unique(np.linspace(0, len(res["path"]), num=max(2, min(25, len(res["path"]) + 1)), dtype=int))
    f3 = np.unique(np.linspace(0, len(res["smoothed_path"]), num=max(2, min(25, len(res["smoothed_path"]) + 1)), dtype=int))

    def arr(points):
        return np.empty((0, 2), float) if not points else np.array(points, float)

    def update(i):
        if i < len(f1):
            n = int(f1[i])
            if res["tree_edges"]:
                edges.set_segments(res["tree_edges"][:n])
            elif n:
                dots.set_offsets(arr(res["expanded_points"][:n]))
        elif i < len(f1) + len(f2):
            if res["tree_edges"]:
                edges.set_segments(res["tree_edges"])
            else:
                dots.set_offsets(arr(res["expanded_points"]))
            n = int(f2[i - len(f1)])
            if n and res["path"]:
                p = np.array(res["path"][:n], float)
                raw.set_data(p[:, 0], p[:, 1])
        else:
            if res["tree_edges"]:
                edges.set_segments(res["tree_edges"])
            else:
                dots.set_offsets(arr(res["expanded_points"]))
            if res["path"]:
                p = np.array(res["path"], float)
                raw.set_data(p[:, 0], p[:, 1])
            n = int(f3[i - len(f1) - len(f2)])
            if n and res["smoothed_path"]:
                p = np.array(res["smoothed_path"][:n], float)
                smooth.set_data(p[:, 0], p[:, 1])
        return edges, dots, raw, smooth

    anim = animation.FuncAnimation(fig, update, frames=len(f1) + len(f2) + len(f3), interval=70, blit=False, repeat=False)
    fig.tight_layout()
    return fig, anim


def save_planner_animation(anim, output_path, fps=15, dpi=120):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        return output
    if output.suffix.lower() == ".mp4":
        try:
            anim.save(output, writer=animation.FFMpegWriter(fps=fps), dpi=dpi)
            return output
        except Exception:
            output = output.with_suffix(".gif")
            anim.save(output, writer=animation.PillowWriter(fps=fps), dpi=dpi)
            return output
    raise ValueError("Animation path must end with .gif or .mp4")


def export_planner_animations(
    mask,
    results,
    start,
    goal,
    output_dir="artifacts",
    planner_filenames=None,
    max_frames=160,
    fps=15,
    dpi=120,
    figsize=(7.0, 7.0),
):
    output_dir = Path(output_dir)
    if planner_filenames is None:
        planner_filenames = {
            "A*": "astar_search.mp4",
            "RRT": "rrt_search.mp4",
            "RRT*": "rrt_star_search.mp4",
        }
    saved = {}
    for planner_name, filename in planner_filenames.items():
        if planner_name not in results:
            continue
        fig, anim = make_planner_animation(
            mask,
            results[planner_name],
            start=start,
            goal=goal,
            max_frames=max_frames,
            figsize=figsize,
        )
        try:
            saved[planner_name] = save_planner_animation(anim, output_dir / filename, fps=fps, dpi=dpi)
        finally:
            plt.close(fig)
    return saved


def plot_resolution_study(rows):
    factors = [r["factor"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), squeeze=False)
    ax = axes.ravel()
    ax[0].plot(factors, [r["time_ms"] for r in rows], marker="o"); ax[0].set_title("Время работы A*"); ax[0].set_xlabel("коэффициент уменьшения"); ax[0].grid(True, alpha=0.3)
    ax[1].plot(factors, [r["visited_points"] for r in rows], marker="o", color="#ff7f0e"); ax[1].set_title("Посещенные узлы A*"); ax[1].set_xlabel("коэффициент уменьшения"); ax[1].grid(True, alpha=0.3)
    ax[2].plot(factors, [math.nan if r["path_length"] is None else r["path_length"] for r in rows], marker="o", color="#2ca02c"); ax[2].set_title("Длина пути A*"); ax[2].set_xlabel("коэффициент уменьшения"); ax[2].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, axes
