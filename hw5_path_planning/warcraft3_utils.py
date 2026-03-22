from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


FREE_TERRAIN = frozenset({".", "G", "S"})


@dataclass(frozen=True)
class WarcraftQuery:
    bucket: int
    start: tuple[int, int]
    goal: tuple[int, int]
    optimal_length: float


@dataclass(frozen=True)
class WarcraftScene:
    scene_path: Path
    scene_name: str
    map_name: str
    map_path: Path | None
    width: int
    height: int
    queries: tuple[WarcraftQuery, ...]


@dataclass(frozen=True)
class WarcraftMap:
    map_path: Path
    map_name: str
    width: int
    height: int
    terrain_rows: tuple[str, ...]
    free_mask: np.ndarray


def _normalize_scene_name(scene_path: Path) -> str:
    return scene_path.name.removesuffix(".map.scen")


def _split_scene_line(line: str) -> list[str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) == 9:
        return parts
    return line.strip().split()


def _resolve_map_path(scene_path: Path, map_name: str) -> Path | None:
    candidates = [
        scene_path.parent / map_name,
        scene_path.parent / "map" / map_name,
        scene_path.parent.parent / "map" / map_name,
        scene_path.parent.parent / map_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_warcraft3_map(map_path: str | Path) -> WarcraftMap:
    map_path = Path(map_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Map file was not found: {map_path}")

    with map_path.open("r", encoding="utf-8") as handle:
        map_type = handle.readline().strip()
        height_line = handle.readline().strip()
        width_line = handle.readline().strip()
        map_header = handle.readline().strip()

        if not map_type.startswith("type"):
            raise ValueError(f"Malformed map file {map_path}: missing 'type' header.")
        if not height_line.startswith("height "):
            raise ValueError(f"Malformed map file {map_path}: missing 'height' header.")
        if not width_line.startswith("width "):
            raise ValueError(f"Malformed map file {map_path}: missing 'width' header.")
        if map_header != "map":
            raise ValueError(f"Malformed map file {map_path}: missing 'map' marker.")

        height = int(height_line.split()[1])
        width = int(width_line.split()[1])
        terrain_rows = tuple(line.rstrip("\n") for line in handle if line.strip())

    if len(terrain_rows) != height:
        raise ValueError(
            f"Malformed map file {map_path}: expected {height} terrain rows, got {len(terrain_rows)}."
        )

    if any(len(row) != width for row in terrain_rows):
        raise ValueError(
            f"Malformed map file {map_path}: one or more rows do not match width {width}."
        )

    free_mask = np.array(
        [[cell in FREE_TERRAIN for cell in row] for row in terrain_rows],
        dtype=bool,
    )

    return WarcraftMap(
        map_path=map_path,
        map_name=map_path.name,
        width=width,
        height=height,
        terrain_rows=terrain_rows,
        free_mask=free_mask,
    )


def read_warcraft3_scene(scene_path: str | Path) -> WarcraftScene:
    scene_path = Path(scene_path)

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file was not found: {scene_path}")

    with scene_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if not header.startswith("version"):
            raise ValueError(
                f"Unsupported scene format in {scene_path}. Expected a 'version' header."
            )

        queries: list[WarcraftQuery] = []
        map_name: str | None = None
        width: int | None = None
        height: int | None = None

        for raw_line in handle:
            if not raw_line.strip():
                continue

            parts = _split_scene_line(raw_line)
            if len(parts) != 9:
                raise ValueError(
                    f"Malformed line in {scene_path}: expected 9 columns, got {len(parts)}"
                )

            bucket, current_map, current_width, current_height, start_x, start_y, goal_x, goal_y, optimal = parts

            if map_name is None:
                map_name = current_map
                width = int(current_width)
                height = int(current_height)

            queries.append(
                WarcraftQuery(
                    bucket=int(bucket),
                    start=(int(start_x), int(start_y)),
                    goal=(int(goal_x), int(goal_y)),
                    optimal_length=float(optimal),
                )
            )

    if map_name is None or width is None or height is None:
        raise ValueError(f"Scene file {scene_path} does not contain any queries.")

    return WarcraftScene(
        scene_path=scene_path,
        scene_name=_normalize_scene_name(scene_path),
        map_name=map_name,
        map_path=_resolve_map_path(scene_path, map_name),
        width=width,
        height=height,
        queries=tuple(queries),
    )


def read_warcraft3_scenes(scene_paths: Iterable[str | Path]) -> list[WarcraftScene]:
    return [read_warcraft3_scene(scene_path) for scene_path in scene_paths]


def read_warcraft3_scenes_from_dir(
    scene_dir: str | Path,
    pattern: str = "*.map.scen",
    recursive: bool = True,
) -> list[WarcraftScene]:
    scene_dir = Path(scene_dir)
    glob = scene_dir.rglob if recursive else scene_dir.glob
    return read_warcraft3_scenes(sorted(glob(pattern)))


def _sample_queries(
    queries: Sequence[WarcraftQuery],
    queries_per_scene: int | None,
) -> list[WarcraftQuery]:
    if queries_per_scene is None or queries_per_scene >= len(queries):
        return list(queries)

    indices = np.linspace(0, len(queries) - 1, queries_per_scene, dtype=int)
    return [queries[index] for index in indices]


def _ensure_scene(scene: WarcraftScene | str | Path) -> WarcraftScene:
    if isinstance(scene, WarcraftScene):
        return scene
    return read_warcraft3_scene(scene)


def _ensure_map(
    scene: WarcraftScene,
    map_cache: dict[Path, WarcraftMap],
) -> WarcraftMap | None:
    if scene.map_path is None:
        return None
    if scene.map_path not in map_cache:
        map_cache[scene.map_path] = read_warcraft3_map(scene.map_path)
    return map_cache[scene.map_path]


def plot_warcraft3_scenes(
    scenes: Sequence[WarcraftScene | str | Path],
    queries_per_scene: int | None = 40,
    cols: int = 3,
    figsize: tuple[float, float] | None = None,
    show_map: bool = True,
    invert_y: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    loaded_scenes = [_ensure_scene(scene) for scene in scenes]
    if not loaded_scenes:
        raise ValueError("At least one Warcraft III scene is required for plotting.")

    cols = max(1, cols)
    rows = ceil(len(loaded_scenes) / cols)
    if figsize is None:
        figsize = (5.5 * cols, 5.5 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    flat_axes = axes.ravel()
    cmap = plt.get_cmap("viridis")
    map_cache: dict[Path, WarcraftMap] = {}

    for index, scene in enumerate(loaded_scenes):
        ax = flat_axes[index]
        sampled_queries = _sample_queries(scene.queries, queries_per_scene)
        buckets = np.array([query.bucket for query in sampled_queries], dtype=float)
        scene_map = _ensure_map(scene, map_cache) if show_map else None

        if scene_map is not None:
            ax.imshow(
                scene_map.free_mask,
                cmap="gray",
                origin="upper",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )

        if len(buckets) == 0:
            colors = [cmap(0.0)]
        elif np.allclose(buckets.max(), buckets.min()):
            colors = [cmap(0.5)] * len(sampled_queries)
        else:
            normalized = (buckets - buckets.min()) / (buckets.max() - buckets.min())
            colors = [cmap(value) for value in normalized]

        for query, color in zip(sampled_queries, colors):
            start_x, start_y = query.start
            goal_x, goal_y = query.goal
            ax.plot(
                [start_x, goal_x],
                [start_y, goal_y],
                color=color,
                alpha=0.25,
                linewidth=1.0,
            )

        start_points = np.array([query.start for query in sampled_queries], dtype=float)
        goal_points = np.array([query.goal for query in sampled_queries], dtype=float)

        if len(start_points) > 0:
            ax.scatter(
                start_points[:, 0],
                start_points[:, 1],
                s=18,
                c="#2ca02c",
                marker="o",
                label="старт" if index == 0 else None,
            )
            ax.scatter(
                goal_points[:, 0],
                goal_points[:, 1],
                s=24,
                c="#d62728",
                marker="x",
                linewidths=1.2,
                label="цель" if index == 0 else None,
            )

        ax.set_title(
            f"{scene.scene_name}\n{len(sampled_queries)}/{len(scene.queries)} запросов"
        )
        ax.set_xlim(-0.5, scene.width - 0.5)
        if scene_map is not None:
            ax.set_ylim(scene.height - 0.5, -0.5)
        else:
            ax.set_ylim(0, scene.height)
        if invert_y:
            ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.2, linewidth=0.5)

    for ax in flat_axes[len(loaded_scenes) :]:
        ax.axis("off")

    flat_axes[0].legend(loc="upper right")
    fig.suptitle("Набор сцен Warcraft III", fontsize=14)
    fig.tight_layout()
    return fig, axes
