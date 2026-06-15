"""
semantic_map.py

Builds a 3D object coordinate index for an HM3D scene by parsing vertex colors
from <scene>.semantic.glb.  The companion <scene>.semantic.txt maps each RGB
color to a category name; vertex colors in the GLB encode semantic labels
directly, bypassing Habitat's built-in semantic sensor (which requires
scene-instance JSON files not shipped with the minival split).
"""

import trimesh
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# Chinese instruction keyword → HM3D semantic category list
CHINESE_TO_CATEGORY = {
    "沙发": ["sofa", "couch", "armchair"],
    "床":   ["bed"],
    "椅子": ["chair", "armchair", "seat"],
    "桌子": ["table", "dining table", "desk", "coffee table"],
    "厕所": ["toilet"],
    "水槽": ["sink", "basin", "bath sink"],
    "门":   ["door"],
    "窗户": ["window"],
    "电视": ["tv", "led tv", "television"],
    "柜子": ["cabinet", "wardrobe", "dresser"],
    "冰箱": ["refrigerator", "fridge"],
    "镜子": ["mirror"],
}

# Background / structural categories to exclude from the object index
IGNORE_CATEGORIES = {
    "wall", "floor", "ceiling", "unknown", "door frame", "window frame",
    "wall hanging decoration", "stairs", "vent", "ventilation",
}

DATA_DIR = Path("/data3/liangjy/vln/data/hm3d")


def _parse_semantic_txt(scene_dir: Path) -> dict:
    """Parse <scene>.semantic.txt and return {(r, g, b): category_name}."""
    scene_id = scene_dir.name.split("-", 1)[1]
    txt_path = scene_dir / f"{scene_id}.semantic.txt"
    color_to_cat = {}
    with open(txt_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",", 3)
            if len(parts) >= 3:
                hex_col = parts[1].strip().lower()
                cat = parts[2].strip('"').lower()
                r = int(hex_col[0:2], 16)
                g = int(hex_col[2:4], 16)
                b = int(hex_col[4:6], 16)
                color_to_cat[(r, g, b)] = cat
    return color_to_cat


def build_semantic_map(scene_dir: str, cache: bool = True) -> dict:
    """
    Parse vertex colors from <scene>.semantic.glb and return a dict
    {category: [[x, y, z], ...]} where each entry is the centroid of one
    0.5 m grid cell occupied by that category.

    Results are cached in scene_dir/semantic_cache.json to avoid re-parsing
    the GLB on every run.
    """
    scene_dir = Path(scene_dir)
    cache_path = scene_dir / "semantic_cache.json"

    if cache and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    print(f"[SemanticMap] Building map for {scene_dir.name} ...")
    scene_id = scene_dir.name.split("-", 1)[1]
    sem_glb_path = scene_dir / f"{scene_id}.semantic.glb"

    color_to_cat = _parse_semantic_txt(scene_dir)

    glb_scene = trimesh.load(str(sem_glb_path), force="scene")

    cat_vertices: dict = defaultdict(list)

    for _key, mesh in glb_scene.geometry.items():
        try:
            vc = mesh.visual.to_color().vertex_colors[:, :3]  # (N, 3) RGB uint8
            verts = mesh.vertices                              # (N, 3) XYZ float
        except Exception:
            continue

        for i in range(len(verts)):
            c = (int(vc[i, 0]), int(vc[i, 1]), int(vc[i, 2]))
            cat = color_to_cat.get(c)
            if cat and cat not in IGNORE_CATEGORIES:
                cat_vertices[cat].append(verts[i].tolist())

    # Cluster vertices per category using a 0.5 m voxel grid; store cell centroids
    result = {}
    for cat, verts in cat_vertices.items():
        arr = np.array(verts)
        quantized = np.round(arr / 0.5).astype(int)
        unique_cells = np.unique(quantized, axis=0)
        centers = []
        for cell in unique_cells:
            mask = np.all(quantized == cell, axis=1)
            center = arr[mask].mean(axis=0).tolist()
            centers.append([round(v, 3) for v in center])
        result[cat] = centers

    if cache:
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[SemanticMap] Cached to {cache_path}")

    return result


def query_target(scene_dir: str, chinese_target: str) -> list:
    """Return a list of 3D positions for a Chinese target word (e.g. '沙发')."""
    sem_map = build_semantic_map(scene_dir)
    categories = CHINESE_TO_CATEGORY.get(chinese_target, [chinese_target.lower()])
    results = []
    for cat in categories:
        results.extend(sem_map.get(cat, []))
    return results


def nearest_target(robot_pos: list, positions: list):
    """Return the position in *positions* closest to *robot_pos* (Euclidean)."""
    if not positions:
        return None
    robot = np.array(robot_pos)
    dists = [np.linalg.norm(np.array(p) - robot) for p in positions]
    return positions[int(np.argmin(dists))]


if __name__ == "__main__":
    scene = str(DATA_DIR / "00800-TEEsavR23oF")
    print(f"Scene: {scene}\n")

    sem_map = build_semantic_map(scene, cache=True)

    print(f"\n=== Scene categories ({len(sem_map)} total) ===")
    for cat, positions in sorted(sem_map.items()):
        print(f"  {cat}: {len(positions)} instance(s)")

    print("\n=== Target query test ===")
    for target in ["沙发", "床", "椅子", "桌子", "厕所", "镜子"]:
        positions = query_target(scene, target)
        print(f"  {target}: {len(positions)} position(s)")
        if positions:
            print(f"    example: {positions[0]}")
