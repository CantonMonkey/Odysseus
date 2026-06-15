"""
semantic_map.py
通过解析 HM3D semantic.glb vertex colors，构建场景物体 3D 坐标索引。
semantic.txt 提供 color->category 映射，semantic.glb 的顶点颜色直接对应语义类别。
"""

import trimesh
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# 中文指令 -> HM3D 语义类别映射
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

# 不感兴趣的背景类别（过滤掉）
IGNORE_CATEGORIES = {
    "wall", "floor", "ceiling", "unknown", "door frame", "window frame",
    "wall hanging decoration", "stairs", "vent", "ventilation",
}

DATA_DIR = Path("/data3/liangjy/vln/data/hm3d")


def _parse_semantic_txt(scene_dir: Path) -> dict:
    """返回 {(r,g,b): category_name}"""
    scene_id = scene_dir.name.split("-", 1)[1]
    txt_path = scene_dir / f"{scene_id}.semantic.txt"
    color_to_cat = {}
    with open(txt_path) as f:
        next(f)
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
    解析 semantic.glb vertex colors，返回 {category: [[x,y,z], ...]} 物体坐标表。
    每个坐标是该类别某个实例的近似中心点。
    结果缓存到 scene_dir/semantic_cache.json。
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

    # 加载 semantic.glb，提取所有顶点颜色和坐标
    glb_scene = trimesh.load(str(sem_glb_path), force="scene")

    # 按类别聚合顶点坐标
    cat_vertices = defaultdict(list)

    for key, mesh in glb_scene.geometry.items():
        try:
            vc = mesh.visual.to_color().vertex_colors[:, :3]  # (N, 3) RGB
            verts = mesh.vertices  # (N, 3) xyz
        except Exception:
            continue

        # 对每个顶点查找类别
        for i in range(len(verts)):
            c = (int(vc[i, 0]), int(vc[i, 1]), int(vc[i, 2]))
            cat = color_to_cat.get(c)
            if cat and cat not in IGNORE_CATEGORIES:
                cat_vertices[cat].append(verts[i].tolist())

    # 对每个类别，做简单聚类：合并距离很近的顶点群（>0.5m 分开）
    # 用均值量化：把同一类别的顶点按 0.5m 格子分组，取每组中心
    result = {}
    for cat, verts in cat_vertices.items():
        arr = np.array(verts)
        # 量化到 0.5m 格子
        quantized = np.round(arr / 0.5).astype(int)
        unique_cells = np.unique(quantized, axis=0)
        centers = []
        for cell in unique_cells:
            mask = np.all(quantized == cell, axis=1)
            center = arr[mask].mean(axis=0).tolist()
            centers.append([round(c, 3) for c in center])
        result[cat] = centers

    if cache:
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[SemanticMap] Cached to {cache_path}")

    return result


def query_target(scene_dir: str, chinese_target: str) -> list:
    """根据中文目标词，返回场景中对应物体的 3D 坐标列表。"""
    sem_map = build_semantic_map(scene_dir)
    categories = CHINESE_TO_CATEGORY.get(chinese_target, [chinese_target.lower()])
    results = []
    for cat in categories:
        results.extend(sem_map.get(cat, []))
    return results


def nearest_target(robot_pos: list, positions: list):
    """从候选坐标列表中返回距离机器人最近的一个。"""
    if not positions:
        return None
    robot = np.array(robot_pos)
    dists = [np.linalg.norm(np.array(p) - robot) for p in positions]
    return positions[int(np.argmin(dists))]


if __name__ == "__main__":
    scene = str(DATA_DIR / "00800-TEEsavR23oF")
    print(f"Scene: {scene}\n")

    sem_map = build_semantic_map(scene, cache=True)

    print(f"\n=== 场景物体类别（共 {len(sem_map)} 类）===")
    for cat, positions in sorted(sem_map.items()):
        print(f"  {cat}: {len(positions)} 个实例")

    print("\n=== 目标查询测试 ===")
    for target in ["沙发", "床", "椅子", "桌子", "厕所", "镜子"]:
        positions = query_target(scene, target)
        print(f"  {target}: {len(positions)} 个位置")
        if positions:
            print(f"    坐标示例: {positions[0]}")
