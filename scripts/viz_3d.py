"""Interactive 3D viewer for the reconstructed tennis ball trajectory.

Opens an open3d window with:
  * Court surface (light blue rectangle at z=0)
  * Court line markings (singles court, service lines, net base)
  * Net (vertical mesh at y=0)
  * Ball trajectory as a coloured line strip (gradient by frame index)
  * Bounce points as red spheres

Mouse: orbit / pan / zoom. Press H in the window for default key bindings.

Usage:
    python scripts/viz_3d.py data/game1_clip1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from matplotlib import cm


DOUBLES_HALF_W = 5.485
SINGLES_HALF_W = 4.115
COURT_HALF_L = 11.885
SERVICE_LINE = 6.40
NET_HEIGHT = 0.914


def lineset(pts: np.ndarray, edges: list[tuple[int, int]],
            color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


def court_geometries() -> list[o3d.geometry.Geometry]:
    geoms: list[o3d.geometry.Geometry] = []

    # Surface mesh (single coloured quad)
    surface = o3d.geometry.TriangleMesh()
    surface.vertices = o3d.utility.Vector3dVector(
        np.array([
            [-DOUBLES_HALF_W, -COURT_HALF_L, 0],
            [+DOUBLES_HALF_W, -COURT_HALF_L, 0],
            [+DOUBLES_HALF_W, +COURT_HALF_L, 0],
            [-DOUBLES_HALF_W, +COURT_HALF_L, 0],
        ], dtype=np.float64)
    )
    surface.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]]))
    surface.paint_uniform_color([0.55, 0.70, 0.85])
    surface.compute_vertex_normals()
    geoms.append(surface)

    # Lines (white)
    pts = []
    edges = []

    def add_seg(a, b):
        i0 = len(pts)
        pts.append(a)
        pts.append(b)
        edges.append((i0, i0 + 1))

    # Singles court rectangle
    add_seg([-SINGLES_HALF_W, -COURT_HALF_L, 0], [+SINGLES_HALF_W, -COURT_HALF_L, 0])
    add_seg([-SINGLES_HALF_W, +COURT_HALF_L, 0], [+SINGLES_HALF_W, +COURT_HALF_L, 0])
    add_seg([-SINGLES_HALF_W, -COURT_HALF_L, 0], [-SINGLES_HALF_W, +COURT_HALF_L, 0])
    add_seg([+SINGLES_HALF_W, -COURT_HALF_L, 0], [+SINGLES_HALF_W, +COURT_HALF_L, 0])
    # Doubles outer (separate dim colour later if needed)
    add_seg([-DOUBLES_HALF_W, -COURT_HALF_L, 0], [+DOUBLES_HALF_W, -COURT_HALF_L, 0])
    add_seg([-DOUBLES_HALF_W, +COURT_HALF_L, 0], [+DOUBLES_HALF_W, +COURT_HALF_L, 0])
    add_seg([-DOUBLES_HALF_W, -COURT_HALF_L, 0], [-DOUBLES_HALF_W, +COURT_HALF_L, 0])
    add_seg([+DOUBLES_HALF_W, -COURT_HALF_L, 0], [+DOUBLES_HALF_W, +COURT_HALF_L, 0])
    # Service lines + center service line
    add_seg([-SINGLES_HALF_W, +SERVICE_LINE, 0], [+SINGLES_HALF_W, +SERVICE_LINE, 0])
    add_seg([-SINGLES_HALF_W, -SERVICE_LINE, 0], [+SINGLES_HALF_W, -SERVICE_LINE, 0])
    add_seg([0, -SERVICE_LINE, 0], [0, +SERVICE_LINE, 0])
    geoms.append(lineset(np.asarray(pts), edges, (1.0, 1.0, 1.0)))

    # Net (translucent mesh): vertical strip at y=0
    net = o3d.geometry.TriangleMesh()
    net.vertices = o3d.utility.Vector3dVector(
        np.array([
            [-DOUBLES_HALF_W, 0, 0],
            [+DOUBLES_HALF_W, 0, 0],
            [+DOUBLES_HALF_W, 0, NET_HEIGHT],
            [-DOUBLES_HALF_W, 0, NET_HEIGHT],
        ], dtype=np.float64)
    )
    net.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]]))
    net.paint_uniform_color([0.15, 0.15, 0.15])
    net.compute_vertex_normals()
    geoms.append(net)

    # World axes at the origin (centre of court, on the surface)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6))
    return geoms


def trajectory_geometries(df: pd.DataFrame, bounces: pd.DataFrame | None
                          ) -> list[o3d.geometry.Geometry]:
    geoms: list[o3d.geometry.Geometry] = []
    df = df.sort_values("idx").reset_index(drop=True)
    pts = df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    # Detect arc breaks by frame discontinuity (>1 frame gap = new arc)
    frames = df["idx"].to_numpy(dtype=np.int64)
    arc_id = np.zeros(len(df), dtype=np.int64)
    if len(df) > 1:
        gaps = np.where(np.diff(frames) > 1)[0]
        for g in gaps:
            arc_id[g + 1:] += 1

    # Coloured polyline per arc (gradient across the whole trajectory)
    edges = []
    cmap = cm.get_cmap("viridis")
    colors = []
    for i in range(len(df) - 1):
        if arc_id[i] != arc_id[i + 1]:
            continue                                 # don't connect across arcs
        edges.append((i, i + 1))
        t = i / max(1, len(df) - 1)
        c = cmap(t)
        colors.append((c[0], c[1], c[2]))
    if edges:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(edges)
        ls.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(ls)

    # Trajectory points as small spheres (combine into one mesh for speed)
    pt_mesh = o3d.geometry.TriangleMesh()
    base = o3d.geometry.TriangleMesh.create_sphere(radius=0.07, resolution=8)
    base.compute_vertex_normals()
    for i, p in enumerate(pts):
        s = o3d.geometry.TriangleMesh(base)
        s.translate(p, relative=False)
        c = cmap(i / max(1, len(df) - 1))
        s.paint_uniform_color([c[0], c[1], c[2]])
        pt_mesh += s
    geoms.append(pt_mesh)

    # Bounce points as larger red spheres
    if bounces is not None:
        b_mesh = o3d.geometry.TriangleMesh()
        bsph = o3d.geometry.TriangleMesh.create_sphere(radius=0.18, resolution=12)
        bsph.compute_vertex_normals()
        bsph.paint_uniform_color([1.0, 0.1, 0.1])
        for _, r in bounces.iterrows():
            s = o3d.geometry.TriangleMesh(bsph)
            s.translate([r["x"], r["y"], 0.05], relative=False)
            b_mesh += s
        geoms.append(b_mesh)
    return geoms


def render_headless(geoms, out_png: Path,
                    width: int = 1600, height: int = 900,
                    eye=(-7.5, -18.0, 9.0),
                    target=(0.0, 0.0, 1.5),
                    up=(0.0, 0.0, 1.0)) -> None:
    """Render the scene to a PNG with Open3D's offscreen renderer."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([0.96, 0.97, 0.99, 1.0])
    mat_lit = o3d.visualization.rendering.MaterialRecord()
    mat_lit.shader = "defaultLit"
    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 2.5
    for i, g in enumerate(geoms):
        name = f"g{i}"
        if isinstance(g, o3d.geometry.LineSet):
            scene.add_geometry(name, g, mat_line)
        else:
            scene.add_geometry(name, g, mat_lit)
    # camera.look_at(target, eye, up)
    scene.camera.look_at(list(target), list(eye), list(up))
    img = renderer.render_to_image()
    o3d.io.write_image(str(out_png), img)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rally_dir", type=Path)
    p.add_argument("--headless-png", type=Path, default=None,
                   help="If set, render an offscreen PNG instead of opening "
                        "an interactive window (useful when no display).")
    args = p.parse_args()

    df = pd.read_csv(args.rally_dir / "ball_traj_3D.csv")
    bounces_csv = args.rally_dir / "bounces.csv"
    bounces = pd.read_csv(bounces_csv) if bounces_csv.exists() else None

    print(f"[viz_3d] {len(df)} trajectory points, "
          f"{0 if bounces is None else len(bounces)} bounces")

    geoms = court_geometries() + trajectory_geometries(df, bounces)

    if args.headless_png is not None:
        render_headless(geoms, args.headless_png)
        print(f"[viz_3d] wrote {args.headless_png}")
        return 0

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Tennis 3D ball trajectory",
        width=1280,
        height=800,
        zoom=0.8,
        front=[0.0, -0.6, 0.5],
        lookat=[0.0, 0.0, 0.5],
        up=[0.0, 0.0, 1.0],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
