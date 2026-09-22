"""Board textures for the rig's six AprilGrid configurations, plus a logo.

The rig declares six grids that share one tag pool. They differ by layout, not
by tag id, and the solver tells them apart by a hash of the layout:

    3x1   ids 0 3 6                 increment 3
    2x2   ids 0 2 4 6               increment 2
    4x7   ids 0..27                 base order
    4x7   ids 0 14 1 15 ... 13 27   offset 14
    4x7   ids 0 15 1 16 ... 13 28   offset 15
    4x7   ids 0 16 1 17 ... 13 29   offset 16

This file does not change create_aprilgrid.py. It reuses create_grid for the
black squares and add_padding for the border, and runs its own tag pass so the
tag order is an argument instead of a hardcoded list.
"""

import os

import torch

from threedgrut_playground.utils.create_aprilgrid import (
    add_padding, create_grid, create_tag)

TAG_ORDERS = {
    "base":  list(range(28)),
    "off14": [v for i in range(14) for v in (i, i + 14)],
    "off15": [v for i in range(14) for v in (i, i + 15)],
    "off16": [v for i in range(14) for v in (i, i + 16)],
}

BOARD_SPECS = [
    # aprilgrid MUST stay first: a quad added with no material_name gets
    # material id 0 (mesh_io zeroes material_assignments), and the default
    # no-PLAYGROUND_BOARDS launch must show the original single 4x7 board.
    ("aprilgrid",  4, 7, TAG_ORDERS["base"]),   # the 4x7 base board
    ("grid_3x1",   3, 1, [0, 3, 6]),
    ("grid_2x2",   2, 2, [0, 2, 4, 6]),
    ("grid_off14", 4, 7, TAG_ORDERS["off14"]),
    ("grid_off15", 4, 7, TAG_ORDERS["off15"]),
    ("grid_off16", 4, 7, TAG_ORDERS["off16"]),
    ("grid_3x3",   3, 3, list(range(9))),
]


def row_major_to_texture_order(ids, rows, cols):
    """Reorder ids for the texture, which walks the top row first.

    The solver lists tag ids bottom row first, left to right. The texture walks
    the other way. This reproduces create_aprilgrid.py's hardcoded list
    [21, 22, ..., 6] exactly from range(28).
    """
    grid = [ids[r * cols:(r + 1) * cols] for r in range(rows)]
    return [v for row in reversed(grid) for v in row]


def _tag_pass(square, gap, rows, cols, texture, offset, tag_order, device):
    """Draw one tag per cell, in the given order. Mirrors create_grid's layout."""
    idx = 0
    for row in range(rows):
        for col in range(cols):
            start_row = row * int(square + gap + 2 * offset) + offset
            start_col = col * int(square + gap + 2 * offset) + offset
            create_tag(start_row, start_row + square,
                       start_col, start_col + square,
                       square, texture, tagID=tag_order[idx], device=device)
            idx += 1
    return texture


def make_texture(gap_ratio, square, device, rows, cols, tag_order):
    if len(tag_order) < rows * cols:
        raise ValueError(f"tag_order holds {len(tag_order)} ids, "
                         f"a {rows}x{cols} board needs {rows * cols}")
    board_gap = square * gap_ratio
    width = int(square * cols + board_gap * (cols - 1))
    height = int(square * rows + board_gap * (rows - 1))
    square, gap = int(square), int(board_gap)

    tex = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device,
                       dtype=torch.float32).repeat(height, width, 1)
    tex = create_grid(square, gap, 0.0, rows, cols, tex)
    tex = _tag_pass(int(square * 0.5), gap, rows, cols, tex,
                    int(square / 4), tag_order, device)
    return add_padding(width, height, gap, tex, device=device)


def build_materials(device, PBRMaterial, logo_path=None, square=800):
    """One material per board plus the logo. Ids run from 0 with no gaps.

    The gap matters: engine.py sorts materials by material_id and passes the
    list positionally, so a hole makes every later index name the wrong
    material. The original file had one material at id 1 and nothing at id 0,
    which only worked because material_assignments gets clamped to 0.
    """
    def mat(mid, tex):
        return PBRMaterial(
            material_id=mid,
            diffuse_map=tex.contiguous(),
            diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
            emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
            metallic_factor=0.0,
            roughness_factor=0.0,
            transmission_factor=0.0,
            ior=1.0,
        )

    materials = {}
    for mid, (name, rows, cols, ids) in enumerate(BOARD_SPECS):
        order = row_major_to_texture_order(ids, rows, cols)
        materials[name] = mat(mid, make_texture(0.3, square, device,
                                                rows, cols, order))

    # Do NOT alias. ps_gui.py:1377 maps material_id to dict POSITION, so a
    # second dict entry pointing at the same object shifts every later index
    # and the material dropdown selects the wrong thing.

    if logo_path and os.path.exists(logo_path):
        from PIL import Image
        import numpy as np
        # Flatten alpha onto white. A logo PNG has a transparent background,
        # and this material does not set alpha_mode, so the quad renders as
        # nothing.
        src = Image.open(logo_path).convert("RGBA")
        bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
        img = np.asarray(Image.alpha_composite(bg, src), dtype=np.float32) / 255.0
        materials["vilota_logo"] = mat(
            len(BOARD_SPECS),
            torch.from_numpy(img).to(device=device, dtype=torch.float32))
    else:
        print(f"[playground] no logo at {logo_path}, skipping the logo material")

    return materials


BOARD_NAMES = [s[0] for s in BOARD_SPECS] + ["vilota_logo"]


# ---------------------------------------------------------------------------
# Scene spawning

DEFAULT_SCENE = ["aprilgrid", "grid_off14", "grid_off15"]


def _axis(spec, default):
    spec = (spec or default).strip().lower()
    return {"x": 0, "y": 1, "z": 2}[spec.lstrip("+-")], (-1.0 if spec.startswith("-") else 1.0)


def board_half_extents(name, tag_cm):
    for spec_name, rows, cols, _ids in BOARD_SPECS:
        if spec_name == name:
            return ((cols + 0.3 * (cols - 1) + 0.6) * tag_cm * 0.01 / 2.0,
                    (rows + 0.3 * (rows - 1) + 0.6) * tag_cm * 0.01 / 2.0)
    return None


# Vertical spacing of the default scene. The GUI's Square size box defaults to
# 15 cm, and resizing scales a board in place, so the spawn positions must
# already leave room for boards at that size: neighbours are spaced by the
# tallest board's height at LAYOUT_SQUARE_CM plus VERTICAL_GAP_FRACTION of it
# as clear gap. (The old angular layout spaced for the 5.173 cm spawn size;
# resized to 15 cm, neighbours overlapped ~0.41 m in y and near boards hid
# far ones, the v2 "off14 hid base" failure.)
LAYOUT_SQUARE_CM = 15.0
VERTICAL_GAP_FRACTION = 0.2


def board_size_m(name, square_cm):
    """Full (width, height) of a board in metres at the given tag square."""
    half = board_half_extents(name, square_cm)
    if half is None:
        raise ValueError(f"{name} is not in BOARD_SPECS")
    return 2.0 * half[0], 2.0 * half[1]


def vertical_step(names, square_cm=LAYOUT_SQUARE_CM):
    """Centre-to-centre vertical offset that keeps neighbours clear."""
    tallest = max(board_size_m(n, square_cm)[1] for n in names)
    return tallest * (1.0 + VERTICAL_GAP_FRACTION)


def layout_positions(names, tag_cm, near, span,
                     up_i=1, up_s=1.0, fw_i=2, fw_s=-1.0,
                     square_cm=LAYOUT_SQUARE_CM):
    """Default-scene placement, pure so tests can run it without the engine.

    Boards stack vertically around 0 with vertical_step() between centres
    (first board at the bottom) and stagger in depth exactly as before:
    board i sits at near * (1 + span * i/(n-1)).
    Returns [(name, [x, y, z])].
    """
    n = len(names)
    step = vertical_step(names, square_cm)
    placed = []
    for i, name in enumerate(names):
        frac = 0.0 if n == 1 else i / (n - 1)
        dist = near * (1.0 + span * frac)
        pos = [0.0, 0.0, 0.0]
        pos[fw_i] = fw_s * dist
        pos[up_i] = up_s * step * (i - (n - 1) / 2.0)
        placed.append((name, pos))
    return placed


def board_aabbs(placed, square_cm, up_i=1, fw_i=2):
    """Face-on 2D AABBs, one per placed board, at the given square size.

    Projects onto the plane orthogonal to the forward axis - the plane the
    rig actually sees. Boards all share the side coordinate, so an overlap
    here means one board hides another from the view line.
    Returns [(name, (side_min, up_min), (side_max, up_max))].
    """
    side_i = 3 - up_i - fw_i
    out = []
    for name, pos in placed:
        w, h = board_size_m(name, square_cm)
        out.append((name, (pos[side_i] - w / 2.0, pos[up_i] - h / 2.0),
                          (pos[side_i] + w / 2.0, pos[up_i] + h / 2.0)))
    return out


def check_layout_no_overlap(names, tag_cm=5.173, near=None, span=0.6,
                            square_cm=LAYOUT_SQUARE_CM):
    """Raise if any two boards would overlap face-on at square_cm.

    Uses the same layout_positions the spawner uses. Returns the AABBs so
    tests can also assert on the actual gaps. No renders involved.
    """
    if near is None:
        near = 0.20 * tag_cm
    placed = layout_positions(names, tag_cm, near, span, square_cm=square_cm)
    aabbs = board_aabbs(placed, square_cm)
    for i in range(len(aabbs)):
        for j in range(i + 1, len(aabbs)):
            (na, amin, amax), (nb, bmin, bmax) = aabbs[i], aabbs[j]
            if all(amin[k] < bmax[k] and bmin[k] < amax[k] for k in range(2)):
                raise AssertionError(
                    f"{na} and {nb} overlap face-on at {square_cm} cm "
                    f"squares: {amin}..{amax} vs {bmin}..{bmax}")
    return aabbs


def spawn_scene(engine, primitive_type, device):
    """Place boards at different heights AND different distances.

    Boards stack vertically with vertical_step() between centres, sized so
    that even after the GUI resizes every board to LAYOUT_SQUARE_CM squares
    the boards stay clear of each other face-on (see check_layout_no_overlap).
    Depth staggers as before: the near distance defaults to about
    0.2 * tag_cm metres, the far board sits at near * (1 + span).

    Environment:
        PLAYGROUND_BOARDS   1, a comma separated list of names, or 'all'
        BOARD_TAG_CM        tag pitch. Default 5.173, the rig's real size
        BOARD_DISTANCE      near distance in metres. Default 0.2 * BOARD_TAG_CM
        BOARD_DEPTH_SPAN    far board sits at distance * (1 + this). Default 0.6
        BOARD_ANGLES        degrees off axis, one per board. Setting this
                            restores the old angular layout, which overlaps
                            once boards are resized to 15 cm squares
        BOARD_UP_AXIS       default y
        BOARD_FWD_AXIS      default -z
    """
    import math
    import os

    scene_file = os.environ.get("BOARD_SCENE")
    if scene_file:
        return spawn_from_file(engine, primitive_type, device, scene_file)

    want = os.environ.get("PLAYGROUND_BOARDS", "").strip()
    if want.lower() == "all":
        names = list(BOARD_NAMES)
    elif want and want != "1":
        names = [n.strip() for n in want.split(",") if n.strip()]
    else:
        names = list(DEFAULT_SCENE)

    tag_cm = float(os.environ.get("BOARD_TAG_CM", 5.173))
    near = float(os.environ.get("BOARD_DISTANCE", 0.20 * tag_cm))
    span = float(os.environ.get("BOARD_DEPTH_SPAN", 0.6))
    up_i, up_s = _axis(os.environ.get("BOARD_UP_AXIS"), "y")
    fw_i, fw_s = _axis(os.environ.get("BOARD_FWD_AXIS"), "-z")

    n = len(names)
    raw = os.environ.get("BOARD_ANGLES")
    if raw:
        # Old angular layout, kept as an explicit override.
        angles = [float(a) for a in raw.split(",")]
        angles = (angles + [0.0] * n)[:n]
        placed = []
        for i, (name, ang) in enumerate(zip(names, angles)):
            frac = 0.0 if n == 1 else i / (n - 1)
            dist = near * (1.0 + span * frac)
            pos = [0.0, 0.0, 0.0]
            pos[fw_i] = fw_s * dist
            pos[up_i] = up_s * dist * math.tan(math.radians(ang))
            placed.append((name, pos))
        print(f"[playground] BOARD_ANGLES set: angular layout, overlaps at "
              f"{LAYOUT_SQUARE_CM:.0f} cm squares")
    else:
        placed = layout_positions(names, tag_cm, near, span,
                                  up_i=up_i, up_s=up_s, fw_i=fw_i, fw_s=fw_s)

    step = vertical_step(names)
    print(f"[playground] {n} board(s), tag {tag_cm:.3f} cm, "
          f"near {near:.2f} m, far {near * (1 + span):.2f} m, "
          f"vertical step {step:.2f} m (clear up to "
          f"{LAYOUT_SQUARE_CM:.0f} cm squares)")

    for name, pos in placed:
        if name not in engine.primitives.registered_materials:
            print(f"[playground]   skip {name}, no such material")
            continue
        before = set(engine.primitives.objects)
        engine.primitives.add_primitive(
            geometry_type="Quad", primitive_type=primitive_type,
            device=device, material_name=name)
        added = set(engine.primitives.objects) - before
        if not added:
            print(f"[playground]   {name} did not spawn")
            continue
        obj = engine.primitives.objects[added.pop()]

        obj.transform.tx, obj.transform.ty, obj.transform.tz = pos

        size = board_half_extents(name, tag_cm)
        if size is not None:
            obj.transform.sx, obj.transform.sy = size
        print(f"[playground]   {name:<12} "
              f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})"
              + (f"  sx {size[0]:.4f} sy {size[1]:.4f}" if size else "  (size by hand)"))


def spawn_from_file(engine, primitive_type, device, path):
    """Place boards from a JSON scene file. See office_scene.json.

    Each entry: material (required), pos [x,y,z] (required), and either
    tag_cm for a board or sx/sy for anything else. rot [rx,ry,rz] degrees
    is optional and defaults to flat.
    """
    import json
    spec = json.load(open(path))
    boards = spec.get("boards", [])
    print(f"[playground] scene file {path}, {len(boards)} board(s)")
    for entry in boards:
        name = entry["material"]
        if name not in engine.primitives.registered_materials:
            print(f"[playground]   skip {name}, no such material")
            continue
        before = set(engine.primitives.objects)
        engine.primitives.add_primitive(
            geometry_type="Quad", primitive_type=primitive_type,
            device=device, material_name=name)
        added = set(engine.primitives.objects) - before
        if not added:
            print(f"[playground]   {name} did not spawn")
            continue
        obj = engine.primitives.objects[added.pop()]

        pos = entry["pos"]
        obj.transform.tx, obj.transform.ty, obj.transform.tz = pos
        rot = entry.get("rot", [0.0, 0.0, 0.0])
        obj.transform.rx, obj.transform.ry, obj.transform.rz = rot

        if "tag_cm" in entry:
            size = board_half_extents(name, float(entry["tag_cm"]))
        elif "sx" in entry:
            size = (float(entry["sx"]), float(entry["sy"]))
        else:
            size = None
        if size is not None:
            obj.transform.sx, obj.transform.sy = size
        print(f"[playground]   {name:<12} pos ({pos[0]:+.2f}, {pos[1]:+.2f}, "
              f"{pos[2]:+.2f}) rot ({rot[0]:+.1f}, {rot[1]:+.1f}, {rot[2]:+.1f})"
              + (f" sx {size[0]:.4f} sy {size[1]:.4f}" if size else ""))
