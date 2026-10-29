"""Export per-frame camera/aircraft state from the .blend for the cloud renderer.

For every frame, the active camera is evaluated at SUBSAMPLES instants spread over
the shutter interval Cycles uses (centred, 0.5 frame, endpoints included), so the
ray-marched cloud layer gets the same motion blur as the Cycles aircraft layer.

  python export_frames.py --blend B --out frame_data.json [--subsamples 9]
"""
import argparse
import json
import math
import os
import sys

import bpy  # noqa: I001
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as TL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--subsamples", type=int, default=9)
args = ap.parse_args(sys.argv[1:])

bpy.ops.wm.open_mainfile(filepath=args.blend, load_ui=False)
scene = bpy.context.scene
jet = bpy.data.objects["JET"]
shutter = scene.render.motion_blur_shutter
S = args.subsamples


def cam_record(cam):
    m = cam.matrix_world
    loc = m.translation
    right = m.col[0].xyz.normalized()
    up = m.col[1].xyz.normalized()
    fwd = -m.col[2].xyz.normalized()
    d = cam.data
    # sensor_fit AUTO: the sensor width spans the larger (vertical) image dimension
    tan_half = (d.sensor_width / 2.0) / d.lens
    return [*loc, *right, *up, *fwd, tan_half]


def set_time(t):
    f = math.floor(t)
    scene.frame_set(int(f), subframe=float(t - f))


frames = {}
for f in range(scene.frame_start, scene.frame_end + 1):
    cam = bpy.data.objects[TL.active_camera(f)]
    subs = []
    for k in range(S):
        t = f + shutter * (k / (S - 1) - 0.5)
        set_time(t)
        subs.append(cam_record(cam))
    set_time(f)
    mj = jet.matrix_world
    nozzle = mj @ Vector((0.0, TL.NOZZLE_EXIT_Y - 0.3, TL.ENGINE_Z))
    fwd = (mj.to_3x3() @ Vector((0, 1, 0))).normalized()
    centre = cam_record(cam)
    co, cf = Vector(centre[0:3]), Vector(centre[9:12])
    nozzle_depth = (nozzle - co).dot(cf)
    jet_depth = (mj.translation - co).dot(cf)
    frames[f] = {
        "camera": TL.active_camera(f),
        "cams": subs,
        "cam_centre": centre,
        "nozzle": list(nozzle),
        "nozzles": [list(mj @ Vector((sx * 0.68, TL.NOZZLE_EXIT_Y, TL.ENGINE_Z))) for sx in (1, -1)],
        "jet_fwd": list(fwd),
        "jet_loc": list(mj.translation),
        "jet_matrix": [list(row) for row in mj],
        "ab_len": float(scene["ab_len"]),
        "nozzle_depth": nozzle_depth,
        "jet_depth": jet_depth,
        "ab": float(scene["ab"]),
        "trail": float(scene["trail"]),
        "flick": float(scene["flick"]),
        "time": float(scene["time"]),
    }

with open(args.out, "w") as fh:
    json.dump({"width": scene.render.resolution_x, "height": scene.render.resolution_y,
               "fps": scene.render.fps, "subsamples": S, "frames": frames}, fh)
print("EXPORTED", len(frames), "frames ->", args.out)
