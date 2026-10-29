"""Render frames of the saved scene.

  python render.py --blend B --out DIR [--frames 1,40,80 | --range 1 420]
                   [--pct 100] [--samples 48] [--skip-existing]
"""
import argparse
import os
import sys
import time

import bpy  # noqa: I001
import addon_utils

ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="")
ap.add_argument("--range", nargs=2, type=int)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--pct", type=int, default=100)
ap.add_argument("--samples", type=int, default=0)
ap.add_argument("--threshold", type=float, default=0.0)
ap.add_argument("--skip-existing", action="store_true")
ap.add_argument("--no-blur", action="store_true")
ap.add_argument("--no-depth", action="store_true", help="skip the planar depth pass")
args = ap.parse_args(sys.argv[1:])

addon_utils.enable("cycles", default_set=True)
bpy.ops.wm.open_mainfile(filepath=args.blend, load_ui=False)
addon_utils.enable("cycles", default_set=True)
scene = bpy.context.scene
scene.render.resolution_percentage = args.pct
if args.samples:
    scene.cycles.samples = args.samples
if args.threshold:
    scene.cycles.adaptive_threshold = args.threshold
if args.no_blur:
    scene.render.use_motion_blur = False
scene.cycles.device = "CPU"
scene.render.compositor_device = "CPU"

if args.frames:
    frames = [int(x) for x in args.frames.split(",")]
else:
    a, b = args.range if args.range else (scene.frame_start, scene.frame_end)
    frames = list(range(a, b + 1, args.step))

os.makedirs(args.out, exist_ok=True)


def render_depth(frame, path):
    """Planar camera depth of the airframe at half resolution (RGB = depth x coverage, A = coverage)."""
    vl = scene.view_layers[0]
    cy = scene.cycles
    saved = (vl.material_override, cy.samples, cy.max_bounces, cy.use_denoising, scene.render.use_motion_blur,
             scene.render.image_settings.color_depth, cy.use_adaptive_sampling, scene.render.resolution_percentage)
    scene.render.resolution_percentage = max(1, scene.render.resolution_percentage // 2)
    vl.material_override = bpy.data.materials["DepthOverride"]
    cy.samples, cy.max_bounces, cy.use_denoising = 4, 0, False
    scene.render.use_motion_blur = False
    cy.use_adaptive_sampling = False
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    (vl.material_override, cy.samples, cy.max_bounces, cy.use_denoising, scene.render.use_motion_blur,
     scene.render.image_settings.color_depth, cy.use_adaptive_sampling, scene.render.resolution_percentage) = saved

t_all = time.time()
for f in frames:
    path = os.path.join(args.out, f"f{f:04d}.exr")
    zpath = path.replace(".exr", "_z.exr")
    if args.skip_existing and os.path.exists(path) and (args.no_depth or os.path.exists(zpath)):
        continue
    t0 = time.time()
    scene.frame_set(f)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    if not args.no_depth:
        render_depth(f, zpath)
    print(f"FRAME {f} {time.time() - t0:.1f}s", flush=True)
print(f"DONE {len(frames)} frames in {time.time() - t_all:.1f}s", flush=True)
