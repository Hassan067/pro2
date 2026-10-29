"""Cloud layer + depth compositing + finishing for each frame.

  final = clouds_front + T_front * (aircraft_rgb + (1 - aircraft_alpha) * clouds_back)

where the split between front/back cloud integration is the aircraft's planar
depth, so clouds in front of the jet occlude it and the ones behind it don't.
Then heat haze (inside the projected exhaust plume only), HDR bloom, lens
dispersion and vignette in scene-linear, and finally Blender's AgX view
transform (identical colour management to a straight Blender render).

  python compose.py --data work/frame_data.json --cycles DIR --out DIR
                    [--range 1 420 | --frames 1,2] [--cloud-scale 0.5]
"""
import argparse
import json
import math
import os
import sys
import time

import bpy  # noqa: I001
import numba as nb
import numpy as np
import OpenEXR
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clouds as CL  # noqa: E402
import plume as PL  # noqa: E402
import timeline as TL  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOZZLES_LOCAL = np.array([[sx * 0.68, TL.NOZZLE_EXIT_Y, TL.ENGINE_Z] for sx in (1, -1)])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_exr(path):
    with OpenEXR.File(path, separate_channels=True) as f:
        ch = f.channels()
        rgb = np.dstack([np.asarray(ch[c].pixels, np.float32) for c in "RGB"])
        return rgb, np.asarray(ch["A"].pixels, np.float32)


def read_cycles_layer(path):
    """Premultiplied linear RGBA of the aircraft + planar depth (inf where uncovered)."""
    rgb, alpha = read_exr(path)
    zrgb, za = read_exr(path.replace(".exr", "_z.exr"))
    z = np.where(za > 0.02, zrgb[..., 0] / np.maximum(za, 1e-4), np.float32(1e9)).astype(np.float32)
    # depth pass is half resolution: nearest-neighbour up (it only gates the plume)
    fy, fx = alpha.shape[0] // z.shape[0], alpha.shape[1] // z.shape[1]
    z = np.repeat(np.repeat(z, fy, axis=0), fx, axis=1)
    return rgb, alpha, z


def min_pool(z, f):
    H, W = z.shape
    return z[:H // f * f, :W // f * f].reshape(H // f, f, W // f, f).min(axis=(1, 3))


@nb.njit(parallel=True, fastmath=True)
def upsample(src, H, W):
    h, w, c = src.shape
    out = np.empty((H, W, c), np.float32)
    sy = h / H
    sx = w / W
    for y in nb.prange(H):
        fy = min(max((y + 0.5) * sy - 0.5, 0.0), h - 1.001)
        j = int(fy)
        v = fy - j
        for x in range(W):
            fx = min(max((x + 0.5) * sx - 0.5, 0.0), w - 1.001)
            i = int(fx)
            u = fx - i
            for k in range(c):
                out[y, x, k] = (src[j, i, k] * (1 - u) + src[j, i + 1, k] * u) * (1 - v) + \
                               (src[j + 1, i, k] * (1 - u) + src[j + 1, i + 1, k] * u) * v
    return out


@nb.njit(parallel=True, fastmath=True)
def haze_field(H, W, cam, nozzles, fwd, heat, t, amp):
    """Screen-space displacement (dx, dy) in pixels from hot exhaust: strength is the
    projected thickness of the plume around each engine axis."""
    out = np.zeros((H, W, 2), np.float32)
    aspect = W / H
    L = 22.0
    for py in nb.prange(H):
        for px in range(W):
            u = ((px + 0.5) / W * 2.0 - 1.0) * cam[12] * aspect
            v = (1.0 - (py + 0.5) / H * 2.0) * cam[12]
            dx = cam[9] + cam[3] * u + cam[6] * v
            dy = cam[10] + cam[4] * u + cam[7] * v
            dz = cam[11] + cam[5] * u + cam[8] * v
            inv = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            dx *= inv
            dy *= inv
            dz *= inv
            m = 0.0
            for e in range(nozzles.shape[0]):
                # closest approach between the view ray and the plume axis segment
                ax, ay, az = -fwd[0], -fwd[1], -fwd[2]
                wx = cam[0] - nozzles[e, 0]
                wy = cam[1] - nozzles[e, 1]
                wz = cam[2] - nozzles[e, 2]
                b = dx * ax + dy * ay + dz * az
                d_ = dx * wx + dy * wy + dz * wz
                e_ = ax * wx + ay * wy + az * wz
                den = 1.0 - b * b
                if den < 1e-6:
                    continue
                sr = (b * e_ - d_) / den
                sa = (e_ - b * d_) / den
                sa = min(max(sa, 0.0), L)
                sr = max(b * sa - d_, 0.0)
                cx = cam[0] + dx * sr - (nozzles[e, 0] + ax * sa)
                cy = cam[1] + dy * sr - (nozzles[e, 1] + ay * sa)
                cz = cam[2] + dz * sr - (nozzles[e, 2] + az * sa)
                r = math.sqrt(cx * cx + cy * cy + cz * cz)
                R = 0.45 + 0.085 * sa
                f = CL.smooth(sa, 0.0, 0.8) * (1.0 - CL.smooth(sa, 9.0, L))
                m = max(m, math.exp(-(r / R) ** 2) * f)
            if m > 0.003:
                s = 1.0 / 26.0
                n1 = CL.perlin(px * s, py * s, t * 7.0, 256)
                n2 = CL.perlin(px * s + 31.7, py * s - 12.1, t * 7.0 + 5.3, 256)
                out[py, px, 0] = m * heat * amp * n1
                out[py, px, 1] = m * heat * amp * n2
    return out


@nb.njit(parallel=True, fastmath=True)
def remap(img, disp, ca):
    """Bilinear remap by a displacement field, plus radial chromatic aberration."""
    H, W, _ = img.shape
    out = np.empty_like(img)
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5
    nrm = 1.0 / (cx * cx + cy * cy)
    for y in nb.prange(H):
        for x in range(W):
            rx = x - cx
            ry = y - cy
            r2 = (rx * rx + ry * ry) * nrm
            for k in range(3):
                sc = 1.0 + ca * (k - 1) * r2
                fx = cx + rx * sc + disp[y, x, 0]
                fy = cy + ry * sc + disp[y, x, 1]
                fx = min(max(fx, 0.0), W - 1.001)
                fy = min(max(fy, 0.0), H - 1.001)
                i = int(fx)
                j = int(fy)
                u = fx - i
                v = fy - j
                out[y, x, k] = (img[j, i, k] * (1 - u) + img[j, i + 1, k] * u) * (1 - v) + \
                               (img[j + 1, i, k] * (1 - u) + img[j + 1, i + 1, k] * u) * v
    return out


def bloom(img, threshold=1.0, strength=0.12):
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    knee = 0.5
    soft = np.clip(lum - threshold + knee, 0, 2 * knee) ** 2 / (4 * knee + 1e-6)
    contrib = np.maximum(soft, lum - threshold) / np.maximum(lum, 1e-4)
    bright = img * contrib[..., None]
    H, W, _ = img.shape
    f = 4
    h, w = H // f, W // f
    small = bright[:h * f, :w * f].reshape(h, f, w, f, 3).mean(axis=(1, 3))
    acc = np.zeros_like(small)
    for sigma, wgt in ((2.0, 0.45), (6.0, 0.3), (16.0, 0.17), (40.0, 0.08)):
        acc += wgt * gaussian_filter(small, sigma=(sigma, sigma, 0), mode="nearest")
    up = upsample(acc.astype(np.float32), H, W)
    return img + strength * up


def vignette(H, W, amount=0.32):
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (x - W / 2) / (W / 2)
    ny = (y - H / 2) / (H / 2)
    r = np.sqrt(nx * nx * 1.0 + ny * ny * 0.55)
    t = np.clip((r - 0.45) / 0.85, 0, 1)
    return (1.0 - amount * t * t * (3 - 2 * t)).astype(np.float32)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cycles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--range", nargs=2, type=int)
    ap.add_argument("--cloud-scale", type=float, default=0.5)
    ap.add_argument("--samples", type=int, default=4, help="cloud samples per pixel")
    ap.add_argument("--plume-scale", type=int, default=1, help="plume downsample factor")
    ap.add_argument("--plume-steps", type=int, default=40)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--wait", action="store_true", help="wait for Cycles frames to appear")
    args = ap.parse_args(sys.argv[1:])

    data = json.load(open(args.data))
    frames = data["frames"]
    if args.frames:
        todo = [int(x) for x in args.frames.split(",")]
    else:
        a, b = args.range if args.range else (1, len(frames))
        todo = list(range(a, b + 1))

    field = CL.CloudField(TL.cloud_banks(), os.path.join(ROOT, "work")).build()

    scene = bpy.context.scene
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    os.makedirs(args.out, exist_ok=True)

    vig = None
    img_bpy = None
    for f in todo:
        out_path = os.path.join(args.out, f"f{f:04d}.png")
        if args.skip_existing and os.path.exists(out_path):
            continue
        exr = os.path.join(args.cycles, f"f{f:04d}.exr")
        zexr = exr.replace(".exr", "_z.exr")          # written last by render.py
        if args.wait and not os.path.exists(zexr):
            while not os.path.exists(zexr):
                time.sleep(3)
            time.sleep(1.0)
        t0 = time.time()
        rgb, alpha, zdepth = read_cycles_layer(exr)
        H, W = alpha.shape
        fd = frames[str(f)]
        cams = np.array(fd["cams"], np.float64)
        fp = CL.frame_params(fd["ab"] * fd["flick"], fd["nozzle"], fd["jet_fwd"], fd["trail"])
        h, w = int(round(H * args.cloud_scale)), int(round(W * args.cloud_scale))
        split = max(fd["nozzle_depth"] - 2.0, 1.0)
        front, back = CL.render(field, h, w, cams, args.samples, split, f, fp)
        t1 = time.time()
        front = upsample(front, H, W)
        back = upsample(back, H, W)

        # afterburner plume, depth-tested against the airframe
        pf = args.plume_scale
        flame = np.zeros_like(rgb)
        if fd["ab"] > 1e-3:
            zb = min_pool(zdepth, pf) if pf > 1 else zdepth
            flame = PL.render(H // pf, W // pf, fd["cam_centre"], fd["jet_matrix"], NOZZLES_LOCAL, zb,
                              fd["time"], fd["ab"] * fd["flick"], fd["ab_len"], steps=args.plume_steps, seed=f)
            if pf > 1:
                flame = upsample(flame, H, W)
        t2 = time.time()

        comp = front[..., :3] + front[..., 3:4] * (rgb + flame + (1.0 - alpha[..., None]) * back)

        # heat haze: idle shimmer at military power, strong in reheat
        heat = 0.3 + 0.9 * min(fd["ab"], 1.2)
        cam = np.array(fd["cam_centre"], np.float64)
        disp = haze_field(H, W, cam, np.array(fd["nozzles"], np.float64), np.array(fd["jet_fwd"]),
                          heat, fd["time"], 3.2 * H / 1920.0)
        comp = remap(comp.astype(np.float32), disp, 0.0012)

        comp = bloom(comp, threshold=1.0, strength=0.14)
        if vig is None or vig.shape != (H, W):
            vig = vignette(H, W)
        comp *= vig[..., None]

        # AgX view transform via Blender colour management
        if img_bpy is None or tuple(img_bpy.size) != (W, H):
            img_bpy = bpy.data.images.new("comp", W, H, float_buffer=True, alpha=False)
        px = np.ones((H, W, 4), np.float32)
        px[..., :3] = np.maximum(comp, 0.0)
        img_bpy.pixels.foreach_set(px[::-1].ravel())
        img_bpy.save_render(out_path, scene=scene)
        print(f"FRAME {f} clouds {t1 - t0:.1f}s plume {t2 - t1:.1f}s total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
