"""Afterburner plume renderer (Numba).

An emissive volume per engine, ray-marched in the aircraft's local frame and
depth-tested against the Cycles Z pass so the airframe occludes it correctly.

Structure of a reheat plume, from the nozzle exit downstream:
  * translucent blue/violet sheath and a blue-white core right at the exit
  * a train of Mach shock diamonds (bright yellow-white) decaying downstream
  * turbulent orange body that breaks up and cools to red at the tail
The flame front grows from the nozzle at light-off (ab_len 0 -> 1).
"""
import math

import numba as nb
import numpy as np

from clouds import perlin, smooth

PLUME_L = 12.0          # full plume length (m)
R_BOUND = 2.2           # marching bound radius (m)


@nb.njit(inline="always", fastmath=True)
def _fbm(x, y, z, octaves):
    a = 1.0
    s = 0.0
    n = 0.0
    f = 1.0
    for _ in range(octaves):
        s += a * perlin(x * f, y * f, z * f, 256)
        n += a
        a *= 0.55
        f *= 2.0
    return s / n


RAMP_POS = np.array([0.0, 0.04, 0.10, 0.24, 0.55, 1.0])
RAMP_COL = np.array([[0.60, 0.68, 1.00],     # blue-white core at the exit
                     [0.95, 0.80, 0.85],     # violet-white
                     [1.00, 0.62, 0.22],     # yellow
                     [1.00, 0.38, 0.07],     # orange
                     [0.85, 0.17, 0.03],     # deep orange
                     [0.40, 0.07, 0.02]])    # cooling red tail


@nb.njit(inline="always", fastmath=True)
def _ramp(u, out):
    """Body colour vs normalised distance downstream (piecewise-linear ramp)."""
    u = min(max(u, 0.0), 1.0)
    i = 0
    while i < RAMP_POS.shape[0] - 2 and u > RAMP_POS[i + 1]:
        i += 1
    k = (u - RAMP_POS[i]) / (RAMP_POS[i + 1] - RAMP_POS[i])
    for c in range(3):
        out[c] = RAMP_COL[i, c] + (RAMP_COL[i + 1, c] - RAMP_COL[i, c]) * k


@nb.njit(fastmath=True)
def _emission(x, y, z, t, ab, abl, gain, col, out):
    d = -y
    if d < 0.0:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        return
    flow = y + t * 38.0
    wamt = 0.03 + 0.035 * d
    wx = _fbm(x * 1.3 + 7.1, flow * 1.3, z * 1.3, 3) * wamt
    wz = _fbm(x * 1.3, flow * 1.3 + 3.7, z * 1.3 + 9.2, 3) * wamt
    r = math.sqrt((x + wx) ** 2 + (z + wz) ** 2)
    R = 0.40 + 0.06 * d
    rr = r / R
    core = math.exp(-2.6 * rr * rr)
    Lg = abl * PLUME_L
    env = smooth(d, 0.0, 0.3) * (1.0 - smooth(d, 0.35 * Lg, Lg + 1e-3)) * ab
    if env <= 1e-4:
        out[0] = 0.0
        out[1] = 0.0
        out[2] = 0.0
        return
    br = 0.5 + 0.5 * _fbm(x * 2.4, (y * 0.55 + t * 30.0) * 2.4, z * 2.4, 4)
    lo = 0.22 + 0.38 * min(d / PLUME_L, 1.0)
    br = smooth(br, lo, 0.75)
    body = core * env * (0.12 + 1.3 * br)
    ph = math.cos((d - 0.55) * 2.0 * math.pi / 1.05)
    dia = (0.5 + 0.5 * ph) ** 10 * math.exp(-d / 3.2) * math.exp(-(r / 0.2) ** 2) * env * smooth(d, 0.3, 0.6)
    sh = (rr - 0.75) / 0.28
    sheath = math.exp(-sh * sh) * math.exp(-d / 2.6) * env
    _ramp(d / PLUME_L, col)
    k1 = 1.9 * gain
    k2 = 8.0 * gain
    k3 = 1.1 * gain
    out[0] = col[0] * body * k1 + 1.00 * dia * k2 + 0.40 * sheath * k3
    out[1] = col[1] * body * k1 + 0.88 * dia * k2 + 0.52 * sheath * k3
    out[2] = col[2] * body * k1 + 0.66 * dia * k2 + 1.00 * sheath * k3


@nb.njit(parallel=True, fastmath=True)
def render_kernel(H, W, cam, jet_rot, jet_pos, nozzles_local, zbuf, t, ab, abl, gain, steps, seed):
    """cam: (13,) centre camera record. jet_rot: 3x3 world-from-local. zbuf: planar depth (H,W)."""
    out = np.zeros((H, W, 3), np.float32)
    aspect = W / H
    for py in nb.prange(H):
        col = np.zeros(3)
        e = np.zeros(3)
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
            cosf = dx * cam[9] + dy * cam[10] + dz * cam[11]
            tmax_surf = zbuf[py, px] / max(cosf, 1e-4)
            # ray into jet-local space (R^T * (o - p))
            ox = cam[0] - jet_pos[0]
            oy = cam[1] - jet_pos[1]
            oz = cam[2] - jet_pos[2]
            lox = jet_rot[0, 0] * ox + jet_rot[1, 0] * oy + jet_rot[2, 0] * oz
            loy = jet_rot[0, 1] * ox + jet_rot[1, 1] * oy + jet_rot[2, 1] * oz
            loz = jet_rot[0, 2] * ox + jet_rot[1, 2] * oy + jet_rot[2, 2] * oz
            ldx = jet_rot[0, 0] * dx + jet_rot[1, 0] * dy + jet_rot[2, 0] * dz
            ldy = jet_rot[0, 1] * dx + jet_rot[1, 1] * dy + jet_rot[2, 1] * dz
            ldz = jet_rot[0, 2] * dx + jet_rot[1, 2] * dy + jet_rot[2, 2] * dz
            acc_r = 0.0
            acc_g = 0.0
            acc_b = 0.0
            for n in range(nozzles_local.shape[0]):
                qx = lox - nozzles_local[n, 0]
                qy = loy - nozzles_local[n, 1]
                qz = loz - nozzles_local[n, 2]
                # infinite cylinder along local Y: (qx + t ldx)^2 + (qz + t ldz)^2 = R^2
                a = ldx * ldx + ldz * ldz
                b = 2.0 * (qx * ldx + qz * ldz)
                c = qx * qx + qz * qz - R_BOUND * R_BOUND
                if a < 1e-9:
                    if c > 0.0:
                        continue
                    t0 = 0.0
                    t1 = 1e9
                else:
                    disc = b * b - 4.0 * a * c
                    if disc <= 0.0:
                        continue
                    sq = math.sqrt(disc)
                    t0 = (-b - sq) / (2.0 * a)
                    t1 = (-b + sq) / (2.0 * a)
                # slab y in [-(L+1), 0.1]
                if abs(ldy) > 1e-9:
                    ya = (0.1 - qy) / ldy
                    yb = (-(PLUME_L + 1.0) - qy) / ldy
                    if ya > yb:
                        ya, yb = yb, ya
                    t0 = max(t0, ya)
                    t1 = min(t1, yb)
                elif qy > 0.1 or qy < -(PLUME_L + 1.0):
                    continue
                t0 = max(t0, 0.05)
                t1 = min(t1, tmax_surf)
                if t1 <= t0:
                    continue
                dt = (t1 - t0) / steps
                jit = ((px * 1973 + py * 9277 + seed * 26699 + n * 7) % 1024) / 1024.0
                tt = t0 + dt * jit
                for _ in range(steps):
                    _emission(qx + ldx * tt, qy + ldy * tt, qz + ldz * tt, t, ab, abl, gain, col, e)
                    acc_r += e[0] * dt
                    acc_g += e[1] * dt
                    acc_b += e[2] * dt
                    tt += dt
            out[py, px, 0] = acc_r
            out[py, px, 1] = acc_g
            out[py, px, 2] = acc_b
    return out


def render(H, W, cam, jet_matrix, nozzles_local, zbuf, t, ab, abl, gain=1.0, steps=40, seed=0):
    M = np.asarray(jet_matrix, np.float64)
    return render_kernel(H, W, np.asarray(cam, np.float64), np.ascontiguousarray(M[:3, :3]),
                         np.ascontiguousarray(M[:3, 3]), np.asarray(nozzles_local, np.float64),
                         np.ascontiguousarray(zbuf, np.float32), float(t), float(ab), float(abl),
                         float(gain), int(steps), int(seed))
