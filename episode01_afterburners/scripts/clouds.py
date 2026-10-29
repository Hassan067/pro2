"""Volumetric cloud renderer (Numba, multithreaded CPU).

Cycles on 4 CPU cores needs minutes per 1080x1920 frame for full-screen
volumetrics, so the cloud layer is ray-marched here instead, from the exact
Blender camera transforms (exported per shutter sub-sample by export_frames.py),
and depth-composited with the Cycles aircraft layer in compose.py.

Model
  * density: domain-warped fBm cumulus field with an open flight corridor,
    wispy fringes and two cloud banks, baked once to a 3 m world-anchored grid
  * lighting: sun transmittance and sky visibility baked per voxel by marching
    the density grid; dual-lobe Henyey-Greenstein phase + multiple-scattering
    term; height-graded ambient; afterburner glow as an analytic point light
  * rendering: emission/absorption integration with jittered adaptive steps,
    empty-space skipping, shutter sub-samples for true camera motion blur
"""
import inspect
import math
import os
import zlib

import numba as nb
import numpy as np

# Shared with build_scene.py (world shader) so the sky matches the Cycles layer.
import timeline as TL  # noqa: E402

SUN_DIR = np.array(TL.SUN_DIR, np.float64)
SUN_RGB = np.array([1.0, 0.84, 0.70]) * 4.2
AMB_RGB = np.array([0.040, 0.054, 0.082]) * 0.9
GLOW_RGB = np.array([1.0, 0.45, 0.16])

GRID_MIN = np.array([-110.0, -60.0, -125.0])
GRID_MAX = np.array([110.0, 3460.0, 115.0])
VOXEL = 3.0
SIGMA = 0.11            # extinction per metre per unit density
HAZE = 0.0024           # storm haze / aerial perspective (per metre)
FAR = 1400.0
BLOCK = 4               # occupancy block size (voxels)

# --------------------------------------------------------------------------
# Gradient noise
# --------------------------------------------------------------------------
_rs = np.random.RandomState(1337)
PERM = np.concatenate([_rs.permutation(256)] * 2).astype(np.int32)
GRAD = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
                 [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
                 [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1]], np.float64)


@nb.njit(inline="always", fastmath=True)
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@nb.njit(inline="always", fastmath=True)
def _lerp(a, b, t):
    return a + (b - a) * t


@nb.njit(inline="always", fastmath=True)
def _gd(h, x, y, z):
    g = GRAD[h & 15]
    return g[0] * x + g[1] * y + g[2] * z


@nb.njit(fastmath=True)
def perlin(x, y, z, period):
    """Improved Perlin noise in ~[-1, 1]; lattice wraps every `period` cells (<=256)."""
    xf0 = math.floor(x)
    yf0 = math.floor(y)
    zf0 = math.floor(z)
    xi = int(xf0) % period
    yi = int(yf0) % period
    zi = int(zf0) % period
    xj = (xi + 1) % period
    yj = (yi + 1) % period
    zj = (zi + 1) % period
    x -= xf0
    y -= yf0
    z -= zf0
    u = _fade(x)
    v = _fade(y)
    w = _fade(z)
    p = PERM
    a0 = p[p[xi] + yi]
    a1 = p[p[xi] + yj]
    b0 = p[p[xj] + yi]
    b1 = p[p[xj] + yj]
    x1 = _lerp(_gd(p[a0 + zi], x, y, z), _gd(p[b0 + zi], x - 1, y, z), u)
    x2 = _lerp(_gd(p[a1 + zi], x, y - 1, z), _gd(p[b1 + zi], x - 1, y - 1, z), u)
    y1 = _lerp(x1, x2, v)
    x1 = _lerp(_gd(p[a0 + zj], x, y, z - 1), _gd(p[b0 + zj], x - 1, y, z - 1), u)
    x2 = _lerp(_gd(p[a1 + zj], x, y - 1, z - 1), _gd(p[b1 + zj], x - 1, y - 1, z - 1), u)
    y2 = _lerp(x1, x2, v)
    return _lerp(y1, y2, w)


@nb.njit(fastmath=True)
def fbm(x, y, z, octaves, rough, period=256):
    """Normalised fBm mapped to ~[0, 1] around 0.5 (like Blender's Noise 'Fac')."""
    amp = 1.0
    tot = 0.0
    norm = 0.0
    f = 1.0
    per = period
    for _ in range(octaves):
        tot += amp * perlin(x * f, y * f, z * f, per)
        norm += amp
        amp *= rough
        f *= 2.0
        per = min(per * 2, 256)
    return 0.5 + 0.75 * tot / norm


@nb.njit(inline="always", fastmath=True)
def smooth(x, a, b):
    t = (x - a) / (b - a)
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# Density field
# --------------------------------------------------------------------------
@nb.njit(fastmath=True)
def density_at(x, y, z, banks):
    wx = (fbm(x * 0.004 + 11.3, y * 0.004, z * 0.004, 2, 0.5) - 0.5) * 120.0
    wy = (fbm(x * 0.004, y * 0.004 + 37.1, z * 0.004, 2, 0.5) - 0.5) * 120.0
    wz = (fbm(x * 0.004, y * 0.004, z * 0.004 + 71.9, 2, 0.5) - 0.5) * 120.0
    s = 0.0105
    big = fbm((x + wx) * s, (y + wy) * 0.7 * s, (z + wz) * 1.4 * s, 5, 0.56)
    zb = 0.20 * smooth(z, -25.0, -90.0) + 0.10 * smooth(z, 40.0, 100.0)
    d = smooth(big + zb, 0.52, 0.68)
    # flight corridor: meanders along y, elliptical, walls warped by noise
    cx = 16.0 * (fbm(1.7, y * 0.0022, 8.3, 2, 0.5) - 0.5)
    cz = 12.0 * (fbm(9.1, y * 0.0022, 2.6, 2, 0.5) - 0.5)
    qx = x - cx + 14.0 * (fbm(x * 0.02, y * 0.012, z * 0.02 + 4.0, 2, 0.5) - 0.5)
    qz = (z - cz) * 1.25 + 14.0 * (fbm(x * 0.02 + 7.0, y * 0.012, z * 0.02, 2, 0.5) - 0.5)
    r = math.sqrt(qx * qx + qz * qz)
    # cloud canyon: clear between a deck and an overcast, walls well off to the sides
    hz = 14.0 + 18.0 * fbm(3.3, y * 0.004, 5.1, 1, 0.5)
    hx = 34.0 + 30.0 * fbm(6.1, y * 0.003, 1.9, 1, 0.5)
    cor = max(smooth(abs(qz), 0.4 * hz, hz), smooth(abs(qx), 0.45 * hx, hx))
    d *= cor
    wisp = smooth(fbm(x * 0.05, y * 0.0175, z * 0.05, 3, 0.6), 0.55, 0.80)
    near = smooth(r, 70.0, 30.0)
    d += wisp * near * (0.10 + 0.25 * cor)
    bn = smooth(fbm(x * 0.03 + 5.0, y * 0.03, z * 0.03, 3, 0.6), 0.40, 0.70)
    for i in range(banks.shape[0]):
        g = math.exp(-((y - banks[i, 0]) / banks[i, 1]) ** 2) * smooth(r, 45.0, 8.0)
        d += g * bn * 0.55
    ef = smooth(x, -110.0, -80.0) * smooth(x, 110.0, 80.0) * smooth(z, -125.0, -100.0) * smooth(z, 115.0, 90.0)
    return d * ef


@nb.njit(parallel=True, fastmath=True)
def _bake_density(nx, ny, nz, ox, oy, oz, vs, banks):
    D = np.zeros((nx, ny, nz), np.float32)
    for i in nb.prange(nx):
        x = ox + (i + 0.5) * vs
        for j in range(ny):
            y = oy + (j + 0.5) * vs
            for k in range(nz):
                D[i, j, k] = density_at(x, y, oz + (k + 0.5) * vs, banks)
    return D


@nb.njit(inline="always", fastmath=True)
def trilinear(G, ox, oy, oz, ivs, x, y, z):
    fx = (x - ox) * ivs - 0.5
    fy = (y - oy) * ivs - 0.5
    fz = (z - oz) * ivs - 0.5
    nx, ny, nz = G.shape
    if fx < 0.0 or fy < 0.0 or fz < 0.0 or fx >= nx - 1 or fy >= ny - 1 or fz >= nz - 1:
        return 0.0
    i = int(fx)
    j = int(fy)
    k = int(fz)
    u = fx - i
    v = fy - j
    w = fz - k
    c00 = G[i, j, k] * (1 - u) + G[i + 1, j, k] * u
    c10 = G[i, j + 1, k] * (1 - u) + G[i + 1, j + 1, k] * u
    c01 = G[i, j, k + 1] * (1 - u) + G[i + 1, j, k + 1] * u
    c11 = G[i, j + 1, k + 1] * (1 - u) + G[i + 1, j + 1, k + 1] * u
    return (c00 * (1 - v) + c10 * v) * (1 - w) + (c01 * (1 - v) + c11 * v) * w


@nb.njit(inline="always", fastmath=True)
def trilinear_wrap(G, x, y, z):
    n = G.shape[0]
    x = x % n
    y = y % n
    z = z % n
    i = int(x)
    j = int(y)
    k = int(z)
    u = x - i
    v = y - j
    w = z - k
    i1 = (i + 1) % n
    j1 = (j + 1) % n
    k1 = (k + 1) % n
    c00 = G[i, j, k] * (1 - u) + G[i1, j, k] * u
    c10 = G[i, j1, k] * (1 - u) + G[i1, j1, k] * u
    c01 = G[i, j, k1] * (1 - u) + G[i1, j, k1] * u
    c11 = G[i, j1, k1] * (1 - u) + G[i1, j1, k1] * u
    return (c00 * (1 - v) + c10 * v) * (1 - w) + (c01 * (1 - v) + c11 * v) * w


@nb.njit(parallel=True, fastmath=True)
def _bake_transmittance(D, ox, oy, oz, vs, dx, dy, dz, dists, sigma):
    nx, ny, nz = D.shape
    T = np.zeros_like(D)
    ivs = 1.0 / vs
    for i in nb.prange(nx):
        x = ox + (i + 0.5) * vs
        for j in range(ny):
            y = oy + (j + 0.5) * vs
            for k in range(nz):
                z = oz + (k + 0.5) * vs
                tau = 0.0
                prev = 0.0
                for s in dists:
                    tau += trilinear(D, ox, oy, oz, ivs, x + dx * s, y + dy * s, z + dz * s) * (s - prev)
                    prev = s
                T[i, j, k] = math.exp(-tau * sigma)
    return T


@nb.njit(parallel=True)
def _bake_detail(n, period):
    G = np.zeros((n, n, n), np.float32)
    sc = period / n
    for i in nb.prange(n):
        for j in range(n):
            for k in range(n):
                G[i, j, k] = fbm(i * sc, j * sc, k * sc, 3, 0.55, period)
    return G


def _block_max(D, b):
    nx, ny, nz = D.shape
    px, py, pz = (-nx) % b, (-ny) % b, (-nz) % b
    Dp = np.pad(D, ((0, px), (0, py), (0, pz)))
    M = Dp.reshape(Dp.shape[0] // b, b, Dp.shape[1] // b, b, Dp.shape[2] // b, b).max(axis=(1, 3, 5))
    # dilate one block so marching can't step over a boundary voxel
    from scipy.ndimage import maximum_filter
    return maximum_filter(M, size=3).astype(np.float32)


class CloudField:
    def __init__(self, banks, cache_dir):
        self.banks = np.asarray(banks, np.float64).reshape(-1, 2)
        self.cache = os.path.join(cache_dir, "cloud_grids.npz")
        self.shape = tuple(int(round((GRID_MAX[i] - GRID_MIN[i]) / VOXEL)) for i in range(3))

    def build(self, force=False):
        src = zlib.crc32(inspect.getsource(density_at.py_func).encode())     # re-bake on field edits
        key = np.concatenate([self.banks.ravel(), GRID_MIN, GRID_MAX, [VOXEL, SIGMA, src]])
        if not force and os.path.exists(self.cache):
            z = np.load(self.cache)
            if z["key"].shape == key.shape and np.allclose(z["key"], key):
                self.D, self.Ts, self.Tk, self.occ, self.detail = (z[k] for k in ("D", "Ts", "Tk", "occ", "detail"))
                return self
        nx, ny, nz = self.shape
        ox, oy, oz = GRID_MIN
        self.D = _bake_density(nx, ny, nz, ox, oy, oz, VOXEL, self.banks)
        sun_d = np.array([2, 5, 9, 15, 23, 34, 50, 72, 100, 140, 190], np.float64)
        sky_d = np.array([3, 8, 16, 28, 45, 70, 100], np.float64)
        self.Ts = _bake_transmittance(self.D, ox, oy, oz, VOXEL, *SUN_DIR, sun_d, SIGMA)
        self.Tk = _bake_transmittance(self.D, ox, oy, oz, VOXEL, 0.0, 0.0, 1.0, sky_d, SIGMA)
        self.occ = _block_max(self.D, BLOCK)
        self.detail = _bake_detail(64, 8)
        os.makedirs(os.path.dirname(self.cache), exist_ok=True)
        np.savez(self.cache, key=key, D=self.D, Ts=self.Ts, Tk=self.Tk, occ=self.occ, detail=self.detail)
        return self


# --------------------------------------------------------------------------
# Sky (mirrors the Blender world shader in build_scene.py)
# --------------------------------------------------------------------------
_K = TL.SKY
_HOR, _ZEN, _BEL = _K["horizon"], _K["zenith"], _K["below"]
_GLW, _G8, _G120 = _K["glow_rgb"], _K["glow8"], _K["glow120"]
_CLD, _CLIT = _K["cloud_rgb"], _K["cloud_lit"]


@nb.njit(fastmath=True)
def sky_color(dx, dy, dz, out):
    sx, sy, sz = SUN_DIR_NB
    up = smooth(dz, -0.05, 0.6)
    below = smooth(dz, 0.02, -0.25)
    sd = max(0.0, dx * sx + dy * sy + dz * sz)
    glow = _G8 * sd ** 8 + _G120 * sd ** 120
    cl = smooth(fbm(dx * 2.2 + 3.0, dy * 2.2, dz * 6.6, 6, 0.6), 0.42, 0.68)
    m = cl * smooth(dz, -0.3, -0.02) * (1.0 - smooth(dz, 0.12, 0.45))
    lit = _CLIT[0] + _CLIT[1] * sd ** 3
    for c in range(3):
        v = _HOR[c] + (_ZEN[c] - _HOR[c]) * up
        v = v + (_BEL[c] - v) * below
        v += _GLW[c] * glow
        out[c] = v + (_CLD[c] * lit - v) * m


SUN_DIR_NB = (float(SUN_DIR[0]), float(SUN_DIR[1]), float(SUN_DIR[2]))


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------
@nb.njit(inline="always")
def _hash(a, b, c):
    h = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0


@nb.njit(fastmath=True)
def _march(ox_, oy_, oz_, dx, dy, dz, t0, t1, seed,
           D, Ts, Tk, occ, detail, gmin, vs, phase, fp, out):
    """Integrate emission/absorption along [t0, t1]. Writes (r, g, b, T) to out."""
    gox, goy, goz = gmin[0], gmin[1], gmin[2]
    ivs = 1.0 / vs
    obs = 1.0 / (vs * BLOCK)
    onx, ony, onz = occ.shape
    sun_r, sun_g, sun_b = fp[0], fp[1], fp[2]
    amb_r, amb_g, amb_b = fp[3], fp[4], fp[5]
    ab = fp[6]
    nzx, nzy, nzz = fp[7], fp[8], fp[9]
    fwx, fwy, fwz = fp[10], fp[11], fp[12]
    trail = fp[13]
    glow_k = fp[14]
    cr = 0.0
    cg = 0.0
    cb = 0.0
    T = 1.0
    t = t0 + seed * min(0.5 + 0.02 * t0, 9.0)
    ms_k = 0.28 / (4.0 * math.pi)
    while t < t1 and T > 0.004:
        dt = min(0.5 + 0.02 * t, 9.0)
        x = ox_ + dx * t
        y = oy_ + dy * t
        z = oz_ + dz * t
        # exhaust trail (analytic, follows the jet)
        qx = x - nzx
        qy = y - nzy
        qz = z - nzz
        along = -(qx * fwx + qy * fwy + qz * fwz)
        sig_t = 0.0
        tr_r = 0.0
        tr_g = 0.0
        tr_b = 0.0
        if trail > 0.0 and along > 0.0 and along < 115.0:
            rx = qx + along * fwx
            ry = qy + along * fwy
            rz = qz + along * fwz
            rr = math.sqrt(rx * rx + ry * ry + rz * rz)
            R = 0.9 + 0.045 * along
            if rr < 2.5 * R:
                rad = math.exp(-2.0 * (rr / R) ** 2)
                tex = smooth(trilinear_wrap(detail, x * 0.35, y * 0.09, z * 0.35), 0.35, 0.75)
                alongf = smooth(along, 10.0, 28.0) * (1.0 - smooth(along, 40.0, 110.0))
                sig_t = rad * alongf * tex * tex * trail * 0.012
                hot = rad * smooth(along, 3.0, 9.0) * math.exp(-along / 14.0) * ab * 0.12
                tr_r = 0.16 * sig_t / 0.6 + hot
                tr_g = 0.17 * sig_t / 0.6 + hot * 0.42
                tr_b = 0.20 * sig_t / 0.6 + hot * 0.12
                dt = min(dt, max(0.25, 0.35 * R))
        # clouds
        bx = int((x - gox) * obs)
        by = int((y - goy) * obs)
        bz = int((z - goz) * obs)
        occupied = 0 <= bx < onx and 0 <= by < ony and 0 <= bz < onz and occ[bx, by, bz] > 0.0005
        lr = tr_r
        lg = tr_g
        lb = tr_b
        sig = sig_t
        if occupied:
            rho = trilinear(D, gox, goy, goz, ivs, x, y, z)
            if rho > 0.0005:
                det = trilinear_wrap(detail, x * 0.11 * 0.75, y * 0.066 * 0.75, z * 0.11 * 0.75)
                e = 0.55 + (min(max(det, 0.3), 0.7) - 0.3) * 2.25
                rho = max(rho * e - 0.04, 0.0)
                if rho > 0.0:
                    ts = trilinear(Ts, gox, goy, goz, ivs, x, y, z)
                    tk = trilinear(Tk, gox, goy, goz, ivs, x, y, z)
                    s_sun = ts * phase + ms_k * math.sqrt(math.sqrt(ts))
                    h = smooth(z, -110.0, 90.0)
                    amb = (0.25 + 0.75 * tk) * (0.25 + 0.75 * h)
                    gx = x - nzx
                    gy = y - nzy
                    gz = z - nzz
                    glow = ab * glow_k / (gx * gx + gy * gy + gz * gz + 25.0)
                    s = rho * SIGMA
                    lr += s * (sun_r * s_sun + amb_r * amb + 1.0 * glow)
                    lg += s * (sun_g * s_sun + amb_g * amb + 0.45 * glow)
                    lb += s * (sun_b * s_sun + amb_b * amb + 0.16 * glow)
                    sig += s
        elif sig_t == 0.0:
            dt = max(dt, 0.7 * vs * BLOCK)
        # thin haze for aerial perspective
        sig += HAZE
        lr += HAZE * amb_r * 0.6
        lg += HAZE * amb_g * 0.6
        lb += HAZE * amb_b * 0.6
        dt = min(dt, t1 - t + 1e-3)
        a = 1.0 - math.exp(-sig * dt)
        k = T * a / sig
        cr += k * lr
        cg += k * lg
        cb += k * lb
        T *= 1.0 - a
        t += dt
    out[0] = cr
    out[1] = cg
    out[2] = cb
    out[3] = T


@nb.njit(parallel=True, fastmath=True)
def render_kernel(H, W, cams, nsamp, split_depth, frame_seed, D, Ts, Tk, occ, detail, gmin, vs, fp):
    """cams: (S, 13) = origin(3) right(3) up(3) fwd(3) tan_half_v at S instants across the shutter;
    each of the nsamp samples per pixel picks a random instant between them.  split_depth: planar depth
    of the aircraft layer (front/back split) or <=0 for none.
    Returns front (H,W,4: rgb + T) and back (H,W,3: clouds beyond split + sky)."""
    S = cams.shape[0]
    front = np.zeros((H, W, 4), np.float32)
    back = np.zeros((H, W, 3), np.float32)
    sx, sy, sz = SUN_DIR_NB
    aspect = W / H
    for py in nb.prange(H):
        buf = np.zeros(4)
        skyc = np.zeros(3)
        c = np.zeros(13)
        for px in range(W):
            fr = 0.0
            fg = 0.0
            fb = 0.0
            fT = 0.0
            br = 0.0
            bg = 0.0
            bb = 0.0
            for s in range(nsamp):
                # stratified random instant within the shutter, camera interpolated
                tu = (s + _hash(px * 7 + 1, py * 13 + 5, frame_seed * 3 + s)) / nsamp * (S - 1)
                i0 = min(int(tu), S - 2)
                wt = tu - i0
                for q in range(13):
                    c[q] = cams[i0, q] * (1.0 - wt) + cams[i0 + 1, q] * wt
                jx = _hash(px, py, frame_seed * 31 + s)
                jy = _hash(py, px, frame_seed * 17 + s * 7 + 3)
                u = ((px + jx) / W * 2.0 - 1.0) * c[12] * aspect
                v = (1.0 - (py + jy) / H * 2.0) * c[12]
                dx = c[9] + c[3] * u + c[6] * v
                dy = c[10] + c[4] * u + c[7] * v
                dz = c[11] + c[5] * u + c[8] * v
                inv = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
                dx *= inv
                dy *= inv
                dz *= inv
                cosf = dx * c[9] + dy * c[10] + dz * c[11]
                cth = dx * sx + dy * sy + dz * sz
                g1 = 0.62
                g2 = -0.2
                ph = 0.7 * (1 - g1 * g1) / (4 * math.pi * (1 + g1 * g1 - 2 * g1 * cth) ** 1.5) \
                    + 0.3 * (1 - g2 * g2) / (4 * math.pi * (1 + g2 * g2 - 2 * g2 * cth) ** 1.5)
                seed = _hash(px * 3 + s, py * 5 + 1, frame_seed)
                tsplit = split_depth / cosf if split_depth > 0.0 else FAR
                _march(c[0], c[1], c[2], dx, dy, dz, 0.0, tsplit, seed,
                       D, Ts, Tk, occ, detail, gmin, vs, ph, fp, buf)
                fr += buf[0]
                fg += buf[1]
                fb += buf[2]
                fT += buf[3]
                sky_color(dx, dy, dz, skyc)
                if split_depth > 0.0:
                    _march(c[0], c[1], c[2], dx, dy, dz, tsplit, FAR, seed,
                           D, Ts, Tk, occ, detail, gmin, vs, ph, fp, buf)
                    br += buf[0] + buf[3] * skyc[0]
                    bg += buf[1] + buf[3] * skyc[1]
                    bb += buf[2] + buf[3] * skyc[2]
                else:
                    fr += buf[3] * skyc[0]
                    fg += buf[3] * skyc[1]
                    fb += buf[3] * skyc[2]
            inv_s = 1.0 / nsamp
            front[py, px, 0] = fr * inv_s
            front[py, px, 1] = fg * inv_s
            front[py, px, 2] = fb * inv_s
            front[py, px, 3] = fT * inv_s if split_depth > 0.0 else 0.0
            back[py, px, 0] = br * inv_s
            back[py, px, 1] = bg * inv_s
            back[py, px, 2] = bb * inv_s
    return front, back


def frame_params(ab, nozzle, fwd, trail, glow_k=30.0):
    fp = np.zeros(15)
    fp[0:3] = SUN_RGB
    fp[3:6] = AMB_RGB
    fp[6] = ab
    fp[7:10] = nozzle
    fp[10:13] = fwd
    fp[13] = trail
    fp[14] = glow_k
    return fp


def render(field, H, W, cams, nsamp, split_depth, frame_seed, fp):
    return render_kernel(H, W, np.ascontiguousarray(cams, np.float64), int(nsamp), float(split_depth),
                         int(frame_seed),
                         field.D, field.Ts, field.Tk, field.occ, field.detail,
                         GRID_MIN.astype(np.float64), VOXEL, fp)
