"""Shot timing and every animated quantity, as pure functions of the frame number.

World convention: the jet flies along +Y, Z is up, metres throughout.
"""
import math

FPS = 30
FRAME_START, FRAME_END = 1, 420          # 14.0 s
SHOT1 = (1, 150)                         # chase through the clouds
SHOT2 = (151, 285)                       # close on the nozzles, afterburner light-off
SHOT3 = (286, 420)                       # acceleration, camera pulls back
IGNITE = 192                             # afterburner light-off frame

V_CRUISE = 150.0                         # m/s before light-off

# Low sun ahead and to the right of the aircraft (unit vector toward the sun),
# shared by the Blender world/lights and the cloud renderer.
_s = (0.85, 0.75, 0.15)
SUN_DIR = tuple(c / math.sqrt(sum(v * v for v in _s)) for c in _s)

# Storm-sky palette (scene-linear), shared by the Blender world shader and clouds.sky_color
SKY = {
    "horizon": (0.026, 0.031, 0.041),
    "zenith": (0.004, 0.007, 0.014),
    "below": (0.009, 0.011, 0.016),
    "glow_rgb": (1.0, 0.66, 0.42),
    "glow8": 0.16,
    "glow120": 3.0,
    "cloud_rgb": (0.040, 0.044, 0.054),
    "cloud_lit": (0.25, 1.8),
}

NOZZLE_EXIT_Y = -8.95                    # jet-local nozzle exit plane
ENGINE_Z = -0.02


def t_of(f):
    return (f - 1) / FPS


def smoothstep(a, b, x):
    if b == a:
        return 1.0 if x >= b else 0.0
    u = min(1.0, max(0.0, (x - a) / (b - a)))
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return a + (b - a) * u


def lerp3(a, b, u):
    return tuple(lerp(x, y, u) for x, y in zip(a, b))


def wobble(t, seed, freqs=(0.37, 0.71, 1.33, 2.9), amps=(1.0, 0.6, 0.35, 0.15)):
    """Smooth pseudo-random signal in roughly [-1, 1]."""
    s = 0.0
    for i, (fr, am) in enumerate(zip(freqs, amps)):
        s += am * math.sin(2 * math.pi * fr * t + seed * (1.7 + i * 2.3) + i)
    return s / sum(amps)


# --------------------------------------------------------------------------
# Afterburner state
# --------------------------------------------------------------------------
def ab_level(f):
    """Afterburner intensity: 0 = military (dry) power, ~1 = full reheat."""
    t = f - IGNITE
    if t < 0:
        return 0.0
    rise = smoothstep(0, 5, t)
    pop = 0.45 * math.exp(-((t - 4.0) / 2.5) ** 2)          # light-off overshoot
    return rise + pop


def ab_length(f):
    """Flame front position as a fraction of the full plume length."""
    return smoothstep(IGNITE - 1, IGNITE + 9, f)


def nozzle_open(f):
    """Nozzle petals open as reheat is selected (0 closed, 1 fully open)."""
    return smoothstep(IGNITE - 8, IGNITE + 6, f)


def flicker(f):
    t = t_of(f)
    return 1.0 + 0.06 * wobble(t * 6.0, 3.1) + 0.03 * math.sin(f * 2.39)


def strobe(f):
    """Anti-collision strobe on the tail: two short flashes every 1.5 s."""
    ph = (f % 45)
    return 1.0 if ph in (0, 1, 5, 6) else 0.0


def shake_amount(f):
    base = 0.25
    kick = 1.6 * math.exp(-max(0.0, f - IGNITE) / 10.0) if f >= IGNITE else 0.0
    accel = 0.6 * smoothstep(SHOT3[0], SHOT3[0] + 40, f)
    return base + kick + accel


# --------------------------------------------------------------------------
# Jet motion
# --------------------------------------------------------------------------
def jet_speed(f):
    t = t_of(f)
    ti = t_of(IGNITE)
    d = max(0.0, t - ti - 0.15)
    return V_CRUISE + 13.0 * d ** 1.6


_JET_Y = {}


def jet_y(f):
    """Integrated position along +Y (sub-frame accurate via trapezoids)."""
    if not _JET_Y:
        y = 0.0
        _JET_Y[FRAME_START - 5] = 0.0
        for fr in range(FRAME_START - 4, FRAME_END + 6):
            y += 0.5 * (jet_speed(fr - 1) + jet_speed(fr)) / FPS
            _JET_Y[fr] = y
    fi = math.floor(f)
    u = f - fi
    return lerp(_JET_Y[fi], _JET_Y[fi + 1], u)


def jet_location(f):
    t = t_of(f)
    x = 1.2 * wobble(t * 0.5, 1.0)
    z = 0.8 * wobble(t * 0.6, 2.0)
    return (x, jet_y(f), z)


def jet_rotation(f):
    """(pitch, roll, yaw) in radians. Gentle flight-path corrections."""
    t = t_of(f)
    roll = math.radians(4.0 * wobble(t * 0.45, 4.0))
    pitch = math.radians(1.2 * wobble(t * 0.4, 5.0))
    yaw = math.radians(-0.8 * wobble(t * 0.5, 1.0))
    if f >= SHOT3[0]:
        # slight climb and bank as it accelerates away
        u = smoothstep(SHOT3[0], SHOT3[1], f)
        roll += math.radians(-10.0 * u)
        pitch += math.radians(3.0 * u)
    return (pitch, roll, yaw)


# --------------------------------------------------------------------------
# Cameras: each returns (location, look-at target, lens mm, f-stop or None)
# --------------------------------------------------------------------------
def _shake(f, scale):
    t = t_of(f)
    a = shake_amount(f) * scale
    return (a * wobble(t * 3.0, 11.0), a * 0.5 * wobble(t * 2.6, 12.0), a * wobble(t * 3.4, 13.0))


def cam1(f):
    u = smoothstep(SHOT1[0], SHOT1[1], f)
    jx, jy, jz = jet_location(f)
    off = lerp3((-4.0, -36.0, -7.5), (2.5, -27.0, -4.8), u)
    s = _shake(f, 0.12)
    loc = (off[0] + s[0], jy + off[1] + s[1], off[2] + s[2])
    tgt = (jx * 0.6, jy - 1.0, jz * 0.6 + 0.6)
    return loc, tgt, lerp(42.0, 46.0, u), None




def cam2(f):
    u = smoothstep(SHOT2[0], SHOT2[1], f)
    jx, jy, jz = jet_location(f)
    off = lerp3((6.2, -19.5, -3.6), (4.2, -14.8, -2.3), u)
    s = _shake(f, 0.06)
    loc = (jx + off[0] + s[0], jy + off[1] + s[1], jz + off[2] + s[2])
    # aim drifts from the nozzles toward the plume as it lights
    k = smoothstep(IGNITE - 4, IGNITE + 25, f)
    tgt = (jx + lerp(0.2, 0.3, k), jy + lerp(-7.5, -10.5, k), jz + lerp(0.1, 0.0, k))
    return loc, tgt, lerp(38.0, 34.0, u), 4.0


def _sep3(tau):
    return 26.0 + 8.0 * tau + 6.0 * tau ** 2


def cam3(f):
    tau = t_of(f) - t_of(SHOT3[0])
    u = smoothstep(SHOT3[0], SHOT3[1], f)
    jx, jy, jz = jet_location(f)
    sep = _sep3(tau)
    x = lerp(4.5, -6.0, u)
    z = lerp(-4.0, 5.5, smoothstep(SHOT3[0], SHOT3[1] + 30, f))
    s = _shake(f, 0.15)
    loc = (x + s[0], jy - sep + s[1], z + s[2])
    tgt = (jx * 0.5, jy - 6.0, jz + 1.0)
    lens = lerp(34.0, 66.0, smoothstep(SHOT3[0] + 10, SHOT3[1], f))
    return loc, tgt, lens, None


def active_camera(f):
    if f <= SHOT1[1]:
        return "CAM_1"
    if f <= SHOT2[1]:
        return "CAM_2"
    return "CAM_3"


def camera_state(f):
    return {"CAM_1": cam1, "CAM_2": cam2, "CAM_3": cam3}[active_camera(f)](f)


def cloud_banks():
    """(world y, half-width) of the cloud banks the jet punches through in shot 3."""
    return [(jet_y(338), 22.0), (jet_y(392), 30.0)]
