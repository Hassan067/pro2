"""Episode 01 - Why Fighter Jets Need Afterburners.

Builds the complete Blender scene procedurally (aircraft, materials, afterburner
FX, volumetric clouds, lighting, cameras, compositor) and saves it as a .blend.

Run:  python build_scene.py [--out path.blend]
"""
import math
import os
import sys

import bpy  # noqa: I001  (must precede addon_utils)
import addon_utils
import bmesh
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as TL  # noqa: E402
from nodes import NB, new_material  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

addon_utils.enable("cycles", default_set=True)

# ==========================================================================
# Scene reset
# ==========================================================================
bpy.ops.wm.read_factory_settings(use_empty=True)
addon_utils.enable("cycles", default_set=True)
scene = bpy.context.scene
scene.name = "EP01_Afterburners"
bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"


def link(obj, coll=None):
    (coll or scene.collection).objects.link(obj)
    return obj


def new_coll(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c


C_JET = new_coll("Jet")
C_FX = new_coll("FX")
C_ENV = new_coll("Environment")
C_CAM = new_coll("Cameras")

JET = link(bpy.data.objects.new("JET", None), C_JET)
JET.empty_display_size = 3


def mesh_obj(name, bm, mat=None, coll=C_JET, parent=JET, smooth=True, subsurf=0):
    me = bpy.data.meshes.new(name)
    for f in bm.faces:
        f.smooth = smooth
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    link(ob, coll)
    if mat:
        me.materials.append(mat)
    if parent:
        ob.parent = parent
    if subsurf:
        md = ob.modifiers.new("Subdiv", "SUBSURF")
        md.levels = 1
        md.render_levels = subsurf
    return ob


# ==========================================================================
# Geometry helpers
# ==========================================================================
def loft(bm, rings, cap_start=True, cap_end=True, cyclic=False):
    """Skin a list of equal-length closed rings (lists of Vectors) with quads."""
    vrings = [[bm.verts.new(p) for p in r] for r in rings]
    n = len(rings[0])
    pairs = list(zip(vrings[:-1], vrings[1:]))
    if cyclic:
        pairs.append((vrings[-1], vrings[0]))
    for a, b in pairs:
        for j in range(n):
            k = (j + 1) % n
            bm.faces.new((a[j], a[k], b[k], b[j]))
    if cap_start and not cyclic:
        bm.faces.new(vrings[0])
    if cap_end and not cyclic:
        bm.faces.new(list(reversed(vrings[-1])))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return vrings


def super_ring(y, w, ht, hb, zc, n, N=48, xc=0.0):
    pts = []
    e = 2.0 / n
    for i in range(N):
        th = 2 * math.pi * i / N
        c, s = math.cos(th), math.sin(th)
        x = xc + w * math.copysign(abs(c) ** e, c)
        h = ht if s > 0 else hb
        z = zc + h * math.copysign(abs(s) ** e, s)
        pts.append(Vector((x, y, z)))
    return pts


def circle_ring(y, r, xc, zc, N=48, rfun=None, yfun=None):
    pts = []
    for i in range(N):
        th = 2 * math.pi * i / N
        rr = rfun(th) if rfun else r
        yy = yfun(th) if yfun else y
        pts.append(Vector((xc + rr * math.cos(th), yy, zc + rr * math.sin(th))))
    return pts


def naca_profile(t, M=22):
    """Closed NACA 00xx loop as (u along chord 0..1, thickness/chord)."""
    us = [0.5 * (1 - math.cos(math.pi * i / M)) for i in range(M + 1)]

    def yt(u):
        return 5 * t * (0.2969 * math.sqrt(u) - 0.1260 * u - 0.3516 * u * u
                        + 0.2843 * u ** 3 - 0.1036 * u ** 4)

    top = [(u, yt(u)) for u in reversed(us)]
    bot = [(u, -yt(u)) for u in us[1:-1]]
    return top + bot


def lifting_surface(sections, M=22):
    """sections: list of (le_point, chord, thickness, normal_dir). Chord runs to -Y."""
    rings = []
    for le, chord, t, nrm in sections:
        ring = []
        for u, h in naca_profile(t, M):
            ring.append(Vector(le) + Vector((0, -u * chord, 0)) + Vector(nrm) * (h * chord))
        rings.append(ring)
    bm = bmesh.new()
    loft(bm, rings)
    return bm


def mirror_x(bm):
    bmesh.ops.scale(bm, vec=(-1, 1, 1), verts=bm.verts[:])
    bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    return bm


def add_box(bm, center, size, rot=None):
    m = Matrix.Translation(center)
    if rot is not None:
        m = m @ rot
    m = m @ Matrix.Diagonal((*size, 1.0))
    bmesh.ops.create_cube(bm, size=1.0, matrix=m)


def add_cyl(bm, p0, p1, r0, r1, seg=24, caps=True):
    """Cone/cylinder between two points along the Y axis direction p0->p1."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    m = Matrix.Translation((p0 + p1) / 2) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=caps, cap_tris=False, segments=seg,
                          radius1=r0, radius2=r1, depth=d.length, matrix=m)


def add_torus(bm, center, R, r, seg=48, rseg=10):
    rings = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        ring = []
        for j in range(rseg):
            b = 2 * math.pi * j / rseg
            rad = R + r * math.cos(b)
            ring.append(Vector((center[0] + rad * math.cos(a), center[1] + r * math.sin(b),
                                center[2] + rad * math.sin(a))))
        rings.append(ring)
    vr = [[bm.verts.new(p) for p in ring] for ring in rings]
    for i in range(seg):
        a, b = vr[i], vr[(i + 1) % seg]
        for j in range(rseg):
            k = (j + 1) % rseg
            bm.faces.new((a[j], a[k], b[k], b[j]))


# ==========================================================================
# Materials
# ==========================================================================
def mat_paint():
    mat, n = new_material("Paint_AirSuperiorityGrey")
    p = n.coords("Object")
    x, y, z = n.sep(p)
    # two-tone counter-shading camouflage patches
    camo = n.smooth(n.noise(p, scale=0.11, detail=2.0, rough=0.5), 0.47, 0.53)
    base = n.mix(camo, (0.165, 0.185, 0.205, 1), (0.105, 0.118, 0.135, 1))
    # exhaust/oil grime streaked along the airflow
    gp = n.v("MULTIPLY", p, (6.0, 0.35, 6.0))
    grime = n.smooth(n.noise(gp, scale=1.6, detail=3.0, rough=0.6), 0.45, 0.75)
    grime_rear = n.smooth(y, -3.0, -8.6)
    gfac = n.clamp01(n.add(n.mul(grime, 0.55), n.mul(grime_rear, 0.35)))
    base = n.mix(gfac, base, (0.05, 0.05, 0.05, 1), "MULTIPLY")
    base = n.mix(n.mul(gfac, 0.6), base, base)
    # panel lines: staggered grid in jet space
    fy = n.m("FRACT", n.mul(y, 0.82))
    ly = n.m("GREATER_THAN", n.m("ABSOLUTE", n.sub(fy, 0.5)), 0.488)
    cell = n.m("FLOOR", n.mul(y, 0.82))
    fx = n.m("FRACT", n.add(n.mul(x, 1.05), n.mul(cell, 0.37)))
    lx = n.m("GREATER_THAN", n.m("ABSOLUTE", n.sub(fx, 0.5)), 0.486)
    fz = n.m("FRACT", n.add(n.mul(z, 1.3), n.mul(cell, 0.21)))
    lz = n.m("GREATER_THAN", n.m("ABSOLUTE", n.sub(fz, 0.5)), 0.49)
    lines = n.m("MAXIMUM", n.m("MAXIMUM", ly, lx), n.mul(lz, 0.6))
    base = n.mix(n.mul(lines, 0.55), base, (0.02, 0.02, 0.02, 1))
    # rivet-ish micro variation
    rough = n.add(0.38, n.mul(n.noise(p, scale=9.0, detail=3.0), 0.18))
    rough = n.add(rough, n.mul(gfac, 0.15))
    bump = n.new("ShaderNodeBump", invert=True)
    n.set(bump.inputs["Strength"], 0.35)
    n.set(bump.inputs["Distance"], 0.004)
    n.set(bump.inputs["Height"], n.add(lines, n.mul(n.noise(p, scale=40.0, detail=1.0), 0.15)))
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    n.set(bsdf.inputs["Base Color"], base)
    n.set(bsdf.inputs["Metallic"], 0.42)
    n.set(bsdf.inputs["Roughness"], rough)
    n.set(bsdf.inputs["Coat Weight"], 0.18)
    n.set(bsdf.inputs["Coat Roughness"], 0.25)
    n.set(bsdf.inputs["Normal"], bump.outputs["Normal"])
    n.output(bsdf.outputs[0])
    return mat


def mat_nozzle_metal():
    mat, n = new_material("Nozzle_BurntTitanium")
    p = n.coords("Object")
    x, y, z = n.sep(p)
    # heat tint gradient toward the exit (petals at y ~ -8 .. -9)
    h = n.lin(y, -7.6, -8.95)
    tint = n.ramp(h, [(0.0, (0.42, 0.40, 0.37)), (0.35, (0.55, 0.43, 0.26)),
                      (0.62, (0.30, 0.26, 0.42)), (0.85, (0.26, 0.33, 0.45)),
                      (1.0, (0.20, 0.19, 0.20))])
    soot = n.smooth(n.noise(n.v("MULTIPLY", p, (8, 1.0, 8)), scale=3.0, detail=6), 0.5, 0.8)
    col = n.mix(n.mul(soot, 0.6), tint, (0.03, 0.03, 0.03, 1))
    # petal seams: angular stripes around each engine axis
    ex = n.sub(n.m("ABSOLUTE", x), 0.68)
    ang = n.m("ARCTAN2", n.add(z, 0.02), ex)
    seam = n.m("FRACT", n.mul(ang, 16 / (2 * math.pi)))
    seam = n.m("GREATER_THAN", n.m("ABSOLUTE", n.sub(seam, 0.5)), 0.47)
    col = n.mix(n.mul(seam, 0.7), col, (0.01, 0.01, 0.01, 1))
    bump = n.new("ShaderNodeBump", invert=True)
    n.set(bump.inputs["Strength"], 0.5)
    n.set(bump.inputs["Distance"], 0.006)
    n.set(bump.inputs["Height"], seam)
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    n.set(bsdf.inputs["Base Color"], col)
    n.set(bsdf.inputs["Metallic"], 1.0)
    n.set(bsdf.inputs["Roughness"], n.add(0.28, n.mul(soot, 0.35)))
    n.set(bsdf.inputs["Anisotropic"], 0.3)
    n.set(bsdf.inputs["Normal"], bump.outputs["Normal"])
    n.output(bsdf.outputs[0])
    return mat


def mat_engine_hot():
    """Inside of the nozzle / augmentor: sooty metal glowing with exhaust heat."""
    mat, n = new_material("Engine_HotSection")
    p = n.coords("Object")
    x, y, z = n.sep(p)
    ab = n.attr("ab")
    depth = n.lin(y, -8.9, -6.8)          # 0 at exit, 1 deep inside
    glow_idle = n.mul(n.add(0.01, n.mul(depth, 0.12)), 1.0)
    glow_ab = n.mul(n.add(1.0, n.mul(depth, 1.2)), n.mul(ab, 2.0))
    strength = n.mul(n.add(glow_idle, glow_ab), n.attr("flick"))
    col_idle = (1.0, 0.18, 0.03, 1)
    col_ab = (1.0, 0.48, 0.16, 1)
    col = n.mix(n.clamp01(ab), col_idle, col_ab)
    streak = n.noise(n.v("MULTIPLY", p, (10, 1, 10)), scale=4.0, detail=5)
    strength = n.mul(strength, n.add(0.55, streak))
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    n.set(bsdf.inputs["Base Color"], (0.04, 0.035, 0.03, 1))
    n.set(bsdf.inputs["Metallic"], 0.8)
    n.set(bsdf.inputs["Roughness"], 0.55)
    n.set(bsdf.inputs["Emission Color"], col)
    n.set(bsdf.inputs["Emission Strength"], strength)
    n.output(bsdf.outputs[0])
    return mat


def mat_canopy():
    mat, n = new_material("Canopy_GoldTint")
    lw = n.new("ShaderNodeLayerWeight")
    n.set(lw.inputs["Blend"], 0.35)
    col = n.mix(lw.outputs["Facing"], (0.03, 0.028, 0.02, 1), (0.35, 0.28, 0.12, 1))
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    n.set(bsdf.inputs["Base Color"], col)
    n.set(bsdf.inputs["Metallic"], 0.55)
    n.set(bsdf.inputs["Roughness"], 0.04)
    n.set(bsdf.inputs["Coat Weight"], 1.0)
    n.set(bsdf.inputs["Coat Roughness"], 0.02)
    n.output(bsdf.outputs[0])
    return mat


def mat_simple(name, col, metal=0.0, rough=0.5, emit=None, emit_strength=0.0, emit_attr=None):
    mat, n = new_material(name)
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    n.set(bsdf.inputs["Base Color"], (*col, 1))
    n.set(bsdf.inputs["Metallic"], metal)
    n.set(bsdf.inputs["Roughness"], rough)
    if emit:
        n.set(bsdf.inputs["Emission Color"], (*emit, 1))
        s = emit_strength if emit_attr is None else n.mul(n.attr(emit_attr), emit_strength)
        n.set(bsdf.inputs["Emission Strength"], s)
    n.output(bsdf.outputs[0])
    return mat


M_PAINT = mat_paint()
M_NOZZLE = mat_nozzle_metal()
M_HOT = mat_engine_hot()
M_CANOPY = mat_canopy()
M_DARK = mat_simple("Intake_Dark", (0.012, 0.012, 0.014), 0.2, 0.7)
M_MISSILE = mat_simple("Missile_White", (0.62, 0.62, 0.6), 0.1, 0.38)
M_RADOME = mat_simple("Missile_Seeker", (0.08, 0.08, 0.085), 0.0, 0.25)
M_NAV_R = mat_simple("NavLight_Red", (0.3, 0.0, 0.0), 0, 0.2, (1.0, 0.03, 0.02), 60.0)
M_NAV_G = mat_simple("NavLight_Green", (0.0, 0.3, 0.0), 0, 0.2, (0.05, 1.0, 0.25), 60.0)
M_STROBE = mat_simple("Strobe_White", (0.5, 0.5, 0.5), 0, 0.2, (1.0, 0.95, 0.9), 120.0, "strobe")

# ==========================================================================
# Aircraft: twin-engine air-superiority fighter (generic, F-15/F-22 class)
# Nose at +Y.  Length ~18.6 m, span ~13.4 m.
# ==========================================================================
ENG_X, ENG_Z = 0.68, -0.02

# --- fuselage -------------------------------------------------------------
stations = [
    # y,     w,    ht,   hb,   zc,   n
    (9.60, 0.015, 0.015, 0.015, 0.14, 2.0),
    (9.25, 0.16, 0.16, 0.15, 0.13, 2.0),
    (8.60, 0.33, 0.33, 0.30, 0.11, 2.1),
    (7.60, 0.48, 0.48, 0.43, 0.10, 2.3),
    (6.30, 0.62, 0.60, 0.56, 0.12, 2.6),
    (5.00, 0.74, 0.70, 0.64, 0.14, 2.9),
    (3.60, 0.86, 0.76, 0.70, 0.14, 3.1),
    (2.40, 1.25, 0.76, 0.76, 0.10, 3.6),
    (1.20, 1.78, 0.74, 0.82, 0.04, 4.0),
    (-0.6, 2.02, 0.70, 0.84, 0.00, 4.2),
    (-2.6, 2.06, 0.64, 0.82, 0.00, 4.2),
    (-4.6, 1.96, 0.58, 0.78, 0.00, 4.0),
    (-6.2, 1.72, 0.54, 0.72, 0.00, 3.6),
    (-7.3, 1.50, 0.50, 0.66, 0.00, 3.2),
]
bm = bmesh.new()
loft(bm, [super_ring(*s) for s in stations])
FUSE = mesh_obj("Fuselage", bm, M_PAINT, subsurf=2)

# dorsal spine behind the canopy
bm = bmesh.new()
loft(bm, [super_ring(y, w, h, 0.05, zc, 2.4, N=32) for y, w, h, zc in
          [(3.0, 0.05, 0.05, 0.86), (2.0, 0.42, 0.22, 0.82), (0.0, 0.55, 0.16, 0.76),
           (-3.0, 0.50, 0.10, 0.68), (-5.0, 0.30, 0.04, 0.6)]])
mesh_obj("Spine", bm, M_PAINT, subsurf=2)

# canopy
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0,
                          matrix=Matrix.Translation((0, 4.05, 0.80)) @ Matrix.Diagonal((0.56, 2.3, 0.50, 1)))
mesh_obj("Canopy", bm, M_CANOPY)
# canopy frame bow
bm = bmesh.new()
add_torus(bm, (0, 0, 0), 1.0, 0.035, 40, 8)
bmesh.ops.transform(bm, verts=bm.verts[:],
                    matrix=Matrix.Translation((0, 5.2, 0.78)) @ Matrix.Diagonal((0.5, 1.0, 0.42, 1)))
mesh_obj("CanopyBow", bm, M_DARK)

# --- intakes (side boxes with dark throats) -------------------------------
for sx in (1, -1):
    bm = bmesh.new()
    rings = []
    for y, w, h, zc in [(2.9, 0.62, 0.62, -0.12), (2.0, 0.66, 0.64, -0.14), (0.0, 0.6, 0.6, -0.16),
                        (-2.5, 0.5, 0.55, -0.16)]:
        rings.append(super_ring(y, w, h, h, zc, 5.0, N=32, xc=1.22))
    # raked intake lip
    for i, p in enumerate(rings[0]):
        p.y += 0.35 * (p.z + 0.12)
    loft(bm, rings, cap_start=False)
    if sx < 0:
        mirror_x(bm)
    mesh_obj(f"Intake_{'R' if sx > 0 else 'L'}", bm, M_PAINT, subsurf=1)
    bm = bmesh.new()
    add_box(bm, Vector((1.22 * sx, 2.7, -0.13)), (1.05, 0.08, 1.05))
    mesh_obj(f"IntakeThroat_{sx}", bm, M_DARK, smooth=False)

# --- wings ------------------------------------------------------------------
def wing_sections():
    return [((1.2, 1.4, 0.06), 6.7, 0.050, (0, 0, 1)),
            ((3.6, -1.15, 0.02), 4.25, 0.042, (0, 0, 1)),
            ((6.65, -3.85, -0.04), 1.75, 0.034, (0, 0, 1))]


for sx in (1, -1):
    bm = lifting_surface(wing_sections())
    if sx < 0:
        mirror_x(bm)
    mesh_obj(f"Wing_{'R' if sx > 0 else 'L'}", bm, M_PAINT, subsurf=1)

    # horizontal stabilators
    bm = lifting_surface([((1.5, -5.75, -0.06), 3.1, 0.045, (0, 0, 1)),
                          ((4.35, -7.15, -0.16), 1.45, 0.035, (0, 0, 1))])
    if sx < 0:
        mirror_x(bm)
    mesh_obj(f"Stabilator_{'R' if sx > 0 else 'L'}", bm, M_PAINT, subsurf=1)

    # vertical tails, canted outboard 14 deg
    cant = math.radians(14.0)
    nrm = (math.cos(cant), 0, -math.sin(cant))
    span = Vector((math.sin(cant), 0, math.cos(cant)))
    root = Vector((1.08, -3.55, 0.50))
    bm = lifting_surface([(root, 3.9, 0.05, nrm),
                          (root + span * 3.05 + Vector((0, -2.75, 0)), 1.35, 0.04, nrm)])
    if sx < 0:
        mirror_x(bm)
    mesh_obj(f"VTail_{'R' if sx > 0 else 'L'}", bm, M_PAINT, subsurf=1)

    # nav light on wingtip (port red, starboard green), strobe on tail tip
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.06,
                              matrix=Matrix.Translation((6.66 * sx, -4.3, -0.04)))
    mesh_obj(f"NavLight_{sx}", bm, M_NAV_G if sx > 0 else M_NAV_R)
    bm = bmesh.new()
    tip = root + span * 3.05 + Vector((0, -2.75 - 1.35, 0))
    bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=0.05,
                              matrix=Matrix.Translation((tip.x * sx, tip.y + 0.05, tip.z)))
    mesh_obj(f"Strobe_{sx}", bm, M_STROBE)

    # wing pylon + missile
    bm = bmesh.new()
    add_box(bm, Vector((3.3 * sx, -1.6, -0.22)), (0.10, 1.8, 0.32))
    mesh_obj(f"Pylon_{sx}", bm, M_PAINT, smooth=False)

# --- missiles (AIM-class) under wings and fuselage -------------------------
def missile(x, y, z, length=3.65, r=0.09):
    bm = bmesh.new()
    add_cyl(bm, (x, y - length / 2 + 0.5, z), (x, y + length / 2 - 0.5, z), r, r, 20, caps=False)
    add_cyl(bm, (x, y + length / 2 - 0.5, z), (x, y + length / 2, z), r, 0.012, 20)
    add_cyl(bm, (x, y - length / 2, z), (x, y - length / 2 + 0.5, z), r * 0.92, r, 20, caps=True)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        rot = Matrix.Rotation(a, 4, "Y")
        for fy, fl, fs in ((y - length / 2 + 0.3, 0.45, 0.22), (y + 0.35, 0.5, 0.13)):
            off = rot @ Vector((r + fs / 2, 0, 0))
            add_box(bm, Vector((x, fy, z)) + off, (fs, fl, 0.012), rot)
    ob = mesh_obj("Missile", bm, M_MISSILE, smooth=True)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=r * 0.98,
                              matrix=Matrix.Translation((x, y + length / 2 - 0.55, z)) @
                              Matrix.Diagonal((1, 0.2, 1, 1)))
    mesh_obj("MissileBand", bm, M_RADOME)
    return ob


for sx in (1, -1):
    missile(3.3 * sx, -1.4, -0.48)
    missile(1.05 * sx, 0.9, -0.92)
    missile(1.05 * sx, -3.4, -0.90)

# --- engines: nacelles, CD nozzles with petals, hot section ----------------
NOZ_Y0, NOZ_Y1 = -7.6, -8.95
for sx in (1, -1):
    ex = ENG_X * sx
    # nacelle shroud blending out of the rear fuselage
    bm = bmesh.new()
    loft(bm, [circle_ring(y, r, ex, ENG_Z, 48) for y, r in
              [(-4.8, 0.62), (-6.4, 0.66), (-7.35, 0.665), (NOZ_Y0, 0.655)]], cap_start=False, cap_end=False)
    mesh_obj(f"Nacelle_{sx}", bm, M_PAINT, subsurf=2)

    # nozzle: outer petals (convergent-divergent), serrated trailing edge, thickness
    N = 96
    PET = 16

    def serr(th):
        ph = (th * PET / (2 * math.pi)) % 1.0
        return 0.09 * (1 - abs(ph * 2 - 1))

    def nozzle_rings(r_exit):
        prof = [(NOZ_Y0, 0.655), (-7.95, 0.64), (-8.35, 0.60), (-8.7, r_exit + 0.03)]
        outer = [circle_ring(y, r, ex, ENG_Z, N) for y, r in prof]
        lip_o = circle_ring(0, r_exit + 0.012, ex, ENG_Z, N, yfun=lambda th: NOZ_Y1 + serr(th))
        lip_i = circle_ring(0, r_exit - 0.02, ex, ENG_Z, N, yfun=lambda th: NOZ_Y1 + 0.02 + serr(th))
        inner = [circle_ring(-8.4, r_exit - 0.035, ex, ENG_Z, N), circle_ring(-7.9, 0.56, ex, ENG_Z, N),
                 circle_ring(NOZ_Y0, 0.6, ex, ENG_Z, N)]
        return outer + [lip_o, lip_i] + inner

    closed = nozzle_rings(0.43)
    opened = nozzle_rings(0.535)
    bm = bmesh.new()
    loft(bm, closed, cyclic=True)
    noz = mesh_obj(f"Nozzle_{sx}", bm, M_NOZZLE, subsurf=1)
    noz.shape_key_add(name="Closed")
    sk = noz.shape_key_add(name="Open")
    flat = [p for r in opened for p in r]
    for i, v in enumerate(sk.data):
        v.co = flat[i]

    # hot section: liner, flame holder rings, spray bars, turbine face
    bm = bmesh.new()
    add_cyl(bm, (ex, -8.6, ENG_Z), (ex, -6.6, ENG_Z), 0.555, 0.5, 48, caps=False)
    bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    add_torus(bm, (ex, -7.25, ENG_Z), 0.34, 0.03, 40, 8)
    add_torus(bm, (ex, -7.25, ENG_Z), 0.18, 0.025, 32, 8)
    for k in range(12):
        a = 2 * math.pi * k / 12
        rot = Matrix.Rotation(-a, 4, "Y")
        add_box(bm, Vector((ex, -7.1, ENG_Z)) + rot @ Vector((0.33, 0, 0)), (0.4, 0.035, 0.03), rot)
    # turbine face: hub + blades
    add_cyl(bm, (ex, -6.75, ENG_Z), (ex, -6.95, ENG_Z), 0.16, 0.1, 32)
    for k in range(40):
        a = 2 * math.pi * k / 40
        rot = Matrix.Rotation(-a, 4, "Y") @ Matrix.Rotation(math.radians(35), 4, "X")
        add_box(bm, Vector((ex, -6.7, ENG_Z)) + Matrix.Rotation(-a, 4, "Y") @ Vector((0.33, 0, 0)),
                (0.36, 0.12, 0.012), rot)
    add_cyl(bm, (ex, -6.6, ENG_Z), (ex, -6.55, ENG_Z), 0.52, 0.52, 48)
    mesh_obj(f"HotSection_{sx}", bm, M_HOT, smooth=True)

# beaver tail between the engines + arrestor hook fairing
bm = bmesh.new()
loft(bm, [super_ring(y, w, 0.12, 0.10, -0.05, 3.0, N=24) for y, w in
          [(-6.6, 0.28), (-7.6, 0.24), (-8.3, 0.16), (-8.55, 0.05)]])
mesh_obj("BeaverTail", bm, M_PAINT, subsurf=1)
bm = bmesh.new()
add_cyl(bm, (0, -2.5, -0.92), (0, -6.4, -0.82), 0.07, 0.05, 12)
mesh_obj("Hook", bm, M_PAINT)

# ventral fins
for sx in (1, -1):
    bm = bmesh.new()
    nrm = (math.cos(math.radians(-20)), 0, math.sin(math.radians(-20)))
    bm = lifting_surface([((1.0 * sx, -5.2, -0.62), 1.6, 0.05, nrm),
                          ((1.25 * sx, -5.9, -1.25), 0.9, 0.04, nrm)])
    mesh_obj(f"VentralFin_{sx}", bm, M_PAINT, subsurf=1)

# ==========================================================================
# Afterburner lighting (per engine).  The emissive plume itself is ray-marched by
# plume.py and the heat haze is applied in compose.py.
# ==========================================================================

AB_LIGHTS = []
for sx in (1, -1):
    ex = ENG_X * sx
    # exhaust glow light
    ld = bpy.data.lights.new(f"AB_Light_{sx}", "POINT")
    ld.color = (1.0, 0.45, 0.15)
    ld.shadow_soft_size = 0.55
    ld.energy = 0.0
    lo = link(bpy.data.objects.new(f"AB_Light_{sx}", ld), C_FX)
    lo.parent = JET
    lo.location = (ex, NOZ_Y1 - 3.0, ENG_Z)
    AB_LIGHTS.append(ld)

    # faint idle glow light deep in the nozzle (lights the petals from inside)
    ld = bpy.data.lights.new(f"Core_Light_{sx}", "POINT")
    ld.color = (1.0, 0.3, 0.08)
    ld.shadow_soft_size = 0.2
    ld.energy = 0.3
    lo = link(bpy.data.objects.new(f"Core_Light_{sx}", ld), C_FX)
    lo.parent = JET
    lo.location = (ex, -7.6, ENG_Z)


# ==========================================================================
# Environment: sky dome + lighting.  The volumetric cloud field is ray-marched by
# clouds.py from these same cameras and depth-composited in compose.py.
# ==========================================================================
SUN_DIR = Vector(TL.SUN_DIR)


def build_world():
    """Dark storm sky; the same palette/formula is mirrored by clouds.sky_color."""
    K = TL.SKY
    w = bpy.data.worlds.new("StormSky")
    scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    n = NB(nt)
    d = n.v("NORMALIZE", n.coords("Generated"))
    dx, dy, dz = n.sep(d)
    up = n.smooth(dz, -0.05, 0.6)
    sky = n.mix(up, (*K["horizon"], 1), (*K["zenith"], 1))
    below = n.smooth(dz, 0.02, -0.25)
    sky = n.mix(below, sky, (*K["below"], 1))
    sd = n.clamp01(n.v("DOT_PRODUCT", d, tuple(SUN_DIR)))
    glow = n.add(n.mul(n.pow(sd, 8.0), K["glow8"]), n.mul(n.pow(sd, 120.0), K["glow120"]))
    sky = n.v("ADD", sky, n.cmul(K["glow_rgb"], glow))
    # distant cloud banks painted on the dome (break up the horizon)
    cl = n.noise(n.v("MULTIPLY", d, (1.0, 1.0, 3.0)), scale=2.2, detail=6.0, rough=0.6)
    cl = n.smooth(cl, 0.42, 0.68)
    band = n.mul(n.smooth(dz, -0.3, -0.02), n.sub(1.0, n.smooth(dz, 0.12, 0.45)))
    lit = n.add(K["cloud_lit"][0], n.mul(n.pow(sd, 3.0), K["cloud_lit"][1]))
    sky = n.mix(n.mul(cl, band), sky, n.cmul(K["cloud_rgb"], lit))
    bg = n.new("ShaderNodeBackground")
    n.set(bg.inputs["Color"], sky)
    n.set(bg.inputs["Strength"], 1.0)
    o = n.new("ShaderNodeOutputWorld")
    n.set(o.inputs["Surface"], bg.outputs[0])


build_world()


# key: low sun ahead of the aircraft (backlight / rim)
sun = bpy.data.lights.new("Sun_Key", "SUN")
sun.energy = 4.0
sun.color = (1.0, 0.80, 0.64)
sun.angle = math.radians(1.2)
so = link(bpy.data.objects.new("Sun_Key", sun), C_ENV)
so.rotation_euler = SUN_DIR.to_track_quat("Z", "Y").to_euler()

# fill: light bounced off the cloud deck, only on the aircraft (light linking)
fill = bpy.data.lights.new("CloudDeck_Bounce", "SUN")
fill.energy = 0.9
fill.color = (0.62, 0.70, 0.85)
fill.angle = math.radians(25)
fo = link(bpy.data.objects.new("CloudDeck_Bounce", fill), C_ENV)
fo.rotation_euler = Vector((-0.35, -0.6, -0.7)).normalized().to_track_quat("Z", "Y").to_euler()
fo.light_linking.receiver_collection = C_JET

# cool sky rim from above-behind, aircraft only
rim = bpy.data.lights.new("Sky_Rim", "SUN")
rim.energy = 1.4
rim.color = (0.70, 0.78, 0.95)
rim.angle = math.radians(8)
ro = link(bpy.data.objects.new("Sky_Rim", rim), C_ENV)
ro.rotation_euler = Vector((-0.5, -0.4, 0.75)).normalized().to_track_quat("Z", "Y").to_euler()
ro.light_linking.receiver_collection = C_JET

# ==========================================================================
# Cameras
# ==========================================================================
CAMS = {}
for name in ("CAM_1", "CAM_2", "CAM_3"):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "AUTO"
    cd.sensor_width = 36.0
    cd.clip_start = 0.1
    cd.clip_end = 5000.0
    co = link(bpy.data.objects.new(name, cd), C_CAM)
    co.rotation_mode = "QUATERNION"
    CAMS[name] = co

for name, frame in (("CAM_1", TL.SHOT1[0]), ("CAM_2", TL.SHOT2[0]), ("CAM_3", TL.SHOT3[0])):
    m = scene.timeline_markers.new(name.replace("CAM", "SHOT"), frame=frame)
    m.camera = CAMS[name]
scene.camera = CAMS["CAM_1"]

# ==========================================================================
# Animation bake (one key per frame, linear) -- everything comes from timeline
# ==========================================================================
for k, v in (("ab", 0.0), ("ab_len", 0.0), ("time", 0.0), ("flick", 1.0), ("strobe", 0.0),
             ("trail", 0.0), ("nx", 0.0), ("ny", 0.0), ("nz", 0.0)):
    scene[k] = v

JET.rotation_mode = "XYZ"
NOZZLES = [bpy.data.objects["Nozzle_1"], bpy.data.objects["Nozzle_-1"]]
cam_fns = {"CAM_1": TL.cam1, "CAM_2": TL.cam2, "CAM_3": TL.cam3}
shot_of = {"CAM_1": TL.SHOT1, "CAM_2": TL.SHOT2, "CAM_3": TL.SHOT3}

for f in range(TL.FRAME_START - 1, TL.FRAME_END + 2):
    loc = TL.jet_location(f)
    pitch, roll, yaw = TL.jet_rotation(f)
    JET.location = loc
    JET.rotation_euler = (pitch, roll, yaw)
    JET.keyframe_insert("location", frame=f)
    JET.keyframe_insert("rotation_euler", frame=f)

    ab = TL.ab_level(f)
    fl = TL.flicker(f)
    vals = {"ab": ab, "ab_len": TL.ab_length(f) * (1.0 + 0.25 * TL.smoothstep(TL.SHOT3[0], TL.SHOT3[1], f)),
            "time": TL.t_of(f), "flick": fl, "strobe": TL.strobe(f),
            "trail": TL.smoothstep(TL.IGNITE, TL.IGNITE + 30, f)}
    nzw = Vector(loc) + Vector((0.0, NOZ_Y1 - 1.0, ENG_Z))     # exhaust glow centre (world)
    vals.update({"nx": nzw.x, "ny": nzw.y, "nz": nzw.z})
    for k, v in vals.items():
        scene[k] = v
        scene.keyframe_insert(f'["{k}"]', frame=f)

    for ld in AB_LIGHTS:
        ld.energy = 600.0 * ab * fl
        ld.keyframe_insert("energy", frame=f)
    for noz in NOZZLES:
        kb = noz.data.shape_keys.key_blocks["Open"]
        kb.value = TL.nozzle_open(f)
        kb.keyframe_insert("value", frame=f)

    # cameras: every camera gets keys across its own shot (+1 frame margins for motion blur)
    for name, co in CAMS.items():
        s0, s1 = shot_of[name]
        if not (s0 - 1 <= f <= s1 + 1):
            continue
        cloc, ctgt, lens, fstop = cam_fns[name](f)
        co.location = cloc
        dvec = Vector(ctgt) - Vector(cloc)
        q = dvec.to_track_quat("-Z", "Y")
        co.rotation_quaternion = q
        co.keyframe_insert("location", frame=f)
        co.keyframe_insert("rotation_quaternion", frame=f)
        co.data.lens = lens
        co.data.keyframe_insert("lens", frame=f)
        if fstop:
            co.data.dof.use_dof = True
            co.data.dof.aperture_fstop = fstop
            nz = Vector(loc) + Vector((0.68, -9.4, 0.0))
            co.data.dof.focus_distance = (nz - Vector(cloc)).length
            co.data.dof.keyframe_insert("focus_distance", frame=f)


# ==========================================================================
# Render settings
# ==========================================================================
r = scene.render
r.engine = "CYCLES"
r.resolution_x, r.resolution_y = 1080, 1920
r.resolution_percentage = 100
r.fps = TL.FPS
scene.frame_start, scene.frame_end = TL.FRAME_START, TL.FRAME_END
r.use_motion_blur = True
r.motion_blur_shutter = 0.5
r.film_transparent = True
r.use_persistent_data = True
r.compositor_device = "CPU"
r.use_compositing = False
r.image_settings.media_type = "IMAGE"
r.image_settings.file_format = "OPEN_EXR"
r.image_settings.color_mode = "RGBA"
r.image_settings.color_depth = "16"
r.image_settings.exr_codec = "PIZ"

cy = scene.cycles
cy.device = "CPU"
cy.samples = 16
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.03
cy.use_denoising = True
cy.denoiser = "OPENIMAGEDENOISE"
cy.denoising_quality = "BALANCED"
cy.denoising_prefilter = "FAST"
cy.max_bounces = 6
cy.diffuse_bounces = 2
cy.glossy_bounces = 2
cy.transmission_bounces = 6
cy.volume_bounces = 0
cy.volume_biased = True
cy.transparent_max_bounces = 8
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.sample_clamp_indirect = 8.0

vs = scene.view_settings
vs.view_transform = "AgX"
vs.look = "AgX - Medium High Contrast"
vs.exposure = 0.0

# Depth pass via material override (this headless Blender build writes only the
# Combined pass): emission = planar camera depth, rendered by render.py --depth.
mat, n = new_material("DepthOverride")
em = n.new("ShaderNodeEmission")
cam_data = n.new("ShaderNodeCameraData")
n.set(em.inputs["Color"], n.comb(cam_data.outputs["View Z Depth"], cam_data.outputs["View Z Depth"],
                                 cam_data.outputs["View Z Depth"]))
n.set(em.inputs["Strength"], 1.0)
n.output(em.outputs[0])
mat.use_fake_user = True

# ==========================================================================
out_path = os.path.join(ROOT, "blend", "ep01_afterburners.blend")
if "--out" in sys.argv:
    out_path = sys.argv[sys.argv.index("--out") + 1]
scene.frame_set(TL.FRAME_START)
bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)
print("SAVED", out_path)
