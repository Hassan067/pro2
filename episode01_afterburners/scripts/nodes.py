"""Small helper layer for building Cycles shader node trees from Python."""
import bpy


class NB:
    """Node builder: wraps a node tree and lets math be written as nested calls."""

    def __init__(self, nt):
        self.nt = nt

    def new(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, src, dst):
        self.nt.links.new(src, dst)

    def set(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.nt.links.new(v, sock)
        else:
            sock.default_value = v

    # --- scalar / vector math -------------------------------------------
    def m(self, op, a, b=None, c=None, clamp=False):
        n = self.new("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.set(n.inputs[0], a)
        self.set(n.inputs[1], b)
        self.set(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b): return self.m("ADD", a, b)
    def sub(self, a, b): return self.m("SUBTRACT", a, b)
    def mul(self, a, b): return self.m("MULTIPLY", a, b)
    def div(self, a, b): return self.m("DIVIDE", a, b)
    def pow(self, a, b): return self.m("POWER", a, b)
    def clamp01(self, a): return self.m("MULTIPLY", a, 1.0, clamp=True)

    def v(self, op, a, b=None, scale=None):
        n = self.new("ShaderNodeVectorMath", operation=op)
        self.set(n.inputs[0], a)
        self.set(n.inputs[1], b)
        if scale is not None:
            self.set(n.inputs["Scale"], scale)
        out = n.outputs["Value"] if op in ("LENGTH", "DOT_PRODUCT", "DISTANCE") else n.outputs["Vector"]
        return out

    def sep(self, vec):
        n = self.new("ShaderNodeSeparateXYZ")
        self.set(n.inputs[0], vec)
        return n.outputs["X"], n.outputs["Y"], n.outputs["Z"]

    def comb(self, x, y, z):
        n = self.new("ShaderNodeCombineXYZ")
        self.set(n.inputs[0], x)
        self.set(n.inputs[1], y)
        self.set(n.inputs[2], z)
        return n.outputs[0]

    def smooth(self, val, lo, hi, to_lo=0.0, to_hi=1.0):
        n = self.new("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self.set(n.inputs["Value"], val)
        self.set(n.inputs["From Min"], lo)
        self.set(n.inputs["From Max"], hi)
        self.set(n.inputs["To Min"], to_lo)
        self.set(n.inputs["To Max"], to_hi)
        return n.outputs["Result"]

    def lin(self, val, lo, hi, to_lo=0.0, to_hi=1.0, clamp=True):
        n = self.new("ShaderNodeMapRange", interpolation_type="LINEAR", clamp=clamp)
        self.set(n.inputs["Value"], val)
        self.set(n.inputs["From Min"], lo)
        self.set(n.inputs["From Max"], hi)
        self.set(n.inputs["To Min"], to_lo)
        self.set(n.inputs["To Max"], to_hi)
        return n.outputs["Result"]

    # --- textures / colour ------------------------------------------------
    def noise(self, vec, scale=1.0, detail=4.0, rough=0.5, w=None, dims="3D",
              distortion=0.0, lacunarity=2.0, color=False):
        n = self.new("ShaderNodeTexNoise", noise_dimensions=dims)
        self.set(n.inputs["Vector"], vec)
        if w is not None:
            self.set(n.inputs["W"], w)
        self.set(n.inputs["Scale"], scale)
        self.set(n.inputs["Detail"], detail)
        self.set(n.inputs["Roughness"], rough)
        self.set(n.inputs["Lacunarity"], lacunarity)
        self.set(n.inputs["Distortion"], distortion)
        return n.outputs[1] if color else n.outputs[0]

    def ramp(self, fac, stops):
        n = self.new("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.interpolation = "B_SPLINE" if len(stops) > 2 else "LINEAR"
        while len(cr.elements) < len(stops):
            cr.elements.new(0.5)
        for el, (pos, col) in zip(cr.elements, stops):
            el.position = pos
            el.color = col if len(col) == 4 else (*col, 1.0)
        self.set(n.inputs["Fac"], fac)
        return n.outputs["Color"]

    def mix(self, fac, a, b, blend="MIX"):
        n = self.new("ShaderNodeMix", data_type="RGBA", blend_type=blend, clamp_result=False)
        self.set(n.inputs["Factor"], fac)
        self.set(n.inputs[6], a)
        self.set(n.inputs[7], b)
        return n.outputs[2]

    def cmul(self, col, s):
        """Scale a colour (or vector) by a scalar socket or float."""
        n = self.new("ShaderNodeVectorMath", operation="SCALE")
        self.set(n.inputs[0], col)
        self.set(n.inputs["Scale"], s)
        return n.outputs["Vector"]

    def attr(self, name, kind="VIEW_LAYER"):
        n = self.new("ShaderNodeAttribute", attribute_type=kind, attribute_name=name)
        return n.outputs["Fac"]

    def coords(self, which="Object"):
        if which == "GN":
            return self.new("GeometryNodeInputPosition").outputs[0]
        if which == "Position":
            return self.new("ShaderNodeNewGeometry").outputs["Position"]
        return self.new("ShaderNodeTexCoord").outputs[which]

    def output(self, surface=None, volume=None, target="ALL"):
        o = self.new("ShaderNodeOutputMaterial", target=target)
        self.set(o.inputs["Surface"], surface)
        self.set(o.inputs["Volume"], volume)
        return o


def new_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    return mat, NB(nt)
