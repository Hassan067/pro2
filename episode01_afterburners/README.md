# Episode 01: Why Fighter Jets Need Afterburners

A 14-second vertical (1080×1920, 30 fps, H.264) cinematic 3D documentary shot. Everything is
generated procedurally from Python: aircraft, materials, afterburner, clouds, lighting, cameras.

| File | What it is |
|---|---|
| `output/EP01_Why_Fighter_Jets_Need_Afterburners.mp4` | Final cut with documentary titles |
| `output/EP01_Why_Fighter_Jets_Need_Afterburners_clean.mp4` | Same shot without titles |
| `blend/ep01_afterburners.blend` | Blender 5.2 project: aircraft, materials, lights, 3 animated cameras with shot markers |
| `scripts/` | The whole pipeline (see below) |
| `previews/` | Look-development stills from the iteration passes |

## Verified output

| File | Codec | Resolution | FPS | Frames | Duration | Size |
|---|---|---|---|---|---|---|
| `EP01_Why_Fighter_Jets_Need_Afterburners.mp4` | H.264 High, yuv420p | 1080×1920 | 30 | 420 | 14.00 s | 38.1 MB |
| `EP01_Why_Fighter_Jets_Need_Afterburners_clean.mp4` | H.264 High, yuv420p | 1080×1920 | 30 | 420 | 14.00 s | 39.8 MB |

Checked with `ffprobe -count_frames`, and a full decode (`ffmpeg -v error -f null`) reports
0 errors. Stills from the encoded video: `previews/final_video_stills.png`.

Final render: about 2 h 10 min on 4 CPU cores (Cycles about 17 s/frame, with clouds and
compositing running concurrently).

## Shots

1. **Opening (0–5 s).** Chase camera behind and below the jet through a storm-cloud canyon at
   military (dry) power. The nozzle interiors glow dull red.
2. **Afterburner ignition (5–9.5 s).** The camera closes on the nozzles from the starboard quarter.
   At 6.37 s the nozzle petals open and the reheat lights off with an overshoot "pop": a blue-white
   core, Mach shock diamonds, turbulent orange plume, heat haze, and orange light washing the
   airframe. The camera shakes on light-off.
3. **Acceleration (9.5–14 s).** The jet accelerates (150 → 470 m/s) away from the camera, which pulls
   back and climbs while zooming. The jet punches through two cloud banks, lighting them from
   inside with its exhaust, and disappears into the storm.

## Pipeline

```
build_scene.py    Blender scene: lofted fuselage, NACA-section wings/tails, CD nozzles with
                  serrated petals (shape-key opening), augmentor/turbine detail, missiles,
                  procedural paint (panel lines, camo, grime), canopy, nav/strobe lights,
                  storm-sky world, sun/fill/rim (light-linked), cameras, per-frame animation
                  baked from timeline.py  ->  blend/ep01_afterburners.blend
timeline.py       Every animated quantity as a function of frame (shots, speeds, cameras,
                  afterburner state, shared sun/sky palette)
render.py         Cycles render of the aircraft layer (transparent film, linear EXR) + depth pass
export_frames.py  Exact camera transforms at 9 shutter instants per frame, jet state
clouds.py         Numba volumetric cloud renderer (see below)
plume.py          Numba emissive afterburner-plume renderer, depth-tested against the airframe
compose.py        Depth compositing, heat haze, bloom, dispersion, vignette, AgX tone mapping
post.py           Title overlays + FFmpeg H.264 encode + ffprobe verification
run_final.sh      Runs build -> export -> Cycles and compositor concurrently (resumable)
```

Run with Blender as a Python module (`pip install bpy==5.2.3 numba scipy OpenEXR`):

```
scripts/run_final.sh 1 420
python scripts/post.py --frames work/comp --out output
```

### Why the clouds aren't rendered in Cycles

The first version rendered the clouds in Cycles: a procedural volume, then a Geometry-Nodes voxel
grid with baked lighting. On the 4-core, GPU-less render machine that cost about 2–10 minutes per
1080×1920 frame. So the hero aircraft is rendered in Cycles, and the cloudscape is ray-marched
by `clouds.py` from the same cameras:

* a domain-warped fBm cumulus field baked into a 3 m world-anchored grid along the 3.5 km flight
  path, with a meandering cloud canyon, wisps and two cloud banks
* sun transmittance and sky visibility baked per voxel (self-shadowing), a dual-lobe
  Henyey-Greenstein phase function, a multiple-scattering term and height-graded ambient
* the afterburner lights nearby cloud (inverse-square), plus the exhaust trail and storm haze
* true camera motion blur: each sample picks a random instant in the 0.5-frame shutter and
  interpolates the exported Blender camera

`compose.py` splits each cloud ray at the aircraft's depth:
`final = clouds_front + T_front * (aircraft + plume + (1 - alpha) * clouds_behind)`.
That way cloud in front of the jet hides it, which is what happens when it punches through the
banks. The sky formula is shared with the Blender world shader, so reflections on the airframe
match the background.

## Render settings

* Aircraft layer: Cycles CPU, 16 spp adaptive + OpenImageDenoise, motion blur (0.5 shutter),
  1080×1920
* Clouds: 540×960, 3 shutter/AA samples per pixel, upsampled
* Plume: 540×960 emissive ray-march, 40 steps per engine
* AgX view transform, "Medium High Contrast" look

## Notes / limitations

* The aircraft is a generic twin-engine air-superiority fighter in the F-15/F-22 class, built
  procedurally. External 3D-asset hosts were unreachable from the render environment, so no
  third-party models are used.
* The video has no audio track.
* "Roughly +50% thrust" is a representative figure. Afterburning turbofans typically gain about
  35–65% thrust in reheat, at several times the fuel flow.
