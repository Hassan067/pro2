"""Titles, encode and verification.

Builds transparent title overlays with PIL, then encodes the composited frames
with FFmpeg to H.264 MP4 (1080x1920, 30 fps): a titled cut and a clean cut.

  python post.py --frames work/comp --out output
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as TL  # noqa: E402

W, H = 1080, 1920
FONT_DIR = "/usr/share/fonts/opentype/inter"
F_BOLD = os.path.join(FONT_DIR, "Inter-Bold.otf")
F_MED = os.path.join(FONT_DIR, "Inter-Medium.otf")
F_LIGHT = os.path.join(FONT_DIR, "Inter-Light.otf")
ACCENT = (255, 150, 60)

# (start s, end s, kicker, headline lines, sub line)
TITLES = [
    (0.6, 4.6, "EPISODE 01", ["WHY FIGHTER JETS", "NEED AFTERBURNERS"], None),
    (TL.t_of(TL.IGNITE) + 0.25, TL.t_of(TL.SHOT2[1]) - 0.15, "AFTERBURNER IGNITION",
     ["Fuel sprayed into the", "hot exhaust ignites"], "roughly +50% thrust, at several times the fuel burn"),
    (TL.t_of(TL.SHOT3[0]) + 0.5, TL.t_of(TL.SHOT3[1]) - 0.6, "ACCELERATION",
     ["Extra thrust to punch", "through the sound barrier"], None),
]


def font(path, size):
    return ImageFont.truetype(path, size)


def tracked(draw, xy, text, fnt, fill, tracking):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + tracking
    return x


def make_overlay(kicker, lines, sub, path):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # soft dark gradient behind the text for legibility
    grad = Image.new("L", (1, 760))
    for i in range(760):
        grad.putpixel((0, i), int(150 * (i / 759) ** 1.6))
    shade = Image.new("RGBA", (W, 760), (0, 0, 0, 255))
    shade.putalpha(grad.resize((W, 760)))
    im.alpha_composite(shade, (0, H - 760))
    d = ImageDraw.Draw(im)
    x0 = 84
    y = H - 520
    k = font(F_MED, 34)
    d.rectangle((x0, y + 8, x0 + 6, y + 40), fill=ACCENT + (255,))
    tracked(d, (x0 + 26, y), kicker, k, ACCENT + (255,), 7)
    y += 78
    hb = font(F_BOLD, 76)
    for ln in lines:
        d.text((x0, y), ln, font=hb, fill=(244, 244, 246, 255))
        y += 92
    if sub:
        y += 14
        d.text((x0, y), sub, font=font(F_LIGHT, 36), fill=(214, 218, 226, 235))
    im.save(path)


def run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt",
                          "-show_entries", "format=duration,size", "-of", "json", path],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tdir = os.path.join(args.out, "..", "work", "titles")
    os.makedirs(tdir, exist_ok=True)

    n = TL.FRAME_END - TL.FRAME_START + 1
    dur = n / TL.FPS
    src = ["-framerate", str(TL.FPS), "-start_number", str(TL.FRAME_START), "-i",
           os.path.join(args.frames, "f%04d.png")]
    grade = (f"fade=t=in:st=0:d=0.35,fade=t=out:st={dur - 0.5:.3f}:d=0.5,"
             "noise=alls=3:allf=t,format=yuv420p")
    enc = ["-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-r", str(TL.FPS), "-movflags", "+faststart", "-an"]

    clean = os.path.join(args.out, "EP01_Why_Fighter_Jets_Need_Afterburners_clean.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", *src, "-vf", grade, *enc, clean])

    inputs, chain, last = [], [], "[0:v]"
    for i, (t0, t1, kick, lines, sub) in enumerate(TITLES):
        p = os.path.join(tdir, f"title_{i}.png")
        make_overlay(kick, lines, sub, p)
        inputs += ["-loop", "1", "-t", f"{dur:.3f}", "-i", p]
        chain.append(f"[{i + 1}:v]format=rgba,fade=t=in:st={t0:.3f}:d=0.45:alpha=1,"
                     f"fade=t=out:st={t1 - 0.45:.3f}:d=0.45:alpha=1[t{i}]")
        chain.append(f"{last}[t{i}]overlay=0:0:enable='between(t,{t0:.3f},{t1:.3f})'[v{i}]")
        last = f"[v{i}]"
    chain.append(f"{last}{grade}[out]")
    titled = os.path.join(args.out, "EP01_Why_Fighter_Jets_Need_Afterburners.mp4")
    run(["ffmpeg", "-y", "-loglevel", "error", *src, *inputs, "-filter_complex", ";".join(chain),
         "-map", "[out]", *enc, titled])

    for path in (titled, clean):
        info = probe(path)
        st, fm = info["streams"][0], info["format"]
        print(f"VERIFY {os.path.basename(path)}: {st['codec_name']} {st['width']}x{st['height']} "
              f"{st['r_frame_rate']} fps, {st['nb_read_frames']} frames, {float(fm['duration']):.2f}s, "
              f"{int(fm['size']) / 1e6:.1f} MB, {st['pix_fmt']}")


if __name__ == "__main__":
    main()
