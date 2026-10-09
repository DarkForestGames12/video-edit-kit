"""Drawing helpers: fonts, icons, sprites (premultiplied float RGBA), blitting, easing."""
import os, glob, subprocess
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from .common import ASSETS


def font_path(spec):
    """spec 'family/weight', e.g. 'montserrat/900Black', 'anton/400Regular' (any @expo-google-fonts family)."""
    fam, weight = spec.split("/")
    d = os.path.join(ASSETS, f"_expo-google-fonts_{fam}")
    if not os.path.isdir(d):
        t = subprocess.run(["npm", "pack", "--silent", f"@expo-google-fonts/{fam}"], cwd=ASSETS,
                           capture_output=True, text=True).stdout.strip().splitlines()[-1]
        os.makedirs(d, exist_ok=True)
        subprocess.run(["tar", "xzf", os.path.join(ASSETS, t), "-C", d], check=True)
        os.remove(os.path.join(ASSETS, t))
    hits = glob.glob(os.path.join(d, "package", weight, "*.ttf"))
    if not hits:
        raise FileNotFoundError(f"font {spec}: available weights {os.listdir(os.path.join(d, 'package'))}")
    return hits[0]


FA = os.path.join(ASSETS, "_fortawesome_fontawesome-free", "package")


@lru_cache(None)
def icon_codes():
    import yaml
    with open(os.path.join(FA, "metadata", "icons.yml")) as f:
        d = yaml.safe_load(f)
    return {k: int(v["unicode"], 16) for k, v in d.items() if "solid" in v.get("styles", ["solid"])}


def icon_char(name):
    codes = icon_codes()
    n = name.replace("_", "-")
    if n not in codes:
        raise KeyError(f"icon '{name}' not in Font Awesome Free solid set")
    return chr(codes[n])


def hexc(h, a=1.0):
    if isinstance(h, (tuple, list)):
        return tuple(h) if len(h) == 4 else tuple(h) + (int(a * 255),)
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (int(a * 255),)


def rgbf(h):
    return np.array(hexc(h)[:3], np.float32) / 255


@lru_cache(None)
def font(spec, size):
    if spec == "fa":
        return ImageFont.truetype(os.path.join(FA, "webfonts", "fa-solid-900.woff2"), size)
    return ImageFont.truetype(font_path(spec), size)


def pil_to_premul(im):
    a = np.asarray(im, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def text_sprite(txt, fspec, size, fill, stroke=0, stroke_fill="#071A19", tracking=0, pad=None, shadow=None):
    f = font(fspec, size)
    asc, desc = f.getmetrics()
    pad = pad if pad is not None else stroke + 4
    if tracking:
        widths = [f.getlength(c) for c in txt]
        w = int(sum(widths) + tracking * (len(txt) - 1)) + 2 * pad
    else:
        w = int(f.getlength(txt)) + 2 * pad
    h = asc + desc + 2 * pad
    im = Image.new("RGBA", (max(1, w), h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fc, sc = hexc(fill), hexc(stroke_fill)
    if tracking:
        x = pad
        for c, cw in zip(txt, widths):
            d.text((x, pad + asc), c, font=f, fill=fc, anchor="ls", stroke_width=stroke, stroke_fill=sc)
            x += cw + tracking
    else:
        d.text((pad, pad + asc), txt, font=f, fill=fc, anchor="ls", stroke_width=stroke, stroke_fill=sc)
    spr = pil_to_premul(im)
    return add_shadow(spr, *shadow) if shadow else spr


def icon_sprite(name, size, fill, stroke=0, stroke_fill="#071A19"):
    ch = icon_char(name)
    f = font("fa", size)
    pad = stroke + 6
    bb = f.getbbox(ch, stroke_width=stroke)
    w = bb[2] - bb[0] + 2 * pad; h = bb[3] - bb[1] + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), ch, font=f, fill=hexc(fill), stroke_width=stroke,
                            stroke_fill=hexc(stroke_fill))
    return pil_to_premul(im)


def add_shadow(spr, blur=10, dx=0, dy=6, op=0.5, color=(0, 0, 0)):
    m = int(blur * 2 + max(abs(dx), abs(dy)))
    h, w = spr.shape[:2]
    out = np.zeros((h + 2 * m, w + 2 * m, 4), np.float32)
    a = np.zeros((h + 2 * m, w + 2 * m), np.float32)
    a[m + dy:m + dy + h, m + dx:m + dx + w] = spr[..., 3]
    a = cv2.GaussianBlur(a, (0, 0), blur) * op
    out[..., :3] = a[..., None] * (np.array(color, np.float32) / 255.0)
    out[..., 3] = a
    reg = out[m:m + h, m:m + w]
    reg[...] = reg * (1 - spr[..., 3:4]) + spr
    return out


def rounded_rect(w, h, r, fill, border=0, border_fill=None, ss=2):
    im = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], r * ss, fill=hexc(fill))
    if border:
        d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], r * ss, outline=hexc(border_fill), width=border * ss)
    return pil_to_premul(im.resize((w, h), Image.LANCZOS))


def circle_sprite(d_, fill, ss=3):
    im = Image.new("RGBA", (d_ * ss, d_ * ss), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([0, 0, d_ * ss - 1, d_ * ss - 1], fill=hexc(fill))
    return pil_to_premul(im.resize((d_, d_), Image.LANCZOS))


def ring_sprite(d, color, width=8):
    im = Image.new("RGBA", (d * 2, d * 2), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([4, 4, d * 2 - 5, d * 2 - 5], outline=hexc(color), width=width)
    return pil_to_premul(im.resize((d, d), Image.LANCZOS))


def blit(dst, spr, x, y, scale=1.0, rot=0.0, op=1.0, ax=0.5, ay=0.5, clip=None):
    if op <= 0.003 or scale <= 0.01:
        return
    h, w = spr.shape[:2]
    c = np.cos(np.radians(rot)) * scale; s = np.sin(np.radians(rot)) * scale
    ox, oy = ax * w, ay * h
    M = np.array([[c, -s, x - (c * ox - s * oy)], [s, c, y - (s * ox + c * oy)]], np.float32)
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    H, W = dst.shape[:2]
    x0 = int(np.floor(max(0, pts[:, 0].min()))); x1 = int(np.ceil(min(W, pts[:, 0].max())))
    y0 = int(np.floor(max(0, pts[:, 1].min()))); y1 = int(np.ceil(min(H, pts[:, 1].max())))
    if clip is not None:
        x0 = max(x0, int(clip[0])); y0 = max(y0, int(clip[1])); x1 = min(x1, int(clip[2])); y1 = min(y1, int(clip[3]))
    if x1 <= x0 or y1 <= y0:
        return
    M2 = M.copy(); M2[0, 2] -= x0; M2[1, 2] -= y0
    if abs(scale - 1) < 1e-4 and abs(rot) < 1e-4 and abs(M2[0, 2] - round(M2[0, 2])) < 1e-3 and abs(M2[1, 2] - round(M2[1, 2])) < 1e-3:
        tx, ty = int(round(-M2[0, 2])), int(round(-M2[1, 2]))
        warped = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
        sx0, sy0 = max(0, tx), max(0, ty)
        sx1, sy1 = min(w, tx + x1 - x0), min(h, ty + y1 - y0)
        if sx1 > sx0 and sy1 > sy0:
            warped[sy0 - ty:sy1 - ty, sx0 - tx:sx1 - tx] = spr[sy0:sy1, sx0:sx1]
    else:
        warped = cv2.warpAffine(spr, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4] * op
    reg = dst[y0:y1, x0:x1]
    if dst.shape[2] == 4:
        reg[...] = reg * (1 - a) + warped * op
    else:
        reg[...] = reg * (1 - a) + warped[..., :3] * op


def over(dst, layer, op=1.0):
    a = layer[..., 3:4] * op
    if dst.shape[2] == 4:
        dst[...] = dst * (1 - a) + layer * op
    else:
        dst *= (1 - a)
        dst += layer[..., :3] * op


def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def lin(t, t0, t1): return clamp((t - t0) / (t1 - t0)) if t1 > t0 else float(t >= t1)
def e_out(x): x = clamp(x); return 1 - (1 - x) ** 3
def e_in(x): x = clamp(x); return x ** 3
def e_io(x): x = clamp(x); return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2
def e_back(x, s=1.70158): x = clamp(x); return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2
def smooth(x): x = clamp(x); return x * x * (3 - 2 * x)
