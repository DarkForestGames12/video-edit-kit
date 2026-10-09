"""Stage 3 - colour grade. Uses the person matte to treat subject and background separately:
background is dimmed, cooled and softly blurred (fake depth of field; rescues blown-out walls),
the subject is lifted, warmed and lightly cleaned. Light touch on skin: heavy smoothing +
sharpening makes upscaled webcam faces look blotchy/painted."""
import numpy as np, cv2

DEFAULT = dict(
    lut_gamma=0.80, lut_s=0.25, lift=0.02, gain=0.955, saturation=1.08,
    subj_gamma=0.84, subj_exposure=1.06, subj_warm=[1.025, 1.0, 0.965], subj_smooth=0.30, subj_clarity=0.10,
    bg_dim=0.84, bg_tint=[0.955, 1.0, 1.025], bg_blur=3.0, bg_light=0.50, bg_light_min=0.52,
    simple_knee=0.70, simple_roll=0.55,
)


def make_lut(p):
    x = np.linspace(0, 1, 256)
    y = x ** p["lut_gamma"]
    s = 1 / (1 + np.exp(-(y - 0.5) * 5.0)); s = (s - s.min()) / (s.max() - s.min())
    y = (1 - p["lut_s"]) * y + p["lut_s"] * s
    y = p["lift"] + y * p["gain"]
    return (np.clip(y, 0, 1) * 255).astype(np.uint8)


class Grader:
    def __init__(self, params=None):
        self.p = dict(DEFAULT)
        if params: self.p.update(params)
        self.LUT = make_lut(self.p)
        self._grid = {}

    def grid(self, H, W):
        if (H, W) not in self._grid:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            self._grid[(H, W)] = (yy, xx)
        return self._grid[(H, W)]

    def finish(self, out):
        out = cv2.LUT(out, self.LUT)
        if abs(self.p["saturation"] - 1) > 1e-3:
            hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * self.p["saturation"], 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return out

    def full(self, fr, al, headx, heady):
        """fr uint8 RGB, al uint8 matte (0-255). Returns uint8 RGB."""
        p = self.p
        H, W = fr.shape[:2]
        a = cv2.GaussianBlur(al.astype(np.float32) / 255.0, (0, 0), 1.0)[..., None]
        f = fr.astype(np.float32) / 255.0
        inv = 1 - a
        bl = cv2.GaussianBlur(f * inv, (0, 0), p["bg_blur"]); wb = cv2.GaussianBlur(inv, (0, 0), p["bg_blur"])[..., None]
        bg = np.where(wb > 1e-3, bl / np.maximum(wb, 1e-3), f)
        yy, xx = self.grid(H, W)
        d = np.sqrt(((xx - headx) / W) ** 2 + ((yy - heady) / H) ** 2 * 1.3)[..., None]
        light = np.clip(1.0 - p["bg_light"] * d, p["bg_light_min"], 1.0)
        bg = bg * p["bg_dim"] * light * np.array(p["bg_tint"], np.float32)
        sub = np.ascontiguousarray(fr)
        if p["subj_smooth"] > 0:
            sm = cv2.bilateralFilter(sub, 7, 22, 5)
            sub = ((1 - p["subj_smooth"]) * sub.astype(np.float32) + p["subj_smooth"] * sm.astype(np.float32)) / 255.0
        else:
            sub = sub.astype(np.float32) / 255.0
        sub = np.clip(sub, 0, 1) ** p["subj_gamma"] * p["subj_exposure"]
        sub = sub * np.array(p["subj_warm"], np.float32)
        if p["subj_clarity"] > 0:
            blur = cv2.GaussianBlur(sub, (0, 0), 18)
            sub = sub + p["subj_clarity"] * (sub - blur)
        out = (np.clip(sub * a + bg * (1 - a), 0, 1) * 255).astype(np.uint8)
        return self.finish(out)

    def simple(self, fr):
        """Per-pixel only (no matte): used for long videos; identical to the exported .cube LUT."""
        p = self.p
        f = fr.astype(np.float32) / 255.0
        f = np.clip(f, 0, 1) ** p["subj_gamma"] * p["subj_exposure"] * np.array(p["subj_warm"], np.float32)
        k = p["simple_knee"]            # highlight roll-off: tames blown walls/windows without a matte
        f = np.where(f > k, k + (f - k) * p["simple_roll"], f)
        out = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        return self.finish(out)

    def export_cube(self, path, n=33):
        g = np.linspace(0, 1, n)
        b, gg, r = np.meshgrid(g, g, g, indexing="ij")    # .cube order: R fastest
        rgb = np.stack([r, gg, b], -1).reshape(-1, 1, 3)
        img = (rgb * 255 + 0.5).astype(np.uint8)
        out = self.simple(img).reshape(-1, 3).astype(np.float64) / 255
        with open(path, "w") as f:
            f.write(f"TITLE \"vek\"\nLUT_3D_SIZE {n}\n")
            for v in out:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
