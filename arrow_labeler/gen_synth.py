import argparse
import math
import os
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SS = 2  # supersampling factor for antialiasing

WORDS = (
    "API Gateway Server Client Database Queue Cache Storage Service Load Balancer User Router Auth Lambda Bucket "
    "Worker Web App Mobile Backend Frontend Kafka Redis Postgres Cluster Node Pod Container Registry Pipeline Build "
    "Deploy Test Monitor Logs Metrics Events Stream Topic Function Proxy Firewall VPC Subnet Region Zone Cloud CDN DNS "
    "Request Response Data Model Train Batch ETL Lake Warehouse Query Index Search Notify Email SMS Push Token Session "
    "Config Secret Vault Scheduler Cron Job Task Step Process Start End Yes No Input Output Read Write Sync Async "
    "HTTPS TCP gRPC REST SQL S3 EC2 ECS EKS IAM SNS SQS Controller Repository Entity Adapter Broker Publisher Consumer"
).split()

PALETTE = [
    (232, 114, 0), (35, 47, 62), (0, 115, 187), (61, 133, 34), (201, 37, 45), (125, 60, 152), (0, 150, 136),
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88), (96, 125, 139), (121, 85, 72), (233, 30, 99),
]


def _find_fonts():
    try:
        out = subprocess.run(["fc-list", ":lang=en", "file"], capture_output=True, text=True).stdout
    except OSError:
        return []
    fonts = []
    for line in out.splitlines():
        path = line.split(":")[0].strip()
        low = path.lower()
        if not low.endswith((".ttf", ".otf")):
            continue
        if any(k in low for k in ("dejavu", "liberation", "ubuntu", "noto sans", "notosans-", "freesans", "roboto", "opensans", "lato", "carlito", "caladea")):
            if not any(k in low for k in ("symbol", "emoji", "math", "mono-")):
                fonts.append(path)
    return sorted(set(fonts))[:60]


FONTS = _find_fonts()

# Optional folder of real PNG icons (e.g. the `resources/` folder of github.com/mingrammer/diagrams)
ICONS_DIR = os.environ.get("ARROW_ICONS_DIR")
ICONS = sorted(str(p) for p in Path(ICONS_DIR).rglob("*.png")) if ICONS_DIR else []


def paste_icon(img, rng, cx, y0, side):
    try:
        icon = Image.open(rng.choice(ICONS)).convert("RGBA")
    except OSError:
        return None
    k = s(side) / max(icon.size)
    icon = icon.resize((max(2, int(icon.width * k)), max(2, int(icon.height * k))), Image.LANCZOS)
    x = int(s(cx) - icon.width / 2)
    if rng.random() < 0.12:  # icon on its own white tile, as in screenshots pasted into slides
        tile = Image.new("RGBA", icon.size, (255, 255, 255, 255))
        tile.alpha_composite(icon)
        icon = tile
    img.paste(icon, (x, int(s(y0))), icon)
    return (x / SS, y0, (x + icon.width) / SS, y0 + icon.height / SS)


_font_cache = {}


def font(rng, size):
    path = rng.choice(FONTS) if FONTS else None
    key = (path, size)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        except OSError:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def s(v):
    return v * SS


def pts_s(pts):
    return [(x * SS, y * SS) for x, y in pts]


def lerp_color(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def path_length(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def trim_path(pts, d_start, d_end):
    pts = list(pts)
    for rev, d in ((False, d_start), (True, d_end)):
        if d <= 0:
            continue
        if rev:
            pts.reverse()
        while len(pts) >= 2 and d > 0:
            seg = math.dist(pts[0], pts[1])
            if seg <= d and len(pts) > 2:
                d -= seg
                pts.pop(0)
            else:
                t = min(d / seg, 0.95) if seg > 0 else 0
                pts[0] = (pts[0][0] + (pts[1][0] - pts[0][0]) * t, pts[0][1] + (pts[1][1] - pts[0][1]) * t)
                d = 0
        if rev:
            pts.reverse()
    return pts


def draw_polyline(draw, pts, color, width, dash=None):
    w = max(1, int(round(s(width))))
    P = pts_s(pts)
    if dash is None:
        draw.line(P, fill=color, width=w, joint="curve")
        return
    on, off = s(dash[0]), s(dash[1])
    pos, drawing = 0.0, True
    for i in range(len(P) - 1):
        (x0, y0), (x1, y1) = P[i], P[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg == 0:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        t = 0.0
        while t < seg:
            remaining = (on if drawing else off) - pos
            step = min(remaining, seg - t)
            if drawing:
                draw.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * (t + step), y0 + uy * (t + step))], fill=color, width=w)
            t += step
            pos += step
            if pos >= (on if drawing else off) - 1e-6:
                pos, drawing = 0.0, not drawing


def head_dir(pts, back=6.0):
    tip = pts[-1]
    acc = 0.0
    for i in range(len(pts) - 2, -1, -1):
        acc += math.dist(pts[i], pts[i + 1])
        if acc >= back or i == 0:
            dx, dy = tip[0] - pts[i][0], tip[1] - pts[i][1]
            n = math.hypot(dx, dy) or 1.0
            return dx / n, dy / n
    return 1.0, 0.0


def draw_head(draw, tip, d, style, L, hw, color, width, bg):
    ux, uy = d
    px, py = -uy, ux
    bx, by = tip[0] - ux * L, tip[1] - uy * L
    left, right = (bx + px * hw, by + py * hw), (bx - px * hw, by - py * hw)
    if style == "tri":
        draw.polygon(pts_s([tip, left, right]), fill=color)
    elif style == "stealth":
        notch = (tip[0] - ux * L * 0.65, tip[1] - uy * L * 0.65)
        draw.polygon(pts_s([tip, left, notch, right]), fill=color)
    elif style == "open":
        w = max(1, int(round(s(width))))
        draw.line(pts_s([left, tip, right]), fill=color, width=w, joint="curve")
    elif style == "hollow":
        draw.polygon(pts_s([tip, left, right]), fill=bg, outline=color, width=max(1, int(round(s(max(1, width * 0.8))))))


def draw_tail_deco(draw, p, d, kind, size, color, bg):
    x, y = p
    if kind == "dot":
        draw.ellipse([s(x - size), s(y - size), s(x + size), s(y + size)], fill=color)
    elif kind == "ring":
        draw.ellipse([s(x - size), s(y - size), s(x + size), s(y + size)], fill=bg, outline=color, width=max(1, int(s(1.2))))
    elif kind == "diamond":
        ux, uy = d
        px, py = -uy, ux
        a = (x, y)
        b = (x + ux * size + px * size * 0.6, y + uy * size + py * size * 0.6)
        c = (x + ux * 2 * size, y + uy * 2 * size)
        e = (x + ux * size - px * size * 0.6, y + uy * size - py * size * 0.6)
        draw.polygon(pts_s([a, b, c, e]), fill=color)
    elif kind == "bar":
        ux, uy = d
        px, py = -uy, ux
        draw.line(pts_s([(x + px * size, y + py * size), (x - px * size, y - py * size)]), fill=color, width=max(1, int(s(1.5))))


def bezier(p0, p1, p2, p3=None, n=40):
    out = []
    for i in range(n + 1):
        t = i / n
        if p3 is None:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]
        else:
            x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t * t * p2[0] + t ** 3 * p3[0]
            y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t * t * p2[1] + t ** 3 * p3[1]
        out.append((x, y))
    return out


def round_corners(pts, r):
    if len(pts) < 3 or r <= 0:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        la, lc = math.dist(a, b), math.dist(b, c)
        rr = min(r, la / 2, lc / 2)
        if rr < 1:
            out.append(b)
            continue
        p = (b[0] + (a[0] - b[0]) * rr / la, b[1] + (a[1] - b[1]) * rr / la)
        q = (b[0] + (c[0] - b[0]) * rr / lc, b[1] + (c[1] - b[1]) * rr / lc)
        out.extend(bezier(p, b, q, n=8))
    out.append(pts[-1])
    return out


def rand_text(rng, n=None):
    n = n or rng.choice([1, 1, 2, 2, 3])
    return " ".join(rng.choice(WORDS) for _ in range(n))


def draw_text_center(draw, rng, cx, cy, text, size, color, max_w=None):
    f = font(rng, int(s(size)))
    if max_w and " " in text and draw.textlength(text, font=f) > s(max_w):
        text = text.replace(" ", "\n", 1)
    box = draw.multiline_textbbox((0, 0), text, font=f, align="center")
    w, h = box[2] - box[0], box[3] - box[1]
    draw.multiline_text((s(cx) - w / 2 - box[0], s(cy) - h / 2 - box[1]), text, font=f, fill=color, align="center")
    return w / SS, h / SS


def draw_glyph(draw, rng, x0, y0, x1, y1, color):
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lw = max(1, int(s(rng.uniform(1.2, 3))))
    kind = rng.choice(["play", "chevrons", "bars", "rings", "grid", "poly", "db", "person", "bolt", "share", "lines", "gear"])
    if kind == "play":
        r = min(w, h) * 0.3
        ang = rng.choice([0, 0, 90, 180, 270])
        P = [(cx + r * math.cos(math.radians(ang + k * 120)), cy + r * math.sin(math.radians(ang + k * 120))) for k in range(3)]
        draw.polygon(pts_s(P), fill=color)
        if rng.random() < 0.5:
            draw.ellipse([s(cx - r * 1.5), s(cy - r * 1.5), s(cx + r * 1.5), s(cy + r * 1.5)], outline=color, width=lw)
    elif kind == "chevrons":
        r = min(w, h) * 0.22
        for k in range(rng.randint(1, 3)):
            ox = cx + (k - 1) * r * 0.9
            draw.line(pts_s([(ox - r / 2, cy - r), (ox + r / 2, cy), (ox - r / 2, cy + r)]), fill=color, width=lw, joint="curve")
    elif kind == "bars":
        n = rng.randint(3, 5)
        for k in range(n):
            bh = rng.uniform(0.2, 0.7) * h
            bx = x0 + w * (0.15 + 0.7 * k / n)
            draw.rectangle([s(bx), s(y1 - h * 0.15 - bh), s(bx + w * 0.5 / n), s(y1 - h * 0.15)], fill=color)
    elif kind == "rings":
        for k in range(rng.randint(1, 3)):
            r = min(w, h) * (0.15 + 0.12 * k)
            draw.ellipse([s(cx - r), s(cy - r), s(cx + r), s(cy + r)], outline=color, width=lw)
    elif kind == "grid":
        n = rng.randint(2, 3)
        g = min(w, h) * 0.6 / n
        for i in range(n):
            for j in range(n):
                gx, gy = cx - n * g / 2 + i * g, cy - n * g / 2 + j * g
                draw.rectangle([s(gx + 1), s(gy + 1), s(gx + g - 2), s(gy + g - 2)], outline=color, width=lw)
    elif kind == "poly":
        n = rng.choice([5, 6, 8])
        r = min(w, h) * 0.32
        P = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
        draw.polygon(pts_s(P), outline=color, width=lw)
    elif kind == "db":
        draw_cylinder(draw, cx - w * 0.25, cy - h * 0.3, cx + w * 0.25, cy + h * 0.3, None, color, lw / SS)
    elif kind == "person":
        r = min(w, h) * 0.14
        draw.ellipse([s(cx - r), s(cy - h * 0.3), s(cx + r), s(cy - h * 0.3 + 2 * r)], outline=color, width=lw)
        draw.arc([s(cx - w * 0.28), s(cy - h * 0.02), s(cx + w * 0.28), s(cy + h * 0.6)], 180, 360, fill=color, width=lw)
    elif kind == "bolt":
        P = [(cx + w * 0.08, cy - h * 0.35), (cx - w * 0.18, cy + h * 0.05), (cx, cy + h * 0.05), (cx - w * 0.08, cy + h * 0.35), (cx + w * 0.18, cy - h * 0.05), (cx, cy - h * 0.05)]
        draw.polygon(pts_s(P), fill=color)
    elif kind == "share":
        P = [(cx - w * 0.25, cy), (cx + w * 0.2, cy - h * 0.25), (cx + w * 0.2, cy + h * 0.25)]
        draw.line(pts_s([P[1], P[0], P[2]]), fill=color, width=lw)
        for p in P:
            r = min(w, h) * 0.09
            draw.ellipse([s(p[0] - r), s(p[1] - r), s(p[0] + r), s(p[1] + r)], fill=color)
    elif kind == "lines":
        for k in range(rng.randint(2, 4)):
            ly = y0 + h * (0.25 + 0.5 * k / 3)
            draw.line(pts_s([(x0 + w * 0.2, ly), (x1 - w * rng.uniform(0.2, 0.4), ly)]), fill=color, width=lw)
    elif kind == "gear":
        r = min(w, h) * 0.22
        draw.ellipse([s(cx - r), s(cy - r), s(cx + r), s(cy + r)], outline=color, width=lw)
        for k in range(8):
            a = 2 * math.pi * k / 8
            draw.line(pts_s([(cx + r * math.cos(a), cy + r * math.sin(a)), (cx + r * 1.45 * math.cos(a), cy + r * 1.45 * math.sin(a))]), fill=color, width=lw)


def draw_cylinder(draw, x0, y0, x1, y1, fill, outline, lw):
    w = max(1, int(s(lw)))
    eh = (y1 - y0) * 0.22
    draw.rectangle([s(x0), s(y0 + eh / 2), s(x1), s(y1 - eh / 2)], fill=fill)
    draw.ellipse([s(x0), s(y1 - eh), s(x1), s(y1)], fill=fill, outline=outline, width=w)
    if fill is not None:
        draw.rectangle([s(x0) + w, s(y0 + eh / 2), s(x1) - w, s(y1 - eh / 2)], fill=fill)
    draw.line([s(x0), s(y0 + eh / 2), s(x0), s(y1 - eh / 2)], fill=outline, width=w)
    draw.line([s(x1) - w + 1, s(y0 + eh / 2), s(x1) - w + 1, s(y1 - eh / 2)], fill=outline, width=w)
    draw.ellipse([s(x0), s(y0), s(x1), s(y0 + eh)], fill=fill, outline=outline, width=w)


def draw_node(draw, rng, box, theme):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    stroke, fg, bg = theme["stroke"], theme["fg"], theme["bg"]
    lw = rng.choice([1, 1, 1.5, 2, 2, 3])
    kind = rng.choices(
        ["rect", "round", "ellipse", "diamond", "cylinder", "icon", "icon", "circle_icon", "hexagon", "text", "person", "cloud", "pill"],
        weights=[14, 12, 5, 5, 6, 16, 8, 6, 4, 5, 4, 4, 5],
    )[0]
    fill = rng.choice([None, None, theme["node_fill"](rng), theme["node_fill"](rng)])
    outline = rng.choice([stroke, stroke, rng.choice(PALETTE)])
    text_color = fg
    if fill is not None and sum(fill) < 380:
        text_color = (255, 255, 255)
    iw = max(1, int(s(lw)))
    label = rand_text(rng)
    fs = rng.uniform(9, 17)

    if ICONS and rng.random() < 0.42:
        side = min(w, h * 0.75, rng.uniform(26, 96))
        ibox = paste_icon(theme["img"], rng, cx, y0 + 2, side)
        if ibox is not None:
            if rng.random() < 0.8:
                draw_text_center(draw, rng, cx, min(ibox[3] + fs * 0.9, y1 + 6), label, fs * 0.85, fg, max_w=w * 1.3)
            return ibox

    if kind in ("icon", "circle_icon", "person", "cloud"):
        side = min(w, h * 0.7, rng.uniform(34, 80))
        ix0, iy0 = cx - side / 2, y0 + 2
        ix1, iy1 = ix0 + side, iy0 + side
        col = rng.choice(PALETTE)
        if kind == "icon":
            if rng.random() < 0.6:
                draw.rounded_rectangle([s(ix0), s(iy0), s(ix1), s(iy1)], radius=s(rng.uniform(0, side * 0.2)), fill=col)
                draw_glyph(draw, rng, ix0, iy0, ix1, iy1, (255, 255, 255))
            else:
                draw.rounded_rectangle([s(ix0), s(iy0), s(ix1), s(iy1)], radius=s(rng.uniform(0, side * 0.2)), outline=col, width=iw)
                draw_glyph(draw, rng, ix0, iy0, ix1, iy1, col)
        elif kind == "circle_icon":
            if rng.random() < 0.5:
                draw.ellipse([s(ix0), s(iy0), s(ix1), s(iy1)], fill=col)
                draw_glyph(draw, rng, ix0, iy0, ix1, iy1, (255, 255, 255))
            else:
                draw.ellipse([s(ix0), s(iy0), s(ix1), s(iy1)], outline=col, width=iw)
                draw_glyph(draw, rng, ix0, iy0, ix1, iy1, col)
        elif kind == "person":
            r = side * 0.2
            pc = rng.choice([stroke, col])
            draw.ellipse([s(cx - r), s(iy0), s(cx + r), s(iy0 + 2 * r)], outline=pc, width=iw)
            draw.arc([s(ix0 + side * 0.1), s(iy0 + 2 * r + 2), s(ix1 - side * 0.1), s(iy1 + side * 0.5)], 180, 360, fill=pc, width=iw)
            draw.line([s(ix0 + side * 0.1), s(iy1), s(ix1 - side * 0.1), s(iy1)], fill=pc, width=iw)
        else:
            cc = rng.choice([stroke, col])
            for ex, ey, er in ((0.3, 0.6, 0.22), (0.5, 0.45, 0.28), (0.72, 0.6, 0.2)):
                draw.ellipse([s(ix0 + side * (ex - er)), s(iy0 + side * (ey - er)), s(ix0 + side * (ex + er)), s(iy0 + side * (ey + er))], fill=cc)
            draw.rectangle([s(ix0 + side * 0.3), s(iy0 + side * 0.6), s(ix0 + side * 0.72), s(iy0 + side * 0.8)], fill=cc)
        if rng.random() < 0.85:
            draw_text_center(draw, rng, cx, min(iy1 + fs * 0.9, y1 - 2), label, fs * 0.85, fg, max_w=w)
        return (ix0, iy0, ix1, iy1)

    if kind == "text":
        tw, th = draw_text_center(draw, rng, cx, cy, label, fs, fg, max_w=w)
        return (cx - tw / 2 - 2, cy - th / 2 - 2, cx + tw / 2 + 2, cy + th / 2 + 2)

    if kind == "rect":
        draw.rectangle([s(x0), s(y0), s(x1), s(y1)], fill=fill, outline=outline, width=iw)
        if rng.random() < 0.2:  # header strip
            draw.rectangle([s(x0), s(y0), s(x1), s(y0 + h * 0.25)], fill=outline)
    elif kind == "round":
        draw.rounded_rectangle([s(x0), s(y0), s(x1), s(y1)], radius=s(rng.uniform(4, 14)), fill=fill, outline=outline, width=iw)
    elif kind == "pill":
        draw.rounded_rectangle([s(x0), s(y0), s(x1), s(y1)], radius=s(min(w, h) / 2 - 0.5), fill=fill, outline=outline, width=iw)
    elif kind == "ellipse":
        draw.ellipse([s(x0), s(y0), s(x1), s(y1)], fill=fill, outline=outline, width=iw)
    elif kind == "diamond":
        draw.polygon(pts_s([(cx, y0), (x1, cy), (cx, y1), (x0, cy)]), fill=fill, outline=outline, width=iw)
    elif kind == "cylinder":
        draw_cylinder(draw, x0, y0, x1, y1, fill if fill is not None else bg, outline, lw)
    elif kind == "hexagon":
        k = w * 0.2
        draw.polygon(pts_s([(x0 + k, y0), (x1 - k, y0), (x1, cy), (x1 - k, y1), (x0 + k, y1), (x0, cy)]), fill=fill, outline=outline, width=iw)
    if rng.random() < 0.9:
        draw_text_center(draw, rng, cx, cy + (h * 0.08 if kind == "cylinder" else 0), label, fs, text_color, max_w=w * 0.9)
    return box


def anchor(box, side, rng, spread=0.3):
    x0, y0, x1, y1 = box
    t = 0.5 + rng.uniform(-spread, spread)
    if side == "l":
        return (x0, y0 + (y1 - y0) * t)
    if side == "r":
        return (x1, y0 + (y1 - y0) * t)
    if side == "t":
        return (x0 + (x1 - x0) * t, y0)
    return (x0 + (x1 - x0) * t, y1)


def clip_to_box(box, p_in, p_out):
    x0, y0, x1, y1 = box
    dx, dy = p_out[0] - p_in[0], p_out[1] - p_in[1]
    ts = []
    if dx:
        ts += [(x0 - p_in[0]) / dx, (x1 - p_in[0]) / dx]
    if dy:
        ts += [(y0 - p_in[1]) / dy, (y1 - p_in[1]) / dy]
    ts = [t for t in ts if t > 0]
    t = min(ts) if ts else 0
    return (p_in[0] + dx * min(t, 1), p_in[1] + dy * min(t, 1))


def offset(p, side, d):
    return {"l": (p[0] - d, p[1]), "r": (p[0] + d, p[1]), "t": (p[0], p[1] - d), "b": (p[0], p[1] + d)}[side]


def route(rng, A, B):
    ca = ((A[0] + A[2]) / 2, (A[1] + A[3]) / 2)
    cb = ((B[0] + B[2]) / 2, (B[1] + B[3]) / 2)
    dx, dy = cb[0] - ca[0], cb[1] - ca[1]
    gap_a, gap_b = rng.choice([0, 0, 2, 4, 8]), rng.choice([0, 0, 1, 3, 6])
    kind = rng.choices(["straight", "elbow", "elbow2", "curve", "aligned"], weights=[30, 30, 12, 10, 18])[0]
    horiz = abs(dx) >= abs(dy)

    if kind == "aligned":
        # axis-aligned straight connector between facing sides, if the boxes overlap on the other axis
        if horiz:
            lo, hi = max(A[1], B[1]), min(A[3], B[3])
            if hi - lo > 6:
                y = rng.uniform(lo + 2, hi - 2)
                sa, sb = ("r", "l") if dx > 0 else ("l", "r")
                pa = (A[2] if dx > 0 else A[0], y)
                pb = (B[0] if dx > 0 else B[2], y)
                return [offset(pa, sa, gap_a), offset(pb, sb, gap_b)], "straight"
        else:
            lo, hi = max(A[0], B[0]), min(A[2], B[2])
            if hi - lo > 6:
                x = rng.uniform(lo + 2, hi - 2)
                sa, sb = ("b", "t") if dy > 0 else ("t", "b")
                pa = (x, A[3] if dy > 0 else A[1])
                pb = (x, B[1] if dy > 0 else B[3])
                return [offset(pa, sa, gap_a), offset(pb, sb, gap_b)], "straight"
        kind = "elbow"

    if kind == "straight":
        pa, pb = clip_to_box(A, ca, cb), clip_to_box(B, cb, ca)
        d = math.dist(pa, pb) or 1
        ux, uy = (pb[0] - pa[0]) / d, (pb[1] - pa[1]) / d
        return [(pa[0] + ux * gap_a, pa[1] + uy * gap_a), (pb[0] - ux * gap_b, pb[1] - uy * gap_b)], "straight"

    if kind == "curve":
        pa, pb = clip_to_box(A, ca, cb), clip_to_box(B, cb, ca)
        d = math.dist(pa, pb) or 1
        nx, ny = -(pb[1] - pa[1]) / d, (pb[0] - pa[0]) / d
        k = rng.uniform(0.15, 0.45) * d * rng.choice([-1, 1])
        if rng.random() < 0.5:
            ctrl = ((pa[0] + pb[0]) / 2 + nx * k, (pa[1] + pb[1]) / 2 + ny * k)
            return bezier(pa, ctrl, pb), "curve"
        c1 = (pa[0] + (pb[0] - pa[0]) * 0.3 + nx * k, pa[1] + (pb[1] - pa[1]) * 0.3 + ny * k)
        c2 = (pa[0] + (pb[0] - pa[0]) * 0.7 + nx * k * rng.uniform(-0.5, 1), pa[1] + (pb[1] - pa[1]) * 0.7 + ny * k * rng.uniform(-0.5, 1))
        return bezier(pa, c1, c2, pb), "curve"

    if kind == "elbow":  # 3 segments, enters the facing side
        if horiz:
            sa, sb = ("r", "l") if dx > 0 else ("l", "r")
            pa, pb = offset(anchor(A, sa, rng), sa, gap_a), offset(anchor(B, sb, rng), sb, gap_b)
            if (pb[0] - pa[0]) * (1 if dx > 0 else -1) > 12:
                mx = pa[0] + (pb[0] - pa[0]) * rng.uniform(0.25, 0.75)
                pts = [pa, (mx, pa[1]), (mx, pb[1]), pb]
            else:
                kind = "elbow2"
        else:
            sa, sb = ("b", "t") if dy > 0 else ("t", "b")
            pa, pb = offset(anchor(A, sa, rng), sa, gap_a), offset(anchor(B, sb, rng), sb, gap_b)
            if (pb[1] - pa[1]) * (1 if dy > 0 else -1) > 12:
                my = pa[1] + (pb[1] - pa[1]) * rng.uniform(0.25, 0.75)
                pts = [pa, (pa[0], my), (pb[0], my), pb]
            else:
                kind = "elbow2"
        if kind == "elbow":
            if abs(pts[1][0] - pts[2][0]) + abs(pts[1][1] - pts[2][1]) < 3:
                pts = [pts[0], pts[3]]
            return pts, "elbow"

    # elbow2: single bend (L shape), leaves A horizontally and enters B vertically, or vice versa
    if rng.random() < 0.5:
        sa = "r" if dx > 0 else "l"
        sb = "t" if dy > 0 else "b"
        pa, pb = offset(anchor(A, sa, rng), sa, gap_a), offset(anchor(B, sb, rng, 0.2), sb, gap_b)
        pts = [pa, (pb[0], pa[1]), pb]
    else:
        sa = "b" if dy > 0 else "t"
        sb = "l" if dx > 0 else "r"
        pa, pb = offset(anchor(A, sa, rng, 0.2), sa, gap_a), offset(anchor(B, sb, rng), sb, gap_b)
        pts = [pa, (pa[0], pb[1]), pb]
    return pts, "elbow"


def make_theme(rng):
    r = rng.random()
    if r < 0.72:
        bg = (255, 255, 255)
    elif r < 0.84:
        bg = tuple(rng.randint(232, 252) for _ in range(3))
    else:
        bg = rng.choice([(0, 0, 0), (30, 30, 30), (44, 52, 64), (13, 17, 23), (20, 30, 50), (40, 44, 52)])
    dark = sum(bg) < 300
    fg = (235, 235, 235) if dark else rng.choice([(0, 0, 0), (30, 30, 30), (60, 60, 60), (35, 47, 62)])

    def node_fill(rg):
        if dark:
            return rg.choice([(0, 120, 200), (60, 60, 70), (0, 0, 0), (20, 90, 60), (255, 255, 255)])
        base = rg.choice(PALETTE)
        return lerp_color(base, (255, 255, 255), rg.uniform(0.55, 0.92)) if rg.random() < 0.8 else base

    return {"bg": bg, "dark": dark, "fg": fg, "stroke": fg, "node_fill": node_fill}


def edge_style(rng, theme, shared=None):
    if shared is not None and rng.random() < 0.7:
        st = dict(shared)
    else:
        if theme["dark"]:
            color = rng.choice([(255, 255, 255), (255, 255, 255), (200, 200, 200), (255, 200, 0), (0, 200, 255), (120, 220, 120)])
        else:
            color = rng.choices(
                [(0, 0, 0), (40, 40, 40), (90, 90, 90), (140, 140, 140), (170, 170, 170), rng.choice(PALETTE)],
                weights=[40, 15, 10, 8, 5, 22],
            )[0]
        if not theme["dark"] and rng.random() < 0.12:  # pale, low-contrast connectors
            color = lerp_color(color, theme["bg"], rng.uniform(0.35, 0.6))
        width = rng.choices([0.8, 1, 1.5, 2, 3, 4, 5, 6], weights=[8, 28, 14, 24, 12, 7, 4, 3])[0]
        L = rng.uniform(max(5, width * 2.6), width * 4.5 + 12)
        st = {
            "color": color, "width": width, "L": L, "ratio": rng.uniform(0.28, 0.6),
            "head": rng.choices(["tri", "stealth", "open", "hollow"], weights=[55, 17, 22, 6])[0],
            "dash": None,
        }
        if rng.random() < 0.22:
            st["dash"] = rng.choice([(width * 3, width * 2), (width * 4, width * 3), (width, width * 2), (8, 5), (3, 3)])
    return st


def draw_edge(draw, rng, pts, st, theme, double, headless, tail_deco, rounded):
    color, width, L = st["color"], st["width"], st["L"]
    hw = L * st["ratio"]
    full = list(pts)
    if rounded:
        full = round_corners(full, rng.uniform(4, 12))
    filled = st["head"] in ("tri", "stealth", "hollow")
    trim_end = (L * (0.6 if st["head"] == "stealth" else 0.9)) if (filled and not headless) else 0
    trim_start = trim_end if double else 0
    body = trim_path(full, trim_start, trim_end)
    if st["head"] == "open" and not headless:
        body = trim_path(full, width * 0.5 if double else 0, width * 0.5)
    draw_polyline(draw, body, color, width, st["dash"])
    if not headless:
        draw_head(draw, full[-1], head_dir(full, back=max(4, L * 0.8)), st["head"], L, hw, color, width, theme["bg"])
        if double:
            rev = full[::-1]
            draw_head(draw, rev[-1], head_dir(rev, back=max(4, L * 0.8)), st["head"], L, hw, color, width, theme["bg"])
    if tail_deco and not double:
        dx, dy = full[1][0] - full[0][0], full[1][1] - full[0][1]
        n = math.hypot(dx, dy) or 1
        draw_tail_deco(draw, full[0], (dx / n, dy / n), tail_deco, max(2.5, width * 1.6), color, theme["bg"])
    xs, ys = [p[0] for p in full], [p[1] for p in full]
    pad = max(width / 2, hw if not headless else 0)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def boxes_overlap(a, b, m=0):
    return not (a[2] + m < b[0] or b[2] + m < a[0] or a[3] + m < b[1] or b[3] + m < a[1])


def generate(seed):
    rng = random.Random(seed)
    W, H = rng.randint(640, 1800), rng.randint(420, 1300)
    theme = make_theme(rng)
    img = Image.new("RGB", (W * SS, H * SS), theme["bg"])
    draw = ImageDraw.Draw(img)
    theme["img"] = img

    # header / footer banners and big wordmarks (hard negatives: large glyphs look like connectors)
    if rng.random() < 0.1:
        bh = rng.uniform(0.05, 0.14) * H
        draw.rectangle([0, 0, s(W), s(bh)], fill=rng.choice(PALETTE))
    for _ in range(rng.choice([0, 0, 0, 1, 2])):
        f = font(rng, int(s(rng.uniform(26, 90))))
        col = rng.choice([theme["fg"], rng.choice(PALETTE), lerp_color(theme["fg"], theme["bg"], 0.5)])
        draw.text((s(rng.uniform(0, W * 0.7)), s(rng.uniform(0, H * 0.85))), rand_text(rng, 1), font=f, fill=col)

    # containers / swimlanes
    for _ in range(rng.choice([0, 0, 1, 2, 3, 4])):
        cw, ch = rng.uniform(0.25, 0.9) * W, rng.uniform(0.25, 0.9) * H
        cx0, cy0 = rng.uniform(5, W - cw - 5), rng.uniform(5, H - ch - 5)
        col = rng.choice(PALETTE)
        fill = None
        if rng.random() < 0.4:
            fill = lerp_color(col, theme["bg"], rng.uniform(0.82, 0.95))
        if fill:
            draw.rectangle([s(cx0), s(cy0), s(cx0 + cw), s(cy0 + ch)], fill=fill)
        rect = [(cx0, cy0), (cx0 + cw, cy0), (cx0 + cw, cy0 + ch), (cx0, cy0 + ch), (cx0, cy0)]
        draw_polyline(draw, rect, col if rng.random() < 0.7 else theme["stroke"], rng.choice([1, 1, 2]), rng.choice([None, (6, 4), (3, 3), (10, 5)]))
        if rng.random() < 0.7:
            f = font(rng, int(s(rng.uniform(10, 16))))
            draw.text((s(cx0 + 8), s(cy0 + 5)), rand_text(rng), font=f, fill=col)

    # sequence-diagram style lifelines (hard negatives: long dashed lines without heads)
    if rng.random() < 0.08:
        for _ in range(rng.randint(2, 5)):
            lx = rng.uniform(20, W - 20)
            draw_polyline(draw, [(lx, rng.uniform(0, H * 0.2)), (lx, rng.uniform(H * 0.7, H))], theme["stroke"], 1, (6, 4))

    # nodes on a jittered grid
    cols, rows = rng.randint(3, max(3, W // 135)), rng.randint(2, max(2, H // 115))
    cell_w, cell_h = W / cols, H / rows
    tight = rng.choice([1.0, 1.0, 1.0, 1.25, 1.4])  # >1: nodes almost touching, very short arrows
    cells = [(i, j) for i in range(cols) for j in range(rows)]
    rng.shuffle(cells)
    cells = cells[: max(3, int(len(cells) * rng.uniform(0.45, 0.95)))]
    nodes = []
    for i, j in cells:
        nw = rng.uniform(0.3, 0.62) * cell_w * tight
        nh = min(rng.uniform(0.3, 0.6) * cell_h * tight, nw * rng.uniform(0.45, 1.1))
        nw, nh = max(nw, 36), max(nh, 26)
        cx = (i + 0.5) * cell_w + rng.uniform(-0.12, 0.12) * cell_w
        cy = (j + 0.5) * cell_h + rng.uniform(-0.12, 0.12) * cell_h
        nodes.append((cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2))

    shapes = []
    for box in nodes:
        shapes.append(draw_node(draw, rng, box, theme))

    labels = []
    shared = edge_style(rng, theme)
    n_edges = int(len(nodes) * rng.uniform(0.6, 1.7))
    used = set()
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in shapes]
    for _ in range(n_edges):
        a = rng.randrange(len(nodes))
        order = sorted((k for k in range(len(nodes)) if k != a), key=lambda k: math.dist(centers[a], centers[k]))
        b = rng.choice(order[:3]) if rng.random() < 0.8 else rng.choice(order)
        if (a, b) in used:
            continue
        used.add((a, b))
        A, B = shapes[a], shapes[b]
        if boxes_overlap(A, B, 14):
            continue
        pts, kind = route(rng, A, B)
        if path_length(pts) < 9:
            continue
        st = edge_style(rng, theme, shared)
        if path_length(pts) < st["L"] * 1.8:
            st["L"] = max(5, path_length(pts) / 2.2)
        r = rng.random()
        headless = r < 0.08
        double = (not headless) and r < 0.18
        tail_deco = rng.choices([None, "dot", "ring", "diamond", "bar"], weights=[84, 6, 3, 4, 3])[0]
        rounded = kind == "elbow" and rng.random() < 0.3
        bbox = draw_edge(draw, rng, pts, st, theme, double, headless, tail_deco, rounded)

        # optional edge label / numbered badge
        if rng.random() < 0.3:
            mid = pts[len(pts) // 2] if len(pts) > 2 else ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
            off = rng.choice([(0, -9), (0, 10), (12, 0), (0, 0)])
            txt = rand_text(rng, rng.choice([1, 2])) if rng.random() < 0.75 else f"{rng.randint(1, 12)}. {rand_text(rng, 1)}"
            if rng.random() < 0.15:
                br = rng.uniform(8, 13)
                bx, by = mid[0] + off[0], mid[1] + off[1] * 2
                draw.rounded_rectangle([s(bx - br), s(by - br), s(bx + br), s(by + br)], radius=s(rng.uniform(2, br)), fill=rng.choice(PALETTE))
                draw_text_center(draw, rng, bx, by, str(rng.randint(1, 9)), br * 1.2, (255, 255, 255))
            else:
                if off == (0, 0):
                    f = font(rng, int(s(10)))
                    tw = draw.textlength(txt, font=f) / SS
                    draw.rectangle([s(mid[0] - tw / 2 - 2), s(mid[1] - 7), s(mid[0] + tw / 2 + 2), s(mid[1] + 7)], fill=theme["bg"])
                draw_text_center(draw, rng, mid[0] + off[0], mid[1] + off[1], txt, rng.uniform(8, 13), theme["fg"])

        if headless:
            continue
        tail, head = pts[0], pts[-1]
        cls = 0
        if double:
            cls = 1
            if (head[0], head[1]) < (tail[0], tail[1]):
                tail, head = head, tail
        labels.append((cls, bbox, tail, head))

    # free-floating arrows (not attached to nodes), e.g. legends and callouts
    for _ in range(rng.choice([0, 0, 0, 1, 2, 3])):
        st = edge_style(rng, theme, shared)
        length = rng.uniform(18, 160)
        ang = rng.choice([0, 90, 180, 270, rng.uniform(0, 360)])
        x, y = rng.uniform(20, W - 20), rng.uniform(20, H - 20)
        e = (x + length * math.cos(math.radians(ang)), y + length * math.sin(math.radians(ang)))
        if not (5 < e[0] < W - 5 and 5 < e[1] < H - 5):
            continue
        st["L"] = min(st["L"], length / 2.2)
        headless = rng.random() < 0.25
        bbox = draw_edge(draw, rng, [(x, y), e], st, theme, False, headless, None, False)
        if not headless:
            labels.append((0, bbox, (x, y), e))
        if rng.random() < 0.5:
            draw_text_center(draw, rng, e[0] + 40, e[1], rand_text(rng), 11, theme["fg"])

    img = img.resize((W, H), Image.LANCZOS)
    if rng.random() < 0.12:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    if rng.random() < 0.15:
        k = rng.uniform(0.5, 0.8)
        img = img.resize((int(W * k), int(H * k)), Image.BILINEAR).resize((W, H), Image.BICUBIC)

    lines = []
    for cls, (x0, y0, x1, y1), tail, head in labels:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            x0, x1 = min(x0, x1 - 2), max(x1, x0 + 2)
            y0, y1 = min(y0, y1 - 2), max(y1, y0 + 2)
        lines.append(
            f"{cls} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f} "
            f"{tail[0] / W:.6f} {tail[1] / H:.6f} 2 {head[0] / W:.6f} {head[1] / H:.6f} 2"
        )
    return img, lines, rng


def _work(args):
    seed, out_images, out_labels = args
    try:
        img, lines, rng = generate(seed)
    except ValueError:  # degenerate geometry in a random layout: skip this seed
        return 0
    stem = f"synth_{seed:06d}"
    if rng.random() < 0.3:
        img.save(Path(out_images) / f"{stem}.jpg", quality=rng.randint(45, 95))
    else:
        img.save(Path(out_images) / f"{stem}.png")
    (Path(out_labels) / f"{stem}.txt").write_text("\n".join(lines))
    return len(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--count", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    oi, ol = Path(a.out) / "images" / a.split, Path(a.out) / "labels" / a.split
    oi.mkdir(parents=True, exist_ok=True)
    ol.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers) as pool:
        n = sum(pool.imap_unordered(_work, [(a.seed + i, oi, ol) for i in range(a.count)], chunksize=8))
    print(f"{a.count} images, {n} arrows -> {oi}")
