"""Compose the README hero: three app screenshots in phone frames."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "docs" / "screens"
BG = (7, 11, 16)


def phone(img: Image.Image, w=520) -> Image.Image:
    img = img.convert("RGB")
    h = int(img.height * w / img.width)
    img = img.resize((w, h), Image.LANCZOS)
    pad, r = 14, 58
    frame = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(frame)
    d.rounded_rectangle([0, 0, frame.width - 1, frame.height - 1], r + pad, fill=(28, 38, 48, 255))
    d.rounded_rectangle([3, 3, frame.width - 4, frame.height - 4], r + pad - 3, fill=(12, 17, 23, 255))
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], r, fill=255)
    frame.paste(img, (pad, pad), mask)
    return frame


def main():
    names = ["02_listening", "04_imbalance", "05_evidence", "06_history"]
    phones = [phone(Image.open(S / f"{n}.png")) for n in names]
    gap = 46
    W = sum(p.width for p in phones) + gap * (len(phones) + 1)
    H = max(p.height for p in phones) + 2 * gap
    canvas = Image.new("RGB", (W, H), BG)
    glow = Image.new("RGB", (W, H), BG)
    ImageDraw.Draw(glow).ellipse([W * 0.15, H * 0.05, W * 0.85, H * 0.9], fill=(14, 52, 52))
    canvas = Image.blend(canvas, glow.filter(ImageFilter.GaussianBlur(160)), 1.0)
    x = gap
    for i, p in enumerate(phones):
        y = gap + (22 if i % 2 else 0)
        shadow = Image.new("RGBA", p.size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle([0, 0, p.width, p.height], 70, fill=(0, 0, 0, 170))
        canvas.paste(shadow.filter(ImageFilter.GaussianBlur(18)), (x + 8, y + 18), shadow.filter(ImageFilter.GaussianBlur(18)))
        canvas.paste(p, (x, y), p)
        x += p.width + gap
    canvas = canvas.resize((W // 2, H // 2), Image.LANCZOS)
    canvas.save(ROOT / "docs" / "showcase.png", optimize=True)
    print("wrote docs/showcase.png", canvas.size)


if __name__ == "__main__":
    main()
