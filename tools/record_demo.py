"""Record the captioned README demo video + stills by driving the real app.

    (cd web && npx vite --port 5173)      # in another terminal
    .venv/bin/python tools/record_demo.py

Frames are timestamped screenshots (crisper than browser video capture);
ffmpeg's concat demuxer replays them with their real durations, and listening
segments are fast-forwarded. Captions are an overlay injected into the page.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:5173/"
TMP = ROOT / "tools" / ".frames"
DOCS = ROOT / "docs"
SHOTS = DOCS / "screens"

CAPTION_JS = """t => {
  let d = document.getElementById('__cap');
  if (!d) {
    d = document.createElement('div'); d.id = '__cap';
    d.style.cssText = 'position:fixed;left:14px;right:14px;bottom:84px;padding:11px 14px;border-radius:14px;' +
      'background:rgba(3,6,9,.86);border:1px solid rgba(25,227,196,.35);color:#fff;font:600 15px/1.35 -apple-system,system-ui,sans-serif;' +
      'text-align:center;z-index:2147483647;box-shadow:0 8px 30px rgba(0,0,0,.5);transition:opacity .25s';
    document.body.appendChild(d);
  }
  d.style.display = t ? 'block' : 'none'; d.innerHTML = t;
}"""

CARD_JS = """([big, small]) => {
  let d = document.getElementById('__card');
  if (!d) { d = document.createElement('div'); d.id = '__card'; document.body.appendChild(d); }
  d.style.cssText = 'position:fixed;inset:0;z-index:2147483646;display:grid;place-content:center;text-align:center;gap:14px;' +
    'background:radial-gradient(120% 80% at 50% 30%,#0f2a2b,#070b10 65%);color:#e6eef5;font-family:-apple-system,system-ui,sans-serif';
  d.innerHTML = big ? `<div style="font-size:44px;font-weight:800;line-height:1.1;letter-spacing:.3px">${big}</div>` +
    `<div style="font-size:16px;font-weight:500;line-height:1.55;color:#8796a5;max-width:330px;margin:auto">${small}</div>` : '';
  if (!big) d.remove();
}"""


class Recorder:
    def __init__(self, page):
        self.page, self.frames, self.n = page, [], 0
        self.speed = 1.0
        self.t_video = 0.0
        self.t_last = time.time()

    def snap(self):
        now = time.time()
        if self.frames:
            self.t_video += (now - self.t_last) / self.speed
        self.t_last = now
        p = TMP / f"{self.n:05d}.jpg"
        self.page.screenshot(path=str(p), type="jpeg", quality=88)
        self.frames.append((p, self.t_video))
        self.n += 1

    def hold(self, sec, speed=1.0):
        self.speed = speed
        end = time.time() + sec
        while time.time() < end:
            self.snap()
        self.speed = 1.0

    def until(self, js, timeout=30, speed=1.0):
        self.speed = speed
        end = time.time() + timeout
        while time.time() < end and not self.page.evaluate(js):
            self.snap()
        self.snap()
        self.speed = 1.0

    def cap(self, text):
        self.page.evaluate(CAPTION_JS, text)

    def card(self, big, small=""):
        self.page.evaluate(CARD_JS, [big, small])

    def still(self, name):
        SHOTS.mkdir(parents=True, exist_ok=True)
        self.cap("")
        self.page.screenshot(path=str(SHOTS / f"{name}.png"))


def run_check(r, page, cond, label, caption_listen, caption_result, scroll=False):
    page.click(f"#cond-{cond}")
    r.cap(f"Condition: <b>{label}</b>")
    r.hold(1.0)
    page.click("#listen")
    r.cap(caption_listen)
    r.until("!!document.querySelector('.result')", timeout=20, speed=2.5)
    r.cap(caption_result)
    r.hold(3.4)
    if scroll:
        page.evaluate("document.querySelector('.card:nth-of-type(2)')?.scrollIntoView({behavior:'smooth'})")
        r.hold(3.2)
        page.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
        r.hold(0.8)


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True,
                              args=["--autoplay-policy=no-user-gesture-required", "--use-angle=metal",
                                    "--enable-unsafe-swiftshader"])
        ctx = b.new_context(viewport={"width": 400, "height": 820}, device_scale_factor=2,
                            is_mobile=True, has_touch=True)
        page = ctx.new_page()
        page.goto(URL)
        page.wait_for_selector("#demo")
        r = Recorder(page)

        r.card("Stetho", "A stethoscope for machines.<br>Hold your phone near a machine; in 10 seconds it tells you what's wrong.")
        r.hold(3.0)
        r.card("")
        r.cap("Predictive maintenance on a phone. The ML runs entirely in the browser.")
        r.hold(2.6)
        r.still("01_home")
        page.click("#demo")
        r.cap("Demo rig: a table fan. Step 1: record it while it is <b>healthy</b>.")
        r.hold(2.2)
        page.click("#listen")
        r.cap("Learning this fan's normal sound… (sped up)")
        r.hold(3.0, speed=2.5)
        r.still("02_listening")
        r.until("document.querySelector('.hint')?.innerText.startsWith('Baseline saved')", timeout=20, speed=2.5)
        rpm = page.inner_text(".hint b")
        r.cap(f"From sound alone: shaft speed <b>{rpm}</b> and blade count, found with a harmonic search.")
        r.hold(3.4)

        run_check(r, page, "healthy", "normal fan", "Listening: a live 3D spectrogram waterfall (Three.js)",
                  "Normal fan → <b>healthy</b>. The autoencoder error matches the baseline.")
        r.still("03_healthy")
        page.click("#again")
        run_check(r, page, "imbalance", "coin taped to one blade", "Listening… (sped up)",
                  "Coin on a blade → <b>imbalance</b>. The 1× shaft modulation jumped about +16 dB.", scroll=True)
        r.still("04_imbalance")
        page.evaluate("document.querySelector('.card:nth-of-type(2)')?.scrollIntoView()")
        r.still("05_evidence")
        page.evaluate("window.scrollTo(0,0)")
        page.click("#again")
        run_check(r, page, "looseness", "loosened mounting screw", "Listening… (sped up)",
                  "Loose screw → <b>looseness</b>. Harmonics and ½-orders appeared (rattle every other turn).")
        page.click("#again")
        run_check(r, page, "blocked", "airflow partly blocked", "Listening… (sped up)",
                  "Blocked airflow → <b>flagged</b>. The blade-pass tone drowned in turbulence.")
        page.click("#again")
        run_check(r, page, "bearing", "worn bearing", "Listening… (sped up)",
                  "Worn bearing → <b>bearing wear</b>. Impacts at a non-synchronous 3.57× order.")
        page.click("#tab-history")
        r.cap("Every check stays on the device: a health timeline for each machine.")
        r.hold(3.6)
        r.still("06_history")
        r.cap("Report written only from measured numbers, then verified. No invented faults.")
        r.hold(2.6)
        r.cap("")
        r.card("Made by Amartya", "Amartya Vikram Singh<br>github.com/amartya1523/Stetho")
        r.hold(3.5)
        b.close()

    lst = TMP / "list.txt"
    with open(lst, "w") as f:
        for i, (fp, t) in enumerate(r.frames):
            f.write(f"file '{fp}'\n")
            nxt = r.frames[i + 1][1] if i + 1 < len(r.frames) else t + 0.5
            f.write(f"duration {max(nxt - t, 0.02):.3f}\n")
        f.write(f"file '{r.frames[-1][0]}'\n")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = DOCS / "demo.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", "fps=30,scale=800:-2:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-crf", "27",
                    "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
    gif = DOCS / "demo.gif"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                    "fps=10,scale=340:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", str(gif)], check=True)
    print(f"{len(r.frames)} frames, video {r.t_video:.1f}s -> {mp4.stat().st_size/1e6:.1f} MB mp4, "
          f"{gif.stat().st_size/1e6:.1f} MB gif")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
