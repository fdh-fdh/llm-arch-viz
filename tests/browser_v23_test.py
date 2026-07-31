# v2.3 browser test: guided tour, tensor search, 2D inspector sync, safetensors drop.
# python3 tests/browser_v23_test.py
import subprocess, time, sys, os, json, struct
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8641
server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.0)
errors = []

# synthetic safetensors header file for drop test
st_json = json.dumps({
    "model.embed_tokens.weight": {"dtype": "BF16", "shape": [1000, 256], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [512, 256], "data_offsets": [0, 0]},
    "model.layers.0.self_attn.k_proj.weight": {"dtype": "BF16", "shape": [256, 256], "data_offsets": [0, 0]},
    "model.layers.0.mlp.gate_proj.weight": {"dtype": "BF16", "shape": [1024, 256], "data_offsets": [0, 0]},
    "model.layers.1.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [512, 256], "data_offsets": [0, 0]},
}).encode()
st_path = "/tmp/tiny_test.safetensors"
with open(st_path, "wb") as f:
    f.write(struct.pack("<Q", len(st_json)) + st_json)

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1560, "height": 950})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{PORT}/?sample=qwen3-30b-a3b", wait_until="networkidle")
        page.wait_for_timeout(1000)

        # 1. guided tour
        page.click("#btnTour")
        page.wait_for_timeout(700)
        ok = not page.locator("#tourBar").is_hidden()
        print("tour bar visible:", ok)
        if not ok: errors.append("tour bar hidden")
        print("station 1:", page.text_content("#tourLabel"))
        # advance through several stations via button + keyboard
        for _ in range(4):
            page.click("#tourNext")
            page.wait_for_timeout(500)
        lbl4 = page.text_content("#tourLabel")
        step = page.text_content("#tourStep")
        print("after 4x next:", step, lbl4)
        if step.split("/")[0] != "5": errors.append("tour step wrong: " + step)
        ins = page.inner_html("#inspector")
        if "crumb" not in ins: errors.append("tour station not pinned in inspector")
        page.screenshot(path="/tmp/v23_tour.png")
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(400)
        if page.text_content("#tourStep").split("/")[0] != "6": errors.append("keyboard nav failed")
        # run to completion
        n = int(page.text_content("#tourStep").split("/")[1])
        for _ in range(n):
            page.keyboard.press("ArrowRight")
            page.wait_for_timeout(120)
        done = page.locator("#tourBar").is_hidden()
        print("tour completes and closes:", done)
        if not done: errors.append("tour did not close at end")

        # 2. tensor search
        page.fill("#searchInput", "37 k_proj")
        page.wait_for_timeout(300)
        nres = page.locator(".sr-item").count()
        print("search results for '37 k_proj':", nres)
        if nres < 1: errors.append("search no results")
        page.click(".sr-item")
        page.wait_for_timeout(800)
        ins = page.inner_html("#inspector")
        ok = "Layer 37" in ins and "k_proj" in ins
        print("search jump pinned layer 37 k_proj:", ok)
        if not ok: errors.append("search jump failed")
        page.screenshot(path="/tmp/v23_search.png")

        # 3. 2D view click -> inspector sync
        page.evaluate("__viz.clearPin()")
        page.click("#btnView")
        page.wait_for_timeout(600)
        page.click('#view2d g[data-nav^="seg:0:0:attn"]')
        page.wait_for_timeout(500)
        ins = page.inner_html("#inspector")
        ok = "Self-Attention" in ins and "fm-frac" in ins
        print("2D click pins attention in inspector:", ok)
        if not ok: errors.append("2D nav failed")
        page.screenshot(path="/tmp/v23_2d.png")
        page.click("#btnView")
        page.wait_for_timeout(300)

        # 4. safetensors file import (header-only, local)
        page.set_input_files("#fileInput", st_path)
        page.wait_for_timeout(900)
        name = page.text_content("#mName")
        print("safetensors import model:", name)
        if "tiny_test" not in name: errors.append("safetensors import failed: " + name)
        total = page.evaluate("__viz.state.graph.meta.params.total")
        print("inferred total params:", total)
        if total <= 0: errors.append("inferred params zero")

        browser.close()
finally:
    server.terminate()
if errors:
    print("\nERRORS:"); [print(" -", e) for e in errors[:12]]; sys.exit(1)
print("\nV2.3 BROWSER TEST PASSED")
