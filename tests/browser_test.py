# Browser smoke test: renders the app in headless Chromium, exercises the main
# flows, and saves screenshots.  python3 tests/browser_test.py
import subprocess, time, sys, os
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8613

server = subprocess.Popen(
    [sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
    cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(1.0)

errors = []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
        page.wait_for_timeout(1200)
        page.screenshot(path="/tmp/shot_default.png")
        print("sidebar name:", page.text_content("#mName"))
        print("params table rows:", page.locator("#mParams tr").count())

        # switch samples: deepseek-v3 (T3, two stacks)
        page.select_option("#sampleSelect", "deepseek-v3")
        page.wait_for_timeout(900)
        page.screenshot(path="/tmp/shot_deepseek.png")
        print("deepseek name:", page.text_content("#mName"))
        anim_disabled = page.locator("#btnAnim").is_disabled()
        print("T3 anim disabled:", anim_disabled)

        # gpt2 (T0) + animation on
        page.select_option("#sampleSelect", "gpt2")
        page.wait_for_timeout(700)
        print("T0 anim disabled:", page.locator("#btnAnim").is_disabled())
        page.click("#btnAnim")
        page.wait_for_timeout(900)
        page.screenshot(path="/tmp/shot_gpt2_anim.png")
        page.click("#btnAnim")

        # expand all on gpt2 (T0 allows)
        page.click("#btnExpandAll")
        page.wait_for_timeout(800)
        page.screenshot(path="/tmp/shot_gpt2_expandall.png")

        # 2D view
        page.select_option("#sampleSelect", "qwen3-30b-a3b")
        page.wait_for_timeout(700)
        page.click("#btnView")
        page.wait_for_timeout(600)
        page.screenshot(path="/tmp/shot_2d.png", full_page=False)
        svg_count = page.locator("#view2d svg").count()
        print("2D svg present:", svg_count == 1)
        page.click("#btnView")
        page.wait_for_timeout(400)

        # paste-config channel (manual / non-open-source path)
        page.click("#btnPaste")
        page.select_option("#tplSelect", "qwen3-moe-2layer")
        page.click("#tplApply")
        page.fill("#cfgText", page.input_value("#cfgText").replace('"num_hidden_layers": 2', '"num_hidden_layers": 4'))
        page.click("#modalOk")
        page.wait_for_timeout(800)
        print("manual model name:", page.text_content("#mName"))
        page.screenshot(path="/tmp/shot_manual.png")

        # exports: llmarch + svg + glb + png via download events
        for kind, suffix in [("llmarch", ".llmarch"), ("svg", ".svg"), ("glb", ".glb"), ("png", ".png")]:
            page.click("#btnExport")
            with page.expect_download(timeout=25000) as dl:
                page.click(f'#exportMenu button[data-x="{kind}"]')
            path = dl.value.path()
            size = os.path.getsize(path)
            ok = dl.value.suggested_filename.endswith(suffix) and size > 200
            print(f"export {kind}: {dl.value.suggested_filename} {size}B {'OK' if ok else 'FAIL'}")
            if not ok:
                errors.append(f"export {kind} failed")
            page.wait_for_timeout(300)

        # click-to-expand: click the stack aggregate (center of canvas region)
        page.select_option("#sampleSelect", "qwen3-30b-a3b")
        page.wait_for_timeout(700)
        n_before = page.evaluate("document.querySelectorAll('#labels .lbl').length")
        page.screenshot(path="/tmp/shot_qwen_final.png")
        print("labels visible:", n_before)

        browser.close()
finally:
    server.terminate()

if errors:
    print("\nCONSOLE/PAGE ERRORS:")
    for e in errors[:20]:
        print(" -", e)
    sys.exit(1)
print("\nBROWSER SMOKE TEST PASSED")
