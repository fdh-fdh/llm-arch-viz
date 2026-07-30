# Expert-grid expansion + full-expand stress test.  python3 tests/browser_experts_test.py
import subprocess, time, sys, os
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8617
server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.0)
errors = []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{PORT}/?sample=qwen3-30b-a3b", wait_until="networkidle")
        page.wait_for_timeout(900)

        # expand expert grid on layer 0 via debug hook
        page.evaluate("__viz.applyAction({kind:'expandExperts', si:0, li:0}); __viz.camera.fit(__viz.state.layout.bounds)")
        page.wait_for_timeout(700)
        count = page.evaluate("__viz.state.layout.soa.count")
        print("instances with 128-expert grid:", count)
        page.screenshot(path="/tmp/shot_experts.png")

        # deepseek stress: expand every MoE layer + expert grid (worst case ~16k instances)
        page.select_option("#sampleSelect", "deepseek-v3")
        page.wait_for_timeout(900)
        t0 = time.time()
        page.evaluate("""
          const s = __viz.state;
          s.expanded.clear(); s.expandedExperts.clear(); s.lru = [];
          s.graph.stacks.forEach((stk, si) => {
            for (let li = 0; li < stk.count; li++) { s.expanded.add(si+':'+li); s.expandedExperts.add(si+':'+li); s.lru.push(si+':'+li); }
          });
          __viz.rebuild(true);
        """)
        page.wait_for_timeout(1200)
        dt = time.time() - t0
        count = page.evaluate("__viz.state.layout.soa.count")
        print(f"deepseek full expand: {count} instances, rebuild+render round-trip {dt:.2f}s")
        page.screenshot(path="/tmp/shot_deepseek_full.png")
        browser.close()
finally:
    server.terminate()
if errors:
    print("ERRORS:"); [print(" -", e) for e in errors[:10]]; sys.exit(1)
print("EXPERT/STRESS TEST PASSED")
