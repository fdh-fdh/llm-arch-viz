# v2 browser test: Inspector, focus mode, element cells, fly-to.
# python3 tests/browser_v2_test.py
import subprocess, time, sys, os
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8621
server = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
                          cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.0)
errors = []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1560, "height": 950})
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{PORT}/?sample=qwen3-30b-a3b", wait_until="networkidle")
        page.wait_for_timeout(1000)

        # 1. inspector default overview
        ins = page.inner_html("#inspector")
        ok = "架构总览" in ins and "data-kv" in ins
        print("overview inspector:", ok)
        if not ok: errors.append("overview inspector missing")
        page.screenshot(path="/tmp/v2_overview.png")

        # 2. pin q_proj via debug hook -> breadcrumb + formula + GQA svg + focus dim
        page.evaluate("""
          const items = __viz.state.layout.items;
          let q = null;
          for (let i = 0; i < __viz.state.layout.soa.count; i++) {
            if (items[i] && items[i].role === 'q_proj') { q = items[i]; break; }
          }
          __viz.pinItem(q);
          __viz.camera.flyTo({target:[q.x,q.y,q.z], dist: 18}, 100);
        """)
        page.wait_for_timeout(600)
        ins = page.inner_html("#inspector")
        checks = {
            "breadcrumb Layer 0": "Layer 0" in ins,
            "formula frac": "fm-frac" in ins,
            "gqa svg": "<svg" in ins,
            "kv slider": "data-kv" in ins,
            "pinned pin icon": page.text_content("#insPin").strip() == "📌",
            "focus active": page.evaluate("__viz.state.focus"),
        }
        for k, v in checks.items():
            print(f"{k}: {v}")
            if not v: errors.append(f"v2 check failed: {k}")
        page.screenshot(path="/tmp/v2_pinned_focus.png")

        # 3. zoom close to see element cells + hover for element tooltip
        page.evaluate("""
          const items = __viz.state.layout.items;
          let q = null;
          for (let i = 0; i < __viz.state.layout.soa.count; i++) {
            if (items[i] && items[i].role === 'q_proj') { q = items[i]; break; }
          }
          __viz.camera.flyTo({target:[q.x,q.y,q.z], dist: 6, pitch: 1.2}, 100);
        """)
        page.wait_for_timeout(500)
        page.mouse.move(760, 430)
        page.wait_for_timeout(300)
        page.mouse.move(770, 440)
        page.wait_for_timeout(400)
        tip = page.inner_html("#tooltip") if page.is_visible("#tooltip") else ""
        print("element tooltip W[:", "W[" in tip)
        if "W[" not in tip: errors.append("element tooltip missing (tip=" + tip[:120] + ")")
        page.screenshot(path="/tmp/v2_cells.png")

        # 4. breadcrumb model click clears pin; KV slider updates value
        page.evaluate("__viz.clearPin()")
        page.wait_for_timeout(200)
        page.eval_on_selector("#inspector input[data-kv]", "el => { el.value = 32768; el.dispatchEvent(new Event('input', {bubbles:true})); }")
        page.wait_for_timeout(200)
        kv = page.text_content("#kvBytesVal")
        print("kv updated:", kv)
        if not kv or kv.strip() == "": errors.append("kv slider not updating")

        # 5. inspector list click navigates (seg nav) — from overview click first component
        page.mouse.move(200, 500)   # leave canvas so hover state clears -> overview shows
        page.wait_for_timeout(300)
        page.click("#inspector .ins-li")
        page.wait_for_timeout(700)
        ins = page.inner_html("#inspector")
        print("seg nav pinned:", "Layer 0" in ins)
        if "Layer 0" not in ins: errors.append("seg nav failed")
        page.screenshot(path="/tmp/v2_segnav.png")

        # 6. gpt2 regression with new code paths + layernorm formula
        page.select_option("#sampleSelect", "gpt2")
        page.wait_for_timeout(800)
        page.evaluate("""
          const items = __viz.state.layout.items;
          let n = null;
          for (let i = 0; i < __viz.state.layout.soa.count; i++) {
            if (items[i] && /LayerNorm/.test(items[i].title || '')) { n = items[i]; break; }
          }
          __viz.pinItem(n);
        """)
        page.wait_for_timeout(300)
        ins = page.inner_html("#inspector")
        print("gpt2 layernorm knowledge:", "层归一化" in ins)
        if "层归一化" not in ins: errors.append("gpt2 layernorm knowledge missing")

        # 7. v2.2: LOD select toggles cellMode; gpt2 layer shows rails + γ/β strips
        page.select_option("#lodSelect", "0")
        page.wait_for_timeout(200)
        cm = page.evaluate("__viz.state.cellMode")
        print("lod off:", cm == 0)
        if cm != 0: errors.append("lodSelect off failed")
        page.select_option("#lodSelect", "2")
        page.wait_for_timeout(200)
        cm = page.evaluate("__viz.state.cellMode")
        print("lod boost:", cm == 2)
        if cm != 2: errors.append("lodSelect boost failed")
        page.select_option("#lodSelect", "1")
        n_strips = page.evaluate("""
          (() => { let g=0,b=0;
            for (let i=0;i<__viz.state.layout.soa.count;i++){
              const it=__viz.state.layout.items[i];
              if (it && it.role==='γ') g++; if (it && it.role==='β') b++; }
            return [g,b]; })()
        """)
        print("gpt2 γ/β strips:", n_strips)
        if n_strips[0] < 2 or n_strips[1] < 2: errors.append("γ/β strips missing")
        page.evaluate("__viz.clearPin(); __viz.camera.fit(__viz.state.layout.bounds)")
        page.wait_for_timeout(500)
        page.screenshot(path="/tmp/v22_gpt2_layer.png")

        # deepseek default cellMode = 0 (T3)
        page.select_option("#sampleSelect", "deepseek-v3")
        page.wait_for_timeout(800)
        cm = page.evaluate("__viz.state.cellMode")
        print("T3 default cells off:", cm == 0)
        if cm != 0: errors.append("T3 default cellMode wrong")

        browser.close()
finally:
    server.terminate()
if errors:
    print("\nERRORS:"); [print(" -", e) for e in errors[:12]]; sys.exit(1)
print("\nV2 BROWSER TEST PASSED")
