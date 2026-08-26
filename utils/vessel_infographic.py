"""
Animated vessel-class comparison infographic.

Renders a blueprint-style, animated SVG/HTML panel comparing the five
tanker classes tracked by this dashboard (Panamax -> ULCC): relative hull
silhouette, length, deadweight tonnage, and typical cargo capacity in
barrels. Ships gently bob, entrance animates in a stagger, and the
headline numbers count up on load.

Self-contained HTML (no external assets) so it renders identically inside
Streamlit's sandboxed component iframe.
"""
from __future__ import annotations

VESSEL_CLASSES = [
    {"name": "Panamax", "length": "~230 meters", "dwt": "~70,000 DWT", "dwt_val": 70000,
     "barrels": 500000, "scale": 0.55},
    {"name": "Aframax", "length": "~245 meters", "dwt": "~110,000 DWT", "dwt_val": 110000,
     "barrels": 700000, "scale": 0.65},
    {"name": "Suezmax", "length": "~285 meters", "dwt": "~160,000 DWT", "dwt_val": 160000,
     "barrels": 1000000, "scale": 0.75},
    {"name": "VLCC", "length": "~330 meters", "dwt": "~300,000 DWT", "dwt_val": 300000,
     "barrels": 2000000, "scale": 0.88},
    {"name": "ULCC", "length": "~380 meters", "dwt": "~440,000 DWT", "dwt_val": 440000,
     "barrels": 3000000, "scale": 1.0},
]

_SHIP_SVG = """
<svg viewBox="0 0 420 150" class="ship-svg" preserveAspectRatio="xMidYMax meet">
  <g class="hull-group">
    <!-- hull -->
    <path class="hull" d="M20,105
             C20,95 30,80 55,78
             L340,78
             C360,78 375,84 392,100
             C398,106 398,112 392,115
             L28,115
             C22,113 20,110 20,105 Z" />
    <!-- deck line -->
    <line class="deck" x1="55" y1="78" x2="340" y2="78" />
    <!-- superstructure -->
    <rect class="bridge" x="300" y="32" width="55" height="46" rx="2" />
    <line class="bridge-window" x1="308" y1="46" x2="348" y2="46" />
    <line class="bridge-window" x1="308" y1="56" x2="348" y2="56" />
    <line class="bridge-window" x1="308" y1="66" x2="348" y2="66" />
    <!-- funnel -->
    <rect class="funnel" x="316" y="14" width="14" height="20" rx="1" />
    <!-- mast + antenna -->
    <line class="mast" x1="327" y1="4" x2="327" y2="14" />
    <line class="yard" x1="320" y1="8" x2="334" y2="8" />
    <!-- bow mast -->
    <line class="mast" x1="70" y1="60" x2="70" y2="78" />
    <line class="yard" x1="62" y1="64" x2="78" y2="64" />
    <!-- pipe deck detail -->
    <line class="deck-detail" x1="90" y1="78" x2="290" y2="78" stroke-dasharray="6 6" />
  </g>
  <!-- waterline -->
  <path class="waterline" d="M0,120 Q20,116 40,120 T80,120 T120,120 T160,120 T200,120 T240,120 T280,120 T320,120 T360,120 T400,120 T420,120" />
</svg>
"""

_BARREL_SVG = """
<svg viewBox="0 0 60 70" class="barrel-svg">
  <rect class="barrel-body" x="8" y="10" width="44" height="52" rx="6" />
  <ellipse class="barrel-cap" cx="30" cy="10" rx="22" ry="6" />
  <line class="barrel-band" x1="8" y1="24" x2="52" y2="24" />
  <line class="barrel-band" x1="8" y1="48" x2="52" y2="48" />
  <path class="barrel-drop" d="M30,30 C34,36 38,41 38,45 C38,49.4 34.4,53 30,53 C25.6,53 22,49.4 22,45 C22,41 26,36 30,30 Z" />
</svg>
"""


def _column_html(v: dict, index: int) -> str:
    delay = index * 0.15
    bob_delay = index * 0.35
    return f"""
    <div class="vc-col" style="animation-delay:{delay}s">
      <div class="vc-name">{v['name']}</div>
      <div class="vc-sub">{v['length']}</div>
      <div class="vc-sub vc-dwt" data-target="{v['dwt_val']}" data-suffix=" DWT">{v['dwt']}</div>
      <div class="ship-wrap" style="animation-delay:{bob_delay}s">
        <div class="ship-scale" style="transform:scale({v['scale']})">
          {_SHIP_SVG}
        </div>
      </div>
      <div class="barrel-wrap">
        {_BARREL_SVG}
      </div>
      <div class="vc-barrels">
        <span class="count-up" data-target="{v['barrels']}">0</span> barrels
      </div>
    </div>
    """


def render_vessel_class_infographic() -> str:
    """Return self-contained HTML for the animated vessel-class comparison."""
    columns = "\n".join(_column_html(v, i) for i, v in enumerate(VESSEL_CLASSES))

    return f"""
    <div class="vc-root">
      <meta charset="utf-8">
      <style>
        .vc-root {{
          background: radial-gradient(ellipse at top, #10203b 0%, #0b1220 70%);
          border-radius: 14px;
          padding: 28px 16px 20px 16px;
          font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          overflow: hidden;
        }}
        .vc-grid {{
          display: flex;
          justify-content: space-between;
          gap: 8px;
        }}
        .vc-col {{
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          opacity: 0;
          transform: translateY(18px);
          animation: vcFadeIn 0.7s ease-out forwards;
        }}
        @keyframes vcFadeIn {{
          to {{ opacity: 1; transform: translateY(0); }}
        }}
        .vc-name {{
          color: #ffffff;
          font-weight: 700;
          font-size: 17px;
          letter-spacing: 0.3px;
        }}
        .vc-sub {{
          color: #b9c2d0;
          font-size: 12.5px;
          margin-top: 2px;
        }}
        .vc-dwt {{ color: #8fa3c2; margin-bottom: 6px; }}
        .ship-wrap {{
          width: 100%;
          height: 100px;
          display: flex;
          align-items: flex-end;
          justify-content: center;
          animation: vcBob 3.6s ease-in-out infinite;
        }}
        @keyframes vcBob {{
          0%, 100% {{ transform: translateY(0px); }}
          50% {{ transform: translateY(-5px); }}
        }}
        .ship-scale {{ transform-origin: bottom center; }}
        .ship-svg {{ width: 165px; height: 88px; overflow: visible; }}
        .hull {{ fill: rgba(76,155,232,0.10); stroke: #eef3fb; stroke-width: 1.8; }}
        .deck, .bridge-window, .deck-detail, .yard, .mast {{
          stroke: #eef3fb; stroke-width: 1.3; fill: none;
        }}
        .bridge {{ fill: rgba(255,255,255,0.07); stroke: #eef3fb; stroke-width: 1.6; }}
        .funnel {{ fill: rgba(242,199,68,0.2); stroke: #f2c744; stroke-width: 1.5; }}
        .waterline {{
          fill: none; stroke: #3a5578; stroke-width: 1.4;
          stroke-dasharray: 4 5;
          animation: vcWater 5s linear infinite;
        }}
        @keyframes vcWater {{
          to {{ stroke-dashoffset: -60; }}
        }}
        .barrel-wrap {{ margin-top: 10px; }}
        .barrel-svg {{ width: 34px; height: 40px; }}
        .barrel-body {{ fill: rgba(242,199,68,0.06); stroke: #f2c744; stroke-width: 2; }}
        .barrel-cap {{ fill: rgba(242,199,68,0.12); stroke: #f2c744; stroke-width: 2; }}
        .barrel-band {{ stroke: #f2c744; stroke-width: 1.4; }}
        .barrel-drop {{
          fill: #f2c744;
          opacity: 0;
          animation: vcDrop 2.4s ease-in-out infinite;
        }}
        @keyframes vcDrop {{
          0% {{ opacity: 0; transform: translateY(-4px); }}
          30% {{ opacity: 1; transform: translateY(0px); }}
          70% {{ opacity: 1; }}
          100% {{ opacity: 0; transform: translateY(2px); }}
        }}
        .vc-barrels {{
          color: #f2c744;
          font-weight: 700;
          font-size: 13.5px;
          margin-top: 8px;
        }}
        .vc-caption {{
          text-align: center;
          color: #6b7a94;
          font-size: 11.5px;
          margin-top: 18px;
        }}
        @media (max-width: 900px) {{
          .vc-grid {{ flex-wrap: wrap; }}
          .vc-col {{ flex: 1 1 30%; margin-bottom: 20px; }}
        }}
        @media (prefers-reduced-motion: reduce) {{
          .vc-col {{ animation: none; opacity: 1; transform: none; }}
          .ship-wrap {{ animation: none; }}
          .waterline {{ animation: none; }}
          .barrel-drop {{ animation: none; opacity: 1; transform: none; }}
        }}
      </style>
      <div class="vc-grid">
        {columns}
      </div>
      <div class="vc-caption">Illustrative reference dimensions &amp; typical cargo capacity &mdash; not live telemetry</div>
    </div>
    <script>
      (function() {{
        var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        function easeOutCubic(t) {{ return 1 - Math.pow(1 - t, 3); }}
        function animateCount(el, target, duration) {{
          var suffix = el.dataset.suffix || "";
          if (reduceMotion) {{
            el.textContent = target.toLocaleString() + suffix;
            return;
          }}
          var start = null;
          function step(ts) {{
            if (!start) start = ts;
            var progress = Math.min((ts - start) / duration, 1);
            var eased = easeOutCubic(progress);
            var value = Math.floor(eased * target);
            el.textContent = value.toLocaleString() + suffix;
            if (progress < 1) {{
              window.requestAnimationFrame(step);
            }} else {{
              el.textContent = target.toLocaleString() + suffix;
            }}
          }}
          window.requestAnimationFrame(step);
        }}
        var counters = document.querySelectorAll('.count-up');
        counters.forEach(function(el, i) {{
          setTimeout(function() {{
            animateCount(el, parseInt(el.dataset.target, 10), 1100);
          }}, reduceMotion ? 0 : 200 + i * 150);
        }});
      }})();
    </script>
    """
