import { useEffect, useRef, useState } from "react";

// Velocity-based cursor control. Decoded class + confidence -> cursor velocity.
// left/right push the cursor horizontally; rest decays it toward stop. This is
// deliberately forgiving of noisy per-epoch decodes.

const SPEED = 6;          // px per update at full confidence
const CONF_GATE = 0.5;    // ignore low-confidence decodes
const FRICTION = 0.85;    // velocity decay per frame
const LATENCY_BUDGET_MS = 200;
const HISTORY_LEN = 40;

// Fixed class -> color mapping, reused everywhere (bars, cursor fill,
// history strip, legend) so the same hue always means the same class.
const CLASSES = ["left", "right", "rest"];
const CLASS_META = {
  left: { name: "Left", color: "#3987e5" },
  right: { name: "Right", color: "#d95926" },
  rest: { name: "Rest", color: "#199e70" },
};
const IDLE_COLOR = "#52514e"; // below-gate / no-decode-yet marker (not a class color)

const INK = {
  primary: "#ffffff",
  secondary: "#c3c2b7",
  muted: "#898781",
  gridline: "#2c2c2a",
  border: "rgba(255,255,255,0.10)",
  surface: "#1a1a19",
  page: "#0d0d0d",
};

const STATUS = {
  good: "#0ca30c",
  critical: "#d03b3b",
};

function ProbBar({ cls, value, isTop }) {
  const meta = CLASS_META[cls];
  const pct = Math.round(value * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
      <div style={{
        width: 46, fontSize: 13, color: INK.secondary,
        fontWeight: isTop ? 700 : 400,
      }}>
        {meta.name}
      </div>
      <div style={{
        position: "relative", flex: 1, height: 16,
        background: "rgba(255,255,255,0.06)", borderRadius: 4, overflow: "visible",
      }}>
        <div style={{
          position: "absolute", left: 0, top: 0, bottom: 0,
          width: `${Math.max(pct, 1.5)}%`,
          background: meta.color, borderRadius: "4px 2px 2px 4px",
          transition: "width 120ms linear",
        }} />
        {/* CONF_GATE threshold marker: decodes below this line don't move the cursor */}
        <div style={{
          position: "absolute", left: `${CONF_GATE * 100}%`, top: -3, bottom: -3,
          width: 1, background: INK.muted, opacity: 0.7,
        }} />
      </div>
      <div style={{
        width: 38, textAlign: "right", fontSize: 13,
        color: isTop ? INK.primary : INK.secondary,
        fontWeight: isTop ? 700 : 400,
        fontVariantNumeric: "tabular-nums",
      }}>
        {pct}%
      </div>
    </div>
  );
}

function Legend() {
  return (
    <div style={{ display: "flex", gap: 16, fontSize: 12, color: INK.secondary }}>
      {CLASSES.map((c) => (
        <div key={c} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{
            width: 9, height: 9, borderRadius: "50%",
            background: CLASS_META[c].color, display: "inline-block",
          }} />
          {CLASS_META[c].name}
        </div>
      ))}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{
          width: 9, height: 9, borderRadius: "50%",
          background: IDLE_COLOR, display: "inline-block",
        }} />
        Below {Math.round(CONF_GATE * 100)}% (ignored)
      </div>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState("connecting");
  const [decode, setDecode] = useState(null);
  const [latency, setLatency] = useState(null);
  const [history, setHistory] = useState([]);
  const pos = useRef({ x: 300, y: 200 });
  const vel = useRef({ x: 0, y: 0 });
  const cursorRef = useRef(null);

  // WebSocket -> latest decode
  useEffect(() => {
    const ws = new WebSocket(`ws://${location.hostname}:8080/ws`);
    ws.onopen = () => setStatus("connected");
    ws.onclose = () => setStatus("disconnected");
    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      setDecode(d);
      if (d.e2e_ms != null) setLatency(d.e2e_ms);
      setHistory((h) => [...h.slice(-(HISTORY_LEN - 1)), d]);
      if (d.confidence >= CONF_GATE) {
        if (d.label === "left") vel.current.x -= SPEED * d.confidence;
        if (d.label === "right") vel.current.x += SPEED * d.confidence;
      }
    };
    return () => ws.close();
  }, []);

  // Animation loop
  useEffect(() => {
    let raf;
    const tick = () => {
      vel.current.x *= FRICTION;
      vel.current.y *= FRICTION;
      pos.current.x = Math.max(0, Math.min(600, pos.current.x + vel.current.x));
      pos.current.y = Math.max(0, Math.min(400, pos.current.y + vel.current.y));
      if (cursorRef.current) {
        cursorRef.current.style.transform =
          `translate(${pos.current.x}px, ${pos.current.y}px)`;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const gated = decode && decode.confidence >= CONF_GATE;
  const cursorFill = gated ? CLASS_META[decode.label].color : IDLE_COLOR;
  const latencyOk = latency == null || latency <= LATENCY_BUDGET_MS;
  const latencyStatusColor = latency == null ? INK.muted : (latencyOk ? STATUS.good : STATUS.critical);
  const connColor = status === "connected" ? STATUS.good : status === "connecting" ? INK.muted : STATUS.critical;

  return (
    <div style={{
      fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
      background: INK.page, color: INK.primary, minHeight: "100vh",
      padding: 28,
    }}>
      <h1 style={{ margin: 0, fontSize: 22 }}>BCI Cursor Control</h1>
      <p style={{ color: INK.secondary, fontSize: 13, marginTop: 6, marginBottom: 20, maxWidth: 640 }}>
        Live motor-imagery EEG is windowed at 250&nbsp;Hz, bandpass-filtered, and
        classified by a CNN+Transformer decoder every ~100&nbsp;ms. Decodes below
        {" "}{Math.round(CONF_GATE * 100)}% confidence are shown but don't move the
        cursor &mdash; that's what keeps it from twitching on noise.
      </p>

      <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
          <span style={{
            width: 8, height: 8, borderRadius: "50%", background: connColor,
            display: "inline-block",
          }} />
          <span style={{ color: INK.secondary }}>Stream:</span>
          <b>{status}</b>
        </div>

        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={{ fontSize: 13, color: INK.secondary }}>Latency</span>
          <span style={{
            fontSize: 20, fontWeight: 600, fontVariantNumeric: "tabular-nums",
          }}>
            {latency != null ? latency.toFixed(0) : "—"}
            <span style={{ fontSize: 12, color: INK.muted }}>ms</span>
          </span>
          <span style={{
            fontSize: 11, color: latencyStatusColor, display: "flex",
            alignItems: "center", gap: 4,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: latencyStatusColor, display: "inline-block",
            }} />
            {latency == null ? "" : latencyOk ? `under ${LATENCY_BUDGET_MS}ms budget` : "over budget"}
          </span>
        </div>
      </div>

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
        {/* Cursor arena */}
        <div>
          <div style={{
            position: "relative", width: 620, height: 420,
            border: `1px solid ${INK.border}`, borderRadius: 8, background: INK.surface,
            overflow: "hidden",
          }}>
            {/* center hairline: the neutral / rest position */}
            <div style={{
              position: "absolute", left: "50%", top: 0, bottom: 0, width: 1,
              background: INK.gridline,
            }} />
            <div style={{
              position: "absolute", left: 16, top: 12, fontSize: 11, color: INK.muted,
            }}>
              ← LEFT
            </div>
            <div style={{
              position: "absolute", right: 16, top: 12, fontSize: 11, color: INK.muted,
            }}>
              RIGHT →
            </div>
            <div style={{
              position: "absolute", left: "50%", top: 12, transform: "translateX(-50%)",
              fontSize: 11, color: INK.muted,
            }}>
              REST
            </div>

            <div ref={cursorRef} style={{
              position: "absolute", width: 20, height: 20, borderRadius: "50%",
              background: cursorFill,
              boxShadow: `0 0 0 2px ${INK.surface}, 0 0 0 4px ${latencyStatusColor}, 0 0 14px ${cursorFill}`,
              transition: "background 120ms linear",
            }} />
          </div>
          <p style={{ color: INK.muted, fontSize: 12, marginTop: 8, maxWidth: 620 }}>
            Dot fill = current decode class (gray while below the confidence gate).
            Outer ring = latency status. Position drifts left/right with
            confidence-weighted pushes and decays back toward center via friction
            &mdash; it never jumps directly to a label.
          </p>
        </div>

        {/* Decode panel */}
        <div style={{ width: 300 }}>
          <h2 style={{ fontSize: 14, color: INK.secondary, fontWeight: 600, margin: "0 0 10px" }}>
            Live classification
          </h2>
          {CLASSES.map((c) => (
            <ProbBar
              key={c}
              cls={c}
              value={decode?.probs?.[c] ?? 0}
              isTop={decode?.label === c}
            />
          ))}
          <div style={{ marginTop: 6, marginBottom: 18 }}>
            <Legend />
          </div>

          <h2 style={{ fontSize: 14, color: INK.secondary, fontWeight: 600, margin: "0 0 8px" }}>
            Recent decodes
          </h2>
          <div style={{ display: "flex", gap: 2, height: 24, alignItems: "flex-end" }}>
            {history.length === 0 && (
              <span style={{ fontSize: 12, color: INK.muted }}>waiting for decodes…</span>
            )}
            {history.map((d, i) => {
              const above = d.confidence >= CONF_GATE;
              return (
                <div
                  key={i}
                  title={`${CLASS_META[d.label]?.name ?? d.label} · ${Math.round(d.confidence * 100)}%`}
                  style={{
                    width: 6, height: above ? 24 : 12,
                    borderRadius: 2,
                    background: above ? CLASS_META[d.label]?.color ?? IDLE_COLOR : IDLE_COLOR,
                    opacity: above ? 1 : 0.5,
                  }}
                />
              );
            })}
          </div>
          <p style={{ color: INK.muted, fontSize: 11, marginTop: 6 }}>
            Newest on the right. Short, dim ticks fell below the confidence gate.
          </p>
        </div>
      </div>
    </div>
  );
}
