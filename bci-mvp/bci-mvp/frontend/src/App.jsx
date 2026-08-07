import { useEffect, useRef, useState } from "react";

// Velocity-based cursor control. Decoded class + confidence -> cursor velocity.
// left/right push the cursor horizontally; rest decays it toward stop. This is
// deliberately forgiving of noisy per-epoch decodes.

const SPEED = 6;          // px per update at full confidence
const CONF_GATE = 0.5;    // ignore low-confidence decodes
const FRICTION = 0.85;    // velocity decay per frame
const LATENCY_BUDGET_MS = 200;
const HISTORY_LEN = 40;
const PAGE_MAX_WIDTH = 1040;

// Fixed class -> color mapping, reused everywhere (bars, cursor fill,
// history strip, legend) so the same hue always means the same class.
const CLASSES = ["left", "right", "rest"];
const CLASS_META = {
  left: { name: "Left", verb: "imagining a LEFT-hand movement", color: "#3987e5" },
  right: { name: "Right", verb: "imagining a RIGHT-hand movement", color: "#d95926" },
  rest: { name: "Rest", verb: "at REST / relaxed", color: "#199e70" },
};
const IDLE_COLOR = "#52514e"; // below-gate / no-decode-yet marker (not a class color)

const INK = {
  primary: "#ffffff",
  secondary: "#c3c2b7",
  muted: "#898781",
  gridline: "#2c2c2a",
  border: "rgba(255,255,255,0.10)",
  surface: "#181817",
  surfaceRaised: "#1e1e1d",
  page: "#0b0b0b",
};

const STATUS = {
  good: "#0ca30c",
  warning: "#fab219",
  critical: "#d03b3b",
};

const GATE_PCT = Math.round(CONF_GATE * 100);
const CHANCE_PCT = Math.round(100 / CLASSES.length);

const CARD = {
  background: INK.surfaceRaised,
  border: `1px solid ${INK.border}`,
  borderRadius: 12,
  padding: 20,
};

function InfoBanner() {
  return (
    <div style={{
      display: "flex", gap: 10, alignItems: "flex-start",
      background: "rgba(250,178,25,0.08)", border: "1px solid rgba(250,178,25,0.35)",
      borderRadius: 12, padding: "12px 16px",
    }}>
      <span style={{ fontSize: 14, color: STATUS.warning, lineHeight: "18px" }}>ⓘ</span>
      <p style={{ margin: 0, fontSize: 13, color: INK.secondary, lineHeight: 1.6 }}>
        <b style={{ color: INK.primary }}>This is simulated data, not a real brain signal.</b>{" "}
        Nobody is wearing a headset right now &mdash; a script is generating fake EEG
        that automatically cycles Left&nbsp;→&nbsp;Right&nbsp;→&nbsp;Rest every 4 seconds,
        so the full pipeline can be tested without hardware. With a real EEG cap, you'd
        replace that script with an actual person imagining a left-hand movement,
        a right-hand movement, or relaxing &mdash; everything downstream (filtering,
        the model, the cursor) works identically either way.
      </p>
    </div>
  );
}

function PipelineStep({ n, title, desc, footer, last }) {
  return (
    <>
      <div style={{
        background: INK.surface, border: `1px solid ${INK.border}`, borderRadius: 10,
        padding: "12px 14px", width: 172, flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 6 }}>
          <span style={{
            width: 18, height: 18, borderRadius: "50%", background: INK.gridline,
            color: INK.secondary, fontSize: 10, fontWeight: 700,
            display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
          }}>
            {n}
          </span>
          <span style={{ fontSize: 12, fontWeight: 700 }}>{title}</span>
        </div>
        <p style={{ margin: 0, fontSize: 11, color: INK.muted, lineHeight: 1.45, minHeight: 48 }}>
          {desc}
        </p>
        {footer && (
          <div style={{
            marginTop: 8, paddingTop: 8, borderTop: `1px solid ${INK.border}`,
            fontSize: 11, color: INK.secondary, fontWeight: 600,
          }}>
            {footer}
          </div>
        )}
      </div>
      {!last && (
        <div style={{ alignSelf: "center", color: INK.muted, fontSize: 16, flexShrink: 0 }}>
          →
        </div>
      )}
    </>
  );
}

function ProbBar({ cls, value, isTop }) {
  const meta = CLASS_META[cls];
  const pct = Math.round(value * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
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
    <div style={{ display: "flex", gap: 16, fontSize: 12, color: INK.secondary, flexWrap: "wrap" }}>
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
        Below {GATE_PCT}% (ignored)
      </div>
    </div>
  );
}

function SectionHeading({ children, sub, eyebrow }) {
  return (
    <div style={{ marginBottom: 14 }}>
      {eyebrow && (
        <div style={{
          fontSize: 10, color: INK.muted, textTransform: "uppercase",
          letterSpacing: 0.6, fontWeight: 700, marginBottom: 3,
        }}>
          {eyebrow}
        </div>
      )}
      <h2 style={{ fontSize: 15, color: INK.primary, fontWeight: 700, margin: 0 }}>
        {children}
      </h2>
      {sub && <p style={{ margin: "4px 0 0", fontSize: 12, color: INK.muted, lineHeight: 1.5 }}>{sub}</p>}
    </div>
  );
}

function Glossary() {
  const items = [
    ["EEG (electroencephalography)", "Electrodes on the scalp pick up the tiny voltage changes produced when neurons fire. It's noisy and indirect compared to something like an implant, but it's fast and needs no surgery."],
    ["Motor imagery", "Vividly imagining a movement — without actually moving — measurably changes rhythms over the motor cortex, roughly the same way actually moving would. That's the real-world signal a headset would pick up in place of this demo's simulated cycle."],
    ["Confidence / softmax", "The model never outputs a flat \"yes\" — it outputs a probability for each of the 3 states that sums to 100%. Step 4 only trusts that guess once one state pulls far enough ahead of the others."],
    ["Why push instead of snap", `Any single ~1s window is noisy, so the model's raw accuracy is well under 100%. Treating each decode as a gentle, confidence-weighted push — rather than teleporting the cursor to a label — lets correct guesses accumulate while wrong ones wash out.`],
  ];
  return (
    <details style={{ ...CARD, padding: 0 }}>
      <summary style={{
        cursor: "pointer", padding: 18, fontSize: 14, fontWeight: 700,
        listStyle: "none", display: "flex", alignItems: "center", gap: 8,
      }}>
        <span style={{ color: INK.muted, fontSize: 12 }}>▸</span>
        Glossary &mdash; what these terms mean
      </summary>
      <div style={{ padding: "0 18px 18px", display: "grid", gap: 14 }}>
        {items.map(([term, def]) => (
          <div key={term}>
            <div style={{ fontSize: 13, fontWeight: 700, color: INK.primary, marginBottom: 2 }}>{term}</div>
            <div style={{ fontSize: 12, color: INK.muted, lineHeight: 1.55 }}>{def}</div>
          </div>
        ))}
      </div>
    </details>
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
  const topPct = decode ? Math.round(decode.confidence * 100) : null;

  let narrative;
  if (!decode) {
    narrative = "Waiting for the first decode…";
  } else if (gated) {
    const dir = decode.label === "rest"
      ? "so nothing pushes the cursor — it drifts back toward the center"
      : `so the cursor is being pushed ${decode.label}`;
    narrative = (
      <>
        The model is <b>{topPct}%</b> sure the current window looks like{" "}
        <b style={{ color: CLASS_META[decode.label].color }}>{CLASS_META[decode.label].verb}</b>.
        That clears the {GATE_PCT}% bar, {dir}.
      </>
    );
  } else {
    narrative = (
      <>
        The model's best guess is{" "}
        <b style={{ color: CLASS_META[decode.label].color }}>{CLASS_META[decode.label].verb}</b>,
        but only at <b>{topPct}%</b> &mdash; below the {GATE_PCT}% bar (barely better than a{" "}
        {CHANCE_PCT}% three-way coin flip), so it's ignored and the cursor drifts back toward center.
      </>
    );
  }

  return (
    <div style={{
      fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
      background: `radial-gradient(ellipse 900px 500px at 50% -10%, #171716 0%, ${INK.page} 60%)`,
      color: INK.primary, minHeight: "100vh",
    }}>
      <style>{`
        @keyframes bci-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
        details > summary::-webkit-details-marker { display: none; }
        details[open] summary span:first-child { transform: rotate(90deg); }
        details summary span:first-child { display: inline-block; transition: transform 120ms ease; }
      `}</style>

      <div style={{ maxWidth: PAGE_MAX_WIDTH, margin: "0 auto", padding: "36px 28px 60px" }}>

        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <span style={{ fontSize: 22 }}>🧠</span>
          <h1 style={{ margin: 0, fontSize: 24 }}>BCI Cursor Control</h1>
          <span style={{
            fontSize: 10, fontWeight: 700, color: INK.muted, border: `1px solid ${INK.border}`,
            borderRadius: 999, padding: "3px 9px", textTransform: "uppercase", letterSpacing: 0.5,
          }}>
            live demo
          </span>
        </div>
        <p style={{ color: INK.secondary, fontSize: 13, marginTop: 0, marginBottom: 22, lineHeight: 1.6, maxWidth: 760 }}>
          A brain&ndash;computer interface reads electrical activity from the scalp (EEG),
          guesses what movement someone is imagining, and turns that guess into a
          cursor movement. This page shows every stage of that pipeline live, in plain
          language, so you can see exactly what the system is doing and why &mdash; not
          just a dot moving around.
        </p>

        <div style={{ display: "grid", gap: 20 }}>
          <InfoBanner />

          {/* Pipeline overview */}
          <div style={CARD}>
            <SectionHeading eyebrow="How it works, end to end">The pipeline, right now</SectionHeading>
            <div style={{ display: "flex", alignItems: "stretch", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
              <PipelineStep
                n={1} title="EEG stream"
                desc="22 electrodes, 250 samples/sec each, arriving continuously over the network."
                footer={
                  <span style={{ display: "flex", alignItems: "center", gap: 6, color: connColor }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: "50%", background: connColor,
                      animation: status === "connected" ? "bci-pulse 1.6s ease-in-out infinite" : "none",
                    }} />
                    {status}
                  </span>
                }
              />
              <PipelineStep
                n={2} title="Filter + window"
                desc="Keep the 8–30Hz band that real motor-imagery rhythms live in; take the last ~1s; normalize each channel."
                footer="always running"
              />
              <PipelineStep
                n={3} title="Model guesses"
                desc="A CNN+Transformer scores how much that 1s window resembles each of the 3 states."
                footer={decode ? `top guess: ${CLASS_META[decode.label].name} (${topPct}%)` : "—"}
              />
              <PipelineStep
                n={4} title="Confidence check"
                desc={`Only act on the guess if it's ≥${GATE_PCT}% sure. Below that, treat it as noise.`}
                footer={decode ? (gated
                  ? <span style={{ color: STATUS.good }}>passed → act</span>
                  : <span style={{ color: INK.muted }}>failed → ignored</span>) : "—"}
              />
              <PipelineStep
                n={5} title="Move cursor" last
                desc="A pass pushes the cursor toward that side; a fail (or Rest) lets it drift back to center."
                footer={gated
                  ? <span style={{ color: CLASS_META[decode.label].color }}>{decode.label === "rest" ? "holding" : `pushing ${decode.label}`}</span>
                  : <span style={{ color: INK.muted }}>drifting to center</span>}
              />
            </div>
          </div>

          {/* Plain-English status line */}
          <div style={CARD}>
            <div style={{
              color: INK.muted, fontSize: 10, textTransform: "uppercase",
              letterSpacing: 0.6, fontWeight: 700, marginBottom: 8,
            }}>
              Right now, in plain English
            </div>
            <div style={{ fontSize: 15, lineHeight: 1.6 }}>{narrative}</div>
          </div>

          {/* Status row: connection + latency */}
          <div style={{ ...CARD, display: "flex", flexWrap: "wrap", alignItems: "center", gap: "16px 32px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: connColor, display: "inline-block" }} />
              <span style={{ color: INK.secondary }}>Stream:</span>
              <b>{status}</b>
            </div>

            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span style={{ fontSize: 13, color: INK.secondary }}>Latency</span>
              <span style={{ fontSize: 22, fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>
                {latency != null ? latency.toFixed(0) : "—"}
                <span style={{ fontSize: 12, color: INK.muted, fontWeight: 400 }}>ms</span>
              </span>
              <span style={{ fontSize: 11, color: latencyStatusColor, display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: latencyStatusColor, display: "inline-block" }} />
                {latency == null ? "" : latencyOk ? `under ${LATENCY_BUDGET_MS}ms budget` : "over budget"}
              </span>
            </div>

            <p style={{ color: INK.muted, fontSize: 12, margin: 0, lineHeight: 1.5, flex: "1 1 320px", minWidth: 260 }}>
              Time from a (simulated) brain signal arriving to its decision reaching this page.
              For a cursor that feels instant, that round trip should stay under{" "}
              {LATENCY_BUDGET_MS}ms &mdash; roughly one eye-blink.
            </p>
          </div>

          {/* Cursor + classification */}
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "stretch" }}>
            <div style={{ ...CARD, flex: "1 1 560px" }}>
              <SectionHeading
                eyebrow="Step 5"
                sub="The actual output of the whole system: a cursor that a person could, in principle, steer with imagined movement alone."
              >
                Cursor
              </SectionHeading>
              <div style={{
                position: "relative", width: "100%", maxWidth: 620, aspectRatio: "620 / 420",
                border: `1px solid ${INK.border}`, borderRadius: 10, background: INK.surface,
                overflow: "hidden", margin: "0 auto",
              }}>
                <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: INK.gridline }} />
                <div style={{ position: "absolute", left: 16, top: 12, fontSize: 11, color: INK.muted }}>← LEFT</div>
                <div style={{ position: "absolute", right: 16, top: 12, fontSize: 11, color: INK.muted }}>RIGHT →</div>
                <div style={{ position: "absolute", left: "50%", top: 12, transform: "translateX(-50%)", fontSize: 11, color: INK.muted }}>REST</div>

                <div ref={cursorRef} style={{
                  position: "absolute", width: 20, height: 20, borderRadius: "50%",
                  background: cursorFill,
                  boxShadow: `0 0 0 2px ${INK.surface}, 0 0 0 4px ${latencyStatusColor}, 0 0 16px ${cursorFill}`,
                  transition: "background 120ms linear",
                }} />
              </div>
              <p style={{ color: INK.muted, fontSize: 12, marginTop: 12, marginBottom: 0, lineHeight: 1.55 }}>
                <b style={{ color: INK.secondary }}>Fill color</b> = current decode class (gray while
                below the {GATE_PCT}% gate). <b style={{ color: INK.secondary }}>Outer ring</b> = latency
                status (green/red, see above). The dot never teleports to a label &mdash; it gets a
                confidence-weighted <i>push</i> each decode and drifts back to center via friction, the
                same way a real pointer would feel under noisy control.
              </p>
            </div>

            <div style={{ ...CARD, flex: "1 1 300px" }}>
              <SectionHeading
                eyebrow="Step 3"
                sub="Out of the 3 possible states, how likely does the model think each one is, based on the last ~1 second of signal?"
              >
                Model's guess
              </SectionHeading>
              {CLASSES.map((c) => (
                <ProbBar key={c} cls={c} value={decode?.probs?.[c] ?? 0} isTop={decode?.label === c} />
              ))}
              <p style={{ color: INK.muted, fontSize: 11, margin: "2px 0 14px" }}>
                The thin vertical tick on each bar is the {GATE_PCT}% action threshold from step 4.
              </p>
              <Legend />

              <div style={{ marginTop: 22, marginBottom: 10 }}>
                <SectionHeading sub="One tick per decode (~10/sec), newest on the right. Lets you see whether the model is flip-flopping or holding a steady guess, at a glance.">
                  History
                </SectionHeading>
              </div>
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
                Short, dim ticks fell below the {GATE_PCT}% gate and were ignored (step 4).
              </p>
            </div>
          </div>

          <Glossary />
        </div>
      </div>
    </div>
  );
}
