import { useEffect, useRef, useState } from "react";

// Velocity-based cursor control. Decoded class + confidence -> cursor velocity.
// left/right push the cursor horizontally; rest decays it toward stop. This is
// deliberately forgiving of noisy per-epoch decodes.

const SPEED = 6;          // px per update at full confidence
const CONF_GATE = 0.5;    // ignore low-confidence decodes
const FRICTION = 0.85;    // velocity decay per frame

export default function App() {
  const [status, setStatus] = useState("connecting");
  const [decode, setDecode] = useState(null);
  const [latency, setLatency] = useState(null);
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

  return (
    <div style={{ fontFamily: "system-ui", padding: 24 }}>
      <h1>BCI Cursor Control</h1>
      <div style={{ display: "flex", gap: 24, marginBottom: 12 }}>
        <span>Status: <b>{status}</b></span>
        <span>Decode: <b>{decode?.label ?? "—"}</b></span>
        <span>Conf: <b>{decode ? (decode.confidence * 100).toFixed(0) + "%" : "—"}</b></span>
        <span>Latency: <b>{latency != null ? latency.toFixed(0) + "ms" : "—"}</b></span>
      </div>
      <div style={{
        position: "relative", width: 620, height: 420,
        border: "2px solid #333", borderRadius: 8, background: "#0b0b0f",
      }}>
        <div ref={cursorRef} style={{
          position: "absolute", width: 20, height: 20, borderRadius: "50%",
          background: latency > 200 ? "#e5484d" : "#30a46c",
          boxShadow: "0 0 12px currentColor",
        }} />
      </div>
      <p style={{ color: "#888", marginTop: 8 }}>
        Latency dot turns red if end-to-end exceeds the 200ms budget.
      </p>
    </div>
  );
}
