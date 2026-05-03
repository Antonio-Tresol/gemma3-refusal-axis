import { useEffect, useState } from "react";
import { palette, fonts } from "../tokens.js";
import { Card, Mono } from "./Primitives.jsx";

const DATA_URL = import.meta.env.BASE_URL + "data/capping.json";

const CAP_DIRECTIONS = [
  { id: "overall_refusal", label: "Overall (mean axis)" },
  { id: "safety", label: "Safety" },
  { id: "capability_boundary", label: "Capability" },
  { id: "privacy", label: "Privacy" },
];
const PROMPT_DOMAINS = [
  { id: "safety", label: "Safety Δ" },
  { id: "capability_boundary", label: "Capability Δ" },
  { id: "privacy", label: "Privacy Δ" },
  { id: "benign", label: "Benign Δ" },
];
const PERCENTILES = ["10", "25", "50", "75", "90", "95", "99"];

// Diverging scale for capping deltas: red = strong reduction, white = none, blue = increase.
function colourForDelta(v) {
  // delta in roughly [-35, +10]; clamp.
  const max = 35;
  const t = Math.max(-1, Math.min(1, v / max));
  const lerp = (a, b, k) => Math.round(a + (b - a) * k);
  if (t < 0) {
    // negative -> red intensity
    const k = Math.abs(t);
    return `rgb(${lerp(255, 230, k)}, ${lerp(255, 57, k)}, ${lerp(255, 70, k)})`;
  }
  // positive -> blue
  const k = t;
  return `rgb(${lerp(255, 29, k)}, ${lerp(255, 53, k)}, ${lerp(255, 87, k)})`;
}

export function CappingMatrix() {
  const [data, setData] = useState(null);
  const [tauIdx, setTauIdx] = useState(2); // p50
  const [hover, setHover] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((r) => r.json())
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => console.error("CappingMatrix data load failed", e));
    return () => {
      cancelled = true;
    };
  }, []);

  const cellW = 130;
  const cellH = 60;
  const gap = 4;
  const margin = { top: 60, left: 180, right: 24, bottom: 30 };
  const w = margin.left + PROMPT_DOMAINS.length * (cellW + gap) + margin.right;
  const h = margin.top + CAP_DIRECTIONS.length * (cellH + gap) + margin.bottom;

  if (!data) {
    return (
      <Card
        style={{
          minHeight: "440px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Mono>Loading capping data…</Mono>
      </Card>
    );
  }

  const tau = PERCENTILES[tauIdx];

  const valueAt = (capDir, prompt) => {
    const blob = data[capDir]?.[prompt]?.[tau];
    if (!blob) return null;
    return blob;
  };

  const isSafetyDiagonal = (capDir, prompt) => capDir === "safety" && prompt === "safety";

  return (
    <Card style={{ padding: "20px 24px" }}>
      {/* Tau slider */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "20px",
          marginBottom: "20px",
          flexWrap: "wrap",
        }}
      >
        <div>
          <Mono style={{ fontSize: "10px", letterSpacing: "0.1em", textTransform: "uppercase" }}>
            Capping threshold τ
          </Mono>
          <div
            style={{
              fontFamily: fonts.display,
              fontSize: "28px",
              fontWeight: 600,
              color: palette.text,
              letterSpacing: "-0.02em",
              marginTop: "4px",
            }}
          >
            p<sub style={{ fontSize: "16px" }}>{tau}</sub>
            <span
              style={{
                fontFamily: fonts.mono,
                fontSize: "12px",
                color: palette.muted,
                marginLeft: "10px",
                fontWeight: 400,
              }}
            >
              percentile of benign-prompt projections
            </span>
          </div>
        </div>
        <div style={{ flex: 1, minWidth: "240px" }}>
          <input
            type="range"
            min={0}
            max={PERCENTILES.length - 1}
            step={1}
            value={tauIdx}
            onChange={(e) => setTauIdx(Number(e.target.value))}
            style={{
              width: "100%",
              accentColor: palette.refuse,
              cursor: "pointer",
            }}
            aria-label="Capping threshold τ"
          />
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: "4px",
              fontFamily: fonts.mono,
              fontSize: "10px",
              color: palette.muted,
            }}
          >
            {PERCENTILES.map((p) => (
              <span key={p} style={{ color: p === tau ? palette.text : palette.muted }}>
                p{p}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ overflowX: "auto" }}>
          <svg width={w} height={h} role="img" aria-label="Capping independence matrix">
            {/* Column headers */}
            {PROMPT_DOMAINS.map((p, j) => (
              <text
                key={"c" + j}
                x={margin.left + j * (cellW + gap) + cellW / 2}
                y={margin.top - 14}
                textAnchor="middle"
                style={{
                  fontFamily: fonts.mono,
                  fontSize: "11px",
                  fill: palette.text,
                  letterSpacing: "0.04em",
                }}
              >
                {p.label}
              </text>
            ))}
            {/* Row headers */}
            {CAP_DIRECTIONS.map((c, i) => (
              <text
                key={"r" + i}
                x={margin.left - 12}
                y={margin.top + i * (cellH + gap) + cellH / 2 + 4}
                textAnchor="end"
                style={{
                  fontFamily: fonts.mono,
                  fontSize: "11px",
                  fill: palette.text,
                  letterSpacing: "0.04em",
                }}
              >
                Cap: {c.label}
              </text>
            ))}
            {/* Cells */}
            {CAP_DIRECTIONS.map((c, i) =>
              PROMPT_DOMAINS.map((p, j) => {
                const blob = valueAt(c.id, p.id);
                const v = blob ? blob.mean_delta : null;
                const x = margin.left + j * (cellW + gap);
                const y = margin.top + i * (cellH + gap);
                const isHover = hover && hover.c === c.id && hover.p === p.id;
                const sweetSpot = isSafetyDiagonal(c.id, p.id) && tau === "50";
                return (
                  <g
                    key={`${i}-${j}`}
                    onMouseEnter={() => setHover({ c: c.id, p: p.id, blob, capLabel: c.label, promptLabel: p.label })}
                    onMouseLeave={() => setHover(null)}
                    style={{ cursor: "pointer" }}
                  >
                    <rect
                      x={x}
                      y={y}
                      width={cellW}
                      height={cellH}
                      rx={6}
                      ry={6}
                      fill={v === null ? "#f0f0f0" : colourForDelta(v)}
                      stroke={isHover ? palette.text : sweetSpot ? palette.orange : palette.border}
                      strokeWidth={isHover || sweetSpot ? 2 : 1}
                      style={{ transition: "stroke 0.12s ease" }}
                    />
                    <text
                      x={x + cellW / 2}
                      y={y + cellH / 2 + 5}
                      textAnchor="middle"
                      style={{
                        fontFamily: fonts.mono,
                        fontSize: "14px",
                        fontWeight: 500,
                        fill: v !== null && Math.abs(v) > 18 ? "#fff" : palette.text,
                        pointerEvents: "none",
                      }}
                    >
                      {v === null ? "n/a" : v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1)}
                    </text>
                  </g>
                );
              })
            )}
          </svg>
        </div>

        {/* Horizontal info strip */}
        <div
          style={{
            background: palette.surfaceAlt,
            border: `1px solid ${palette.border}`,
            borderRadius: "10px",
            padding: "12px 16px",
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "20px",
            rowGap: "8px",
          }}
        >
          {hover && hover.blob ? (
            <>
              <InfoSection label="Capping">
                <span
                  style={{
                    fontFamily: fonts.display,
                    fontSize: "13px",
                    fontWeight: 600,
                    color: palette.text,
                    letterSpacing: "-0.01em",
                  }}
                >
                  {hover.capLabel} → {hover.promptLabel.replace(" Δ", " prompts")}
                </span>
              </InfoSection>
              <InfoSection label="Mean Δ refusal">
                <span
                  style={{
                    fontFamily: fonts.mono,
                    fontSize: "18px",
                    fontWeight: 500,
                    color: hover.blob.mean_delta < 0 ? palette.comply : palette.refuse,
                    letterSpacing: "-0.01em",
                  }}
                >
                  {hover.blob.mean_delta > 0 ? "+" : ""}
                  {hover.blob.mean_delta.toFixed(1)}
                </span>
              </InfoSection>
              <InfoSection label="n coherent / n deltas">
                <span
                  style={{
                    fontFamily: fonts.mono,
                    fontSize: "13px",
                    color: palette.body,
                  }}
                >
                  <strong style={{ color: palette.text }}>{hover.blob.n_coherent}</strong>
                  <span style={{ color: palette.muted }}> / </span>
                  <strong style={{ color: palette.text }}>{hover.blob.n_deltas}</strong>
                </span>
              </InfoSection>
              <Mono style={{ fontSize: "11px", color: palette.muted }}>
                negative = less refusal, positive = more refusal
              </Mono>
              {isSafetyDiagonal(hover.c, hover.p) && tauIdx === 2 && (
                <Mono style={{ fontSize: "11px", color: palette.orange, lineHeight: 1.5 }}>
                  ★ Sweet-spot cell. Test 4a survives (60.9× variance ratio over random
                  directions); test 4e flags this as exploratory selection.
                </Mono>
              )}
            </>
          ) : (
            <>
              <Mono style={{ fontSize: "11px", color: palette.muted }}>
                Hover any cell. Orange outline marks the sweet-spot diagonal.
              </Mono>
              <LegendChip colour={colourForDelta(-30)} label="strong reduction" />
              <LegendChip colour={colourForDelta(0)} label="no effect" bordered />
              <LegendChip colour={colourForDelta(8)} label="increase (rare)" />
            </>
          )}
        </div>
      </div>
    </Card>
  );
}

function InfoSection({ label, children }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      <Mono
        style={{
          fontSize: "9.5px",
          letterSpacing: "0.1em",
          textTransform: "uppercase",
        }}
      >
        {label}
      </Mono>
      {children}
    </div>
  );
}

function LegendChip({ colour, label, bordered = false }) {
  return (
    <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
      <span
        style={{
          width: "14px",
          height: "14px",
          borderRadius: "4px",
          background: colour,
          flexShrink: 0,
          border: bordered ? `1px solid ${palette.border}` : "none",
        }}
      />
      <Mono style={{ fontSize: "11px" }}>{label}</Mono>
    </div>
  );
}
