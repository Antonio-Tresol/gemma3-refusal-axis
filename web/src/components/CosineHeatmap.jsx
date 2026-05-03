import { useEffect, useState, useMemo } from "react";
import { palette, fonts, domains } from "../tokens.js";
import { Card, Mono } from "./Primitives.jsx";

const DATA_URL = import.meta.env.BASE_URL + "data/refusal_axis_results.json";
const FALS_URL = import.meta.env.BASE_URL + "data/all_tests.json";

// Diverging RedBlue scale, matching Lu et al. visual language.
function colourFor(v) {
  // v in [-1, 1]; remap to t in [0, 1]
  const t = (v + 1) / 2;
  // 0  -> #e63946 (red, compliance / negative cosine)
  // 0.5 -> #f7f7f7 (neutral)
  // 1  -> #1d3557 (navy, refusal / positive cosine)
  const lerp = (a, b, k) => Math.round(a + (b - a) * k);
  const stops = [
    { t: 0.0, r: 230, g: 57, b: 70 }, // #e63946
    { t: 0.5, r: 247, g: 247, b: 247 }, // #f7f7f7
    { t: 1.0, r: 29, g: 53, b: 87 }, // #1d3557
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (t >= a.t && t <= b.t) {
      const k = (t - a.t) / (b.t - a.t);
      return `rgb(${lerp(a.r, b.r, k)}, ${lerp(a.g, b.g, k)}, ${lerp(a.b, b.b, k)})`;
    }
  }
  return "#999";
}

function pairKey(a, b) {
  return [a, b].sort().join("_vs_");
}

export function CosineHeatmap() {
  const [matrix, setMatrix] = useState(null);
  const [cis, setCis] = useState(null);
  const [hover, setHover] = useState(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetch(DATA_URL).then((r) => r.json()),
      fetch(FALS_URL).then((r) => r.json()),
    ])
      .then(([axis, fals]) => {
        if (cancelled) return;
        setMatrix(axis.mean_response_token.domain_separation.cosine_matrix);
        setCis(fals.test_1b_bootstrap_ci.pairwise_cis);
      })
      .catch((e) => console.error("CosineHeatmap data load failed", e));
    return () => {
      cancelled = true;
    };
  }, []);

  const ids = useMemo(() => domains.map((d) => d.id), []);
  const labels = useMemo(() => domains.map((d) => d.label), []);

  const cellSize = 84;
  const gap = 2;
  const margin = { top: 90, left: 110, right: 24, bottom: 40 };
  const w = margin.left + ids.length * (cellSize + gap) + margin.right;
  const h = margin.top + ids.length * (cellSize + gap) + margin.bottom;

  if (!matrix) {
    return (
      <Card style={{ minHeight: "440px", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <Mono>Loading cosine matrix…</Mono>
      </Card>
    );
  }

  const ciFor = (a, b) => {
    if (!cis || a === b) return null;
    return cis[pairKey(a, b)] || null;
  };

  return (
    <Card style={{ padding: "20px 24px" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ overflowX: "auto" }}>
          <svg width={w} height={h} role="img" aria-label="Cross-domain cosine heatmap">
            {/* Column labels (top) */}
            {labels.map((lbl, j) => {
              const x = margin.left + j * (cellSize + gap) + cellSize / 2;
              return (
                <text
                  key={"c" + j}
                  x={x}
                  y={margin.top - 14}
                  textAnchor="middle"
                  transform={`rotate(-30 ${x} ${margin.top - 14})`}
                  style={{
                    fontFamily: fonts.mono,
                    fontSize: "11px",
                    fill: palette.text,
                    letterSpacing: "0.04em",
                  }}
                >
                  {lbl}
                </text>
              );
            })}
            {/* Row labels (left) */}
            {labels.map((lbl, i) => (
              <text
                key={"r" + i}
                x={margin.left - 12}
                y={margin.top + i * (cellSize + gap) + cellSize / 2 + 4}
                textAnchor="end"
                style={{
                  fontFamily: fonts.mono,
                  fontSize: "11px",
                  fill: palette.text,
                  letterSpacing: "0.04em",
                }}
              >
                {lbl}
              </text>
            ))}

            {/* Cells */}
            {ids.map((rowId, i) =>
              ids.map((colId, j) => {
                const v = matrix[rowId]?.[colId];
                if (v === undefined) return null;
                const x = margin.left + j * (cellSize + gap);
                const y = margin.top + i * (cellSize + gap);
                const isDiag = rowId === colId;
                const ci = ciFor(rowId, colId);
                const isHover = hover && hover.r === rowId && hover.c === colId;
                return (
                  <g
                    key={`${i}-${j}`}
                    onMouseEnter={() => setHover({ r: rowId, c: colId, v, ci })}
                    onMouseLeave={() => setHover(null)}
                    style={{ cursor: "pointer" }}
                  >
                    <rect
                      x={x}
                      y={y}
                      width={cellSize}
                      height={cellSize}
                      rx={6}
                      ry={6}
                      fill={colourFor(v)}
                      stroke={isHover ? palette.text : "transparent"}
                      strokeWidth={isHover ? 2 : 0}
                      style={{ transition: "stroke 0.12s ease" }}
                    />
                    <text
                      x={x + cellSize / 2}
                      y={y + cellSize / 2 + 5}
                      textAnchor="middle"
                      style={{
                        fontFamily: fonts.mono,
                        fontSize: "13px",
                        fontWeight: 500,
                        fill: Math.abs(v) > 0.55 ? "#fff" : palette.text,
                        pointerEvents: "none",
                      }}
                    >
                      {isDiag ? "1.00" : v.toFixed(2)}
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
          {hover ? (
            <>
              <InfoSection label="Pair">
                <span
                  style={{
                    fontFamily: fonts.display,
                    fontSize: "14px",
                    fontWeight: 600,
                    color: palette.text,
                    letterSpacing: "-0.01em",
                  }}
                >
                  {labelFor(hover.r)} ↔ {labelFor(hover.c)}
                </span>
              </InfoSection>
              <InfoSection label="Cosine">
                <span
                  style={{
                    fontFamily: fonts.mono,
                    fontSize: "16px",
                    fontWeight: 500,
                    color: colourFor(hover.v),
                    letterSpacing: "-0.01em",
                  }}
                >
                  {hover.r === hover.c ? "1.000" : hover.v.toFixed(3)}
                </span>
              </InfoSection>
              {hover.ci && (
                <InfoSection label="95% CI">
                  <span
                    style={{
                      fontFamily: fonts.mono,
                      fontSize: "13px",
                      color: palette.body,
                    }}
                  >
                    [{hover.ci.ci_lo.toFixed(3)}, {hover.ci.ci_hi.toFixed(3)}]
                  </span>
                </InfoSection>
              )}
              {hover.ci && hover.ci.ci_lo < 0 && hover.ci.ci_hi > 0.5 && (
                <Mono style={{ fontSize: "11px", color: palette.orange }}>
                  ⚠ CI not robust (spans 0 and crosses 0.5)
                </Mono>
              )}
              {hover.r === hover.c && (
                <Mono style={{ fontSize: "11px", color: palette.muted }}>
                  Diagonal: directions are unit-normalised, so self-cosine is 1.
                </Mono>
              )}
            </>
          ) : (
            <>
              <Mono style={{ fontSize: "11px", color: palette.muted }}>
                Hover any cell for cosine + bootstrap CI.
              </Mono>
              <LegendChip colour={colourFor(-0.5)} label="negative (anti-aligned)" />
              <LegendChip colour={colourFor(0)} label="0 (orthogonal)" bordered />
              <LegendChip colour={colourFor(1)} label="1 (aligned)" />
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

function labelFor(id) {
  const d = domains.find((x) => x.id === id);
  return d?.label || id;
}
