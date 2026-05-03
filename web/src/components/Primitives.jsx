import { useEffect, useId, useRef, useState } from "react";
import katex from "katex";
import { palette, fonts, layout } from "../tokens.js";
import { glossaryBySlug, sources as glossarySources } from "../glossary.js";
import { methodologyBySlug } from "../methodology.js";

// Inline KaTeX. Pass `display` for block/centred display mode.
// KaTeX CSS is imported once in main.jsx so callers don't need to do it again.
export function Math({ children, display = false }) {
  const html = katex.renderToString(String(children), {
    displayMode: display,
    throwOnError: false,
    output: "html",
  });
  return (
    <span
      style={{ fontFamily: "inherit" }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

// Bespoke Python tokeniser used by the CodeTabs Pseudo-Python view.
// Five visual categories, mapped to the site palette so highlighting reads
// as part of the editorial style rather than a stock IDE theme.
const PY_KEYWORDS = new Set([
  "def", "class", "if", "elif", "else", "for", "while", "in", "return",
  "import", "from", "as", "lambda", "not", "and", "or", "is", "None", "True",
  "False", "pass", "break", "continue", "with", "try", "except", "finally",
  "raise", "global", "nonlocal", "yield", "async", "await", "del",
]);

const PY_TOKEN_STYLES = {
  comment: { color: palette.muted, fontStyle: "italic" },
  string: { color: "#1d3557" },
  number: { color: "#457b9d" },
  keyword: { color: palette.orange, fontWeight: 600 },
  punct: { color: "#737373" },
  ident: { color: palette.text },
  ws: {},
};

function tokenisePython(code) {
  const tokens = [];
  // Master pattern, in priority order: comment, triple-string, single-string,
  // number, identifier (resolved later to keyword vs ident), whitespace, punct.
  const re =
    /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?''')|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(\b\d+\.?\d*\b)|(\b[A-Za-z_][A-Za-z0-9_]*\b)|(\s+)|([^\s\w])/g;
  let m;
  while ((m = re.exec(code)) !== null) {
    if (m[1]) tokens.push({ type: "comment", text: m[1] });
    else if (m[2]) tokens.push({ type: "string", text: m[2] });
    else if (m[3]) tokens.push({ type: "string", text: m[3] });
    else if (m[4]) tokens.push({ type: "number", text: m[4] });
    else if (m[5]) {
      tokens.push({
        type: PY_KEYWORDS.has(m[5]) ? "keyword" : "ident",
        text: m[5],
      });
    } else if (m[6]) tokens.push({ type: "ws", text: m[6] });
    else if (m[7]) tokens.push({ type: "punct", text: m[7] });
  }
  return tokens;
}

// Tabbed code/math block: lets the reader toggle between a Pseudo-Python view
// and a Math notation view. Shares the Card visual language (white surface,
// 10px radius, 1px border). Keyboard accessible (left/right cycles tabs).
// The Python view is syntax-highlighted via tokenisePython using a custom
// non-canonical theme that matches the site palette.
export function CodeTabs({ python, math, label }) {
  const [tab, setTab] = useState("math");

  const tabs = [
    { id: "math", label: "Math notation" },
    { id: "python", label: "Pseudo-Python" },
  ];

  const onKeyDown = (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const idx = tabs.findIndex((t) => t.id === tab);
    const next =
      e.key === "ArrowRight"
        ? tabs[(idx + 1) % tabs.length]
        : tabs[(idx - 1 + tabs.length) % tabs.length];
    setTab(next.id);
  };

  return (
    <div style={{ margin: "20px 0" }}>
      {label && (
        <div style={{ marginBottom: "8px" }}>
          <span
            style={{
              fontFamily: fonts.mono,
              fontSize: "10px",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: palette.muted,
              fontWeight: 500,
            }}
          >
            {label}
          </span>
        </div>
      )}
      <div
        style={{
          background: palette.surface,
          border: `1px solid ${palette.border}`,
          borderRadius: "10px",
          overflow: "hidden",
        }}
      >
        <div
          role="tablist"
          aria-label={label || "Code tabs"}
          onKeyDown={onKeyDown}
          style={{
            display: "flex",
            gap: "20px",
            padding: "10px 16px",
            borderBottom: `1px solid ${palette.border}`,
            background: palette.surfaceAlt,
          }}
        >
          {tabs.map((t) => {
            const active = t.id === tab;
            return (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={active}
                tabIndex={active ? 0 : -1}
                className="link-plain"
                onClick={() => setTab(t.id)}
                style={{
                  background: "transparent",
                  border: "none",
                  padding: "2px 0",
                  cursor: "pointer",
                  fontFamily: fonts.mono,
                  fontSize: "11px",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  fontWeight: 500,
                  color: active ? palette.text : palette.muted,
                  textDecoration: active ? "underline" : "none",
                  textUnderlineOffset: "4px",
                  textDecorationThickness: "1px",
                }}
              >
                {t.label}
              </button>
            );
          })}
        </div>
        <div role="tabpanel" style={{ padding: "16px 18px" }}>
          {tab === "python" ? (
            <pre
              style={{
                margin: 0,
                fontFamily: fonts.mono,
                fontSize: "13px",
                lineHeight: 1.6,
                color: palette.text,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                overflowX: "auto",
              }}
            >
              {tokenisePython(typeof python === "string" ? python : String(python ?? "")).map(
                (tok, i) => (
                  <span key={i} style={PY_TOKEN_STYLES[tok.type] || undefined}>
                    {tok.text}
                  </span>
                )
              )}
            </pre>
          ) : (
            <div
              style={{
                fontSize: "15px",
                lineHeight: 1.6,
                color: palette.text,
                overflowX: "auto",
              }}
            >
              <Math display>{math}</Math>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Outbound-arrow glyph used on every external link.
export function Arrow({ size = 12, color = "currentColor", direction = "out" }) {
  const paths = {
    out: "M1 11L11 1M11 1H4M11 1V8",
    right: "M1 6H11M11 6L7 2M11 6L7 10",
    left: "M11 6H1M1 6L5 2M1 6L5 10",
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 12 12"
      fill="none"
      style={{ flexShrink: 0 }}
      aria-hidden
    >
      <path
        d={paths[direction] || paths.out}
        stroke={color}
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// Plain inline link with the standard "text + small clay arrow" treatment.
// External links (http*) get the outbound arrow + open in a new tab.
// Internal hash links get just an underline-on-hover treatment, no arrow.
// Pass `arrow="right"|"left"` to force a directional arrow on a hash link.
export function InlineLink({ href, children, style, arrow }) {
  const isExternal = href.startsWith("http");
  const showArrow = arrow || (isExternal ? "out" : null);
  return (
    <a
      className="link-plain"
      href={href}
      target={isExternal ? "_blank" : undefined}
      rel={isExternal ? "noopener noreferrer" : undefined}
      style={{
        color: palette.text,
        display: "inline",
        ...style,
      }}
    >
      {showArrow === "left" && (
        <>
          <Arrow size={10} color={palette.orange} direction="left" />
          {" "}
        </>
      )}
      {children}
      {showArrow === "out" && (
        <>
          {" "}
          <Arrow size={10} color={palette.orange} direction="out" />
        </>
      )}
      {showArrow === "right" && (
        <>
          {" "}
          <Arrow size={10} color={palette.orange} direction="right" />
        </>
      )}
    </a>
  );
}

// Inline glossary term. Dotted underline (palette.muted); hover or focus opens a
// small popover with a one-sentence definition + a "See glossary" link to the
// full entry. Click navigates to /glossary#<slug>.
//
// Props: name (slug string), children (optional custom label).
export function Term({ name, children }) {
  return (
    <TermPopover
      entriesBySlug={glossaryBySlug}
      name={name}
      routePrefix="#/glossary"
      footerLabel="See glossary →"
    >
      {children}
    </TermPopover>
  );
}

// Inline ad-hoc note. Dashed underline (palette.muted); hover/focus opens a
// small popover whose content is supplied inline rather than looked up.
// For prose where the reader benefits from extra context but doesn't need
// the interruption of a full Callout. If the content is must-read, use
// Callout instead so the reader cannot skip past it.
//
// Props: title (mono uppercase header), body (popover body), children (the
// underlined inline label in the prose).
export function Note({ title, body, children }) {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const wrapRef = useRef(null);
  const popId = useId();

  useEffect(() => {
    if (!pinned) return;
    function onDoc(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setPinned(false);
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [pinned]);

  return (
    <span
      ref={wrapRef}
      style={{ position: "relative", display: "inline" }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => !pinned && setOpen(false)}
    >
      <span
        tabIndex={0}
        role="button"
        aria-describedby={open ? popId : undefined}
        aria-expanded={open}
        onFocus={() => setOpen(true)}
        onBlur={() => !pinned && setOpen(false)}
        onClick={() => {
          if (!pinned && window.matchMedia && window.matchMedia("(hover: none)").matches) {
            setPinned((p) => !p);
            setOpen((o) => !o);
          }
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setPinned((p) => !p);
            setOpen((o) => !o);
          }
        }}
        style={{
          color: "inherit",
          textDecoration: "underline",
          textDecorationStyle: "dashed",
          textDecorationColor: palette.muted,
          textUnderlineOffset: "3px",
          textDecorationThickness: "1px",
          cursor: "help",
        }}
      >
        {children}
      </span>
      {open && (
        <span
          id={popId}
          role="tooltip"
          style={{
            position: "absolute",
            left: 0,
            top: "calc(100% + 8px)",
            zIndex: 60,
            width: "min(360px, 86vw)",
            padding: "12px 14px",
            background: palette.surface,
            border: `1px solid ${palette.border}`,
            borderRadius: "10px",
            boxShadow: "0 8px 24px rgba(10,10,10,0.08), 0 1px 2px rgba(10,10,10,0.04)",
            fontFamily: fonts.body,
            fontSize: "13.5px",
            lineHeight: 1.55,
            color: palette.body,
            display: "block",
            whiteSpace: "normal",
          }}
        >
          {title && (
            <span
              style={{
                display: "block",
                fontFamily: fonts.mono,
                fontSize: "11px",
                letterSpacing: "0.02em",
                textTransform: "none",
                color: palette.muted,
                marginBottom: "6px",
                fontWeight: 500,
              }}
            >
              {title}
            </span>
          )}
          <span style={{ display: "block", color: palette.text }}>{body}</span>
        </span>
      )}
    </span>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// TermPopover: shared popover machinery for both Term and MethodTerm.
//
// Props:
//   entriesBySlug  object   the lookup map (glossaryBySlug or methodologyBySlug)
//   name           string   the slug to look up
//   children       node     optional custom label text; falls back to entry.term
//   routePrefix    string   hash-router path without trailing slash, e.g. "#/glossary"
//   footerLabel    string   text for the "See …" link at the bottom of the popover
//   underlineStyle object   CSS overrides for the anchor's text-decoration (optional)
//   popoverAccent  string   colour for the popover's "See …" link (optional)
// ──────────────────────────────────────────────────────────────────────────────
function TermPopover({
  entriesBySlug,
  name,
  children,
  routePrefix,
  footerLabel,
  underlineStyle = {},
  popoverAccent,
}) {
  const entry = entriesBySlug[name];
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const wrapRef = useRef(null);
  const popId = useId();

  useEffect(() => {
    if (!pinned) return;
    function onDoc(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setPinned(false);
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [pinned]);

  if (!entry) {
    return <span>{children || name}</span>;
  }

  const label = children || entry.term.toLowerCase();
  const href = `${routePrefix}#${entry.anchor ?? entry.slug}`;
  const accent = popoverAccent || palette.orange;

  return (
    <span
      ref={wrapRef}
      style={{ position: "relative", display: "inline" }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => !pinned && setOpen(false)}
    >
      <a
        href={href}
        aria-describedby={open ? popId : undefined}
        onFocus={() => setOpen(true)}
        onBlur={() => !pinned && setOpen(false)}
        onClick={(e) => {
          if (!pinned && window.matchMedia && window.matchMedia("(hover: none)").matches) {
            e.preventDefault();
            setPinned(true);
            setOpen(true);
          }
        }}
        style={{
          color: "inherit",
          textDecoration: "underline",
          textDecorationStyle: "dotted",
          textDecorationColor: palette.muted,
          textUnderlineOffset: "3px",
          textDecorationThickness: "1px",
          cursor: "help",
          ...underlineStyle,
        }}
      >
        {label}
      </a>
      {open && (
        <span
          id={popId}
          role="tooltip"
          style={{
            position: "absolute",
            left: 0,
            top: "calc(100% + 8px)",
            zIndex: 60,
            width: "min(340px, 86vw)",
            padding: "12px 14px",
            background: palette.surface,
            border: `1px solid ${palette.border}`,
            borderRadius: "10px",
            boxShadow: "0 8px 24px rgba(10,10,10,0.08), 0 1px 2px rgba(10,10,10,0.04)",
            fontFamily: fonts.body,
            fontSize: "13.5px",
            lineHeight: 1.55,
            color: palette.body,
            display: "block",
            whiteSpace: "normal",
          }}
        >
          <span
            style={{
              display: "block",
              fontFamily: fonts.mono,
              fontSize: "11px",
              letterSpacing: "0.02em",
              textTransform: "none",
              color: palette.muted,
              marginBottom: "6px",
              fontWeight: 400,
            }}
          >
            {entry.term}
          </span>
          <span style={{ display: "block", color: palette.text, marginBottom: "8px" }}>
            {entry.short}
          </span>
          <a
            href={href}
            className="link-plain"
            style={{
              fontFamily: fonts.mono,
              fontSize: "11px",
              color: accent,
              textDecoration: "none",
              letterSpacing: "0.04em",
            }}
          >
            {footerLabel}
          </a>
        </span>
      )}
    </span>
  );
}

// Inline methodology term. Solid thin underline (palette.highlight) to
// distinguish at a glance from the glossary's dotted underline; the popover
// header links to the Methodology page section that describes exactly what we
// did. Complements <Term>; use for first-uses of procedural phrases rather than
// vocabulary definitions.
//
// Props: name (slug string), children (optional custom label).
export function MethodTerm({ name, children }) {
  return (
    <TermPopover
      entriesBySlug={methodologyBySlug}
      name={name}
      routePrefix="#/methodology"
      footerLabel="See methodology →"
      popoverAccent={palette.highlight}
      underlineStyle={{
        textDecorationStyle: "solid",
        textDecorationColor: palette.highlight,
        textDecorationThickness: "1px",
      }}
    >
      {children}
    </TermPopover>
  );
}

// Compact pill linking out to a published source by bibtex key.
// Used inside glossary cards so each definition shows where to read more.
export function SourcePill({ bibKey }) {
  const src = glossarySources[bibKey];
  if (!src) return null;
  return (
    <a
      className="link-plain"
      href={src.url}
      target={src.url.startsWith("http") ? "_blank" : undefined}
      rel="noopener noreferrer"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        padding: "2px 9px",
        background: palette.surfaceAlt,
        border: `1px solid ${palette.border}`,
        borderRadius: "999px",
        fontFamily: fonts.mono,
        fontSize: "10.5px",
        color: palette.body,
        textDecoration: "none",
        letterSpacing: "0.02em",
        whiteSpace: "nowrap",
      }}
    >
      {src.label}
      <Arrow size={9} color={palette.orange} />
    </a>
  );
}

// Small tag-style label (e.g. "PART I", "RAW ACTIVATIONS · NO SAE", "FIGURE 3").
export function Label({ children, color, style }) {
  return (
    <div
      style={{
        fontFamily: fonts.mono,
        fontSize: "10px",
        letterSpacing: "0.1em",
        textTransform: "uppercase",
        color: color || palette.muted,
        marginBottom: "12px",
        fontWeight: 500,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

// Small badge marking which methodological regime a section belongs to.
// "raw" (no-SAE residual stream work) vs "sae" (Gemma Scope 2 feature space).
export function RegimeBadge({ regime, style }) {
  const fg = regime.badgeFg;
  const bg = regime.badgeBg;
  return (
    <span
      title={regime.description}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        background: bg,
        color: fg,
        fontFamily: fonts.mono,
        fontSize: "10px",
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        padding: "3px 9px",
        borderRadius: "999px",
        border: `1px solid ${palette.border}`,
        fontWeight: 500,
        ...style,
      }}
    >
      <span style={{ opacity: 0.65 }}>{regime.label}</span>
      <span aria-hidden style={{ opacity: 0.4 }}>·</span>
      <span>{regime.short}</span>
    </span>
  );
}

// Shared surface for content cards. 8px radius, light hover lift.
export function Card({ as: Tag = "div", style, children, hover = true, ...rest }) {
  return (
    <Tag
      className={hover ? "hover-card" : undefined}
      style={{
        background: palette.surface,
        border: `1px solid ${palette.border}`,
        borderRadius: "10px",
        padding: "22px 24px",
        ...style,
      }}
      {...rest}
    >
      {children}
    </Tag>
  );
}

// 820px reading column.
export function Section({ style, children, wide = false }) {
  return (
    <section
      style={{
        maxWidth: wide ? layout.wideMaxWidth : layout.readingMaxWidth,
        margin: "0 auto",
        padding: layout.pagePadding,
        ...style,
      }}
    >
      {children}
    </section>
  );
}

export function H1({ children, style }) {
  return (
    <h1
      style={{
        fontFamily: fonts.display,
        fontSize: "clamp(40px, 5.5vw, 56px)",
        fontWeight: 700,
        lineHeight: 1.05,
        color: palette.text,
        marginBottom: "32px",
        letterSpacing: "-0.02em",
        ...style,
      }}
    >
      {children}
    </h1>
  );
}

export function H2({ children, style, id }) {
  return (
    <h2
      id={id}
      style={{
        fontFamily: fonts.display,
        fontSize: "32px",
        fontWeight: 600,
        color: palette.text,
        letterSpacing: "-0.015em",
        marginTop: "64px",
        marginBottom: "20px",
        scrollMarginTop: "100px",
        ...style,
      }}
    >
      {children}
    </h2>
  );
}

export function H3({ children, style, id }) {
  return (
    <h3
      id={id}
      style={{
        fontFamily: fonts.display,
        fontSize: "20px",
        fontWeight: 600,
        color: palette.text,
        marginTop: "36px",
        marginBottom: "12px",
        letterSpacing: "-0.01em",
        scrollMarginTop: "100px",
        ...style,
      }}
    >
      {children}
    </h3>
  );
}

// Body paragraph with serif body font + comfortable measure.
export function P({ children, style, lead = false }) {
  return (
    <p
      style={{
        fontFamily: fonts.body,
        fontSize: lead ? "20px" : "17px",
        lineHeight: lead ? 1.5 : 1.72,
        color: lead ? palette.text : palette.body,
        marginBottom: "20px",
        ...style,
      }}
    >
      {children}
    </p>
  );
}

// Mono label-style annotation, e.g. captions and meta.
export function Mono({ children, style }) {
  return (
    <span
      style={{
        fontFamily: fonts.mono,
        fontSize: "12px",
        color: palette.muted,
        ...style,
      }}
    >
      {children}
    </span>
  );
}

// Inline code style.
export function Code({ children, style }) {
  return (
    <code
      style={{
        fontFamily: fonts.mono,
        fontSize: "0.92em",
        color: palette.text,
        background: palette.surfaceAlt,
        padding: "1px 6px",
        borderRadius: "4px",
        border: `1px solid ${palette.border}`,
        ...style,
      }}
    >
      {children}
    </code>
  );
}

// Quoted aside / pull-quote / disclaimer block.
// Two tones: `neutral` for asides and notes; `warn` for fragile, exploratory,
// or otherwise should-watch-out content. Anything else falls back to neutral
// so a stale `tone="info"` does not visually clash with the rest of the site.
export function Callout({ children, tone = "neutral", style }) {
  const toneMap = {
    neutral: { border: palette.border, bg: palette.surfaceAlt, accent: palette.muted },
    warn: { border: "#ffe5cc", bg: "#fff5ee", accent: palette.orange },
  };
  const t = toneMap[tone] || toneMap.neutral;
  return (
    <aside
      style={{
        borderLeft: `3px solid ${t.accent}`,
        background: t.bg,
        padding: "16px 20px",
        borderRadius: "8px",
        margin: "24px 0",
        fontFamily: fonts.body,
        fontSize: "15.5px",
        lineHeight: 1.6,
        color: palette.body,
        ...style,
      }}
    >
      {children}
    </aside>
  );
}

// Figure container: image inside, rounded corners, mono caption underneath.
// Click the image to open it in a lightbox at higher resolution; the title
// (label + regime badge) and caption are reproduced inside the lightbox.
export function Figure({ src, alt, caption, label, regime, style }) {
  const [open, setOpen] = useState(false);
  const altText = alt || (typeof caption === "string" ? caption : label || "Figure");

  return (
    <>
      <figure
        style={{
          margin: "32px 0",
          ...style,
        }}
      >
        <button
          type="button"
          onClick={() => setOpen(true)}
          aria-label={`Zoom: ${altText}`}
          style={{
            display: "block",
            width: "100%",
            padding: 0,
            border: "none",
            background: "transparent",
            cursor: "zoom-in",
            textAlign: "left",
          }}
        >
          <div
            style={{
              background: palette.surface,
              border: `1px solid ${palette.border}`,
              borderRadius: "14px",
              padding: "16px",
              overflow: "hidden",
              transition: "border-color 0.2s ease, box-shadow 0.2s ease",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = palette.text;
              e.currentTarget.style.boxShadow = "0 4px 18px rgba(10, 10, 10, 0.06)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = palette.border;
              e.currentTarget.style.boxShadow = "none";
            }}
          >
            <img
              src={src}
              alt={altText}
              loading="lazy"
              style={{
                display: "block",
                width: "100%",
                height: "auto",
                borderRadius: "8px",
              }}
            />
          </div>
        </button>
        {(label || regime) && (
          <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "12px" }}>
            {label && <Label style={{ marginBottom: 0 }}>{label}</Label>}
            {regime && <RegimeBadge regime={regime} />}
          </div>
        )}
        {caption && (
          <figcaption
            style={{
              fontFamily: fonts.body,
              fontStyle: "italic",
              fontSize: "14.5px",
              lineHeight: 1.6,
              color: palette.muted,
              marginTop: "8px",
            }}
          >
            {caption}
          </figcaption>
        )}
      </figure>
      {open && (
        <FigureLightbox
          src={src}
          alt={altText}
          caption={caption}
          label={label}
          regime={regime}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

// Modal overlay shown when a Figure is clicked. Image is constrained to the
// viewport but enlarged from its inline size; the original label/regime/caption
// are reproduced underneath so the reader keeps full context while zoomed.
function FigureLightbox({ src, alt, caption, label, regime, onClose }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
    };
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={alt}
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        background: "rgba(10, 10, 10, 0.88)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px",
        animation: "fadeIn 0.18s ease",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: palette.surface,
          borderRadius: "14px",
          padding: "24px",
          maxWidth: "min(1400px, 96vw)",
          maxHeight: "94vh",
          display: "flex",
          flexDirection: "column",
          gap: "16px",
          overflow: "auto",
          position: "relative",
          boxShadow: "0 18px 60px rgba(0, 0, 0, 0.4)",
        }}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          style={{
            position: "absolute",
            top: "12px",
            right: "12px",
            width: "32px",
            height: "32px",
            border: `1px solid ${palette.border}`,
            background: palette.surface,
            color: palette.muted,
            borderRadius: "999px",
            cursor: "pointer",
            fontFamily: fonts.mono,
            fontSize: "14px",
            lineHeight: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "color 0.15s ease, border-color 0.15s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = palette.text;
            e.currentTarget.style.borderColor = palette.text;
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = palette.muted;
            e.currentTarget.style.borderColor = palette.border;
          }}
        >
          ✕
        </button>
        <img
          src={src}
          alt={alt}
          style={{
            display: "block",
            width: "100%",
            height: "auto",
            maxHeight: "78vh",
            objectFit: "contain",
            borderRadius: "8px",
            background: palette.surfaceAlt,
          }}
        />
        {(label || regime) && (
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            {label && <Label style={{ marginBottom: 0 }}>{label}</Label>}
            {regime && <RegimeBadge regime={regime} />}
          </div>
        )}
        {caption && (
          <figcaption
            style={{
              fontFamily: fonts.body,
              fontStyle: "italic",
              fontSize: "15px",
              lineHeight: 1.6,
              color: palette.body,
            }}
          >
            {caption}
          </figcaption>
        )}
        <Mono style={{ color: palette.muted, fontSize: "11px" }}>
          Press Esc or click outside to close
        </Mono>
      </div>
    </div>
  );
}

// Standard data table with the project's typographic pattern.
export function DataTable({ headers, rows, highlight, style }) {
  return (
    <div
      style={{
        margin: "20px 0",
        border: `1px solid ${palette.border}`,
        borderRadius: "10px",
        overflow: "hidden",
        background: palette.surface,
        ...style,
      }}
    >
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontFamily: fonts.body,
          fontSize: "14.5px",
        }}
      >
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th
                key={i}
                style={{
                  textAlign: "left",
                  fontFamily: fonts.mono,
                  fontSize: "11px",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  color: palette.muted,
                  padding: "12px 16px",
                  background: palette.surfaceAlt,
                  borderBottom: `1px solid ${palette.border}`,
                  fontWeight: 500,
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr
              key={ri}
              style={{
                background:
                  highlight && highlight(row, ri) ? palette.surfaceAlt : "transparent",
              }}
            >
              {row.map((cell, ci) => (
                <td
                  key={ci}
                  style={{
                    padding: "12px 16px",
                    borderTop: ri === 0 ? "none" : `1px solid ${palette.border}`,
                    color: palette.body,
                  }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
