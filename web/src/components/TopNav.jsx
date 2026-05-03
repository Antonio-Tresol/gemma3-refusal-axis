import { palette, fonts, layout } from "../tokens.js";

const navItems = [
  { hash: "#/", label: "Refusal Axis", short: "Overview" },
  { hash: "#/refusal-axis", label: "Part I · No SAEs" },
  { hash: "#/feature-hierarchy", label: "Part II · SAEs" },
  { hash: "#/methodology", label: "Methodology" },
  { hash: "#/glossary", label: "Glossary" },
  { hash: "#/references", label: "References" },
];

export function TopNav({ view, scrolled }) {
  return (
    <nav
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        background: scrolled ? "rgba(244,244,244,0.9)" : palette.bg,
        backdropFilter: scrolled ? "blur(8px)" : undefined,
        WebkitBackdropFilter: scrolled ? "blur(8px)" : undefined,
        borderBottom: scrolled ? `1px solid ${palette.border}` : "1px solid transparent",
        transition: "all 0.2s ease",
      }}
    >
      <div
        style={{
          maxWidth: layout.wideMaxWidth,
          margin: "0 auto",
          padding: "14px 28px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
          flexWrap: "wrap",
          justifyContent: "space-between",
        }}
      >
        <a
          href="#/"
          className="link-plain"
          style={{
            fontFamily: fonts.display,
            fontSize: "14px",
            fontWeight: 600,
            color: palette.text,
            textDecoration: "none",
            letterSpacing: "-0.01em",
            whiteSpace: "nowrap",
          }}
        >
          The Refusal Axis
        </a>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "20px", alignItems: "baseline" }}>
          {navItems.map((item) => {
            const active = view === item.hash;
            return (
              <a
                key={item.hash}
                href={item.hash}
                className={active ? "nav-link nav-link--active" : "nav-link"}
                style={{
                  fontFamily: fonts.mono,
                  fontSize: "11.5px",
                  textDecoration: "none",
                  whiteSpace: "nowrap",
                  letterSpacing: "0.02em",
                }}
              >
                {item.label}
              </a>
            );
          })}
          <a
            href="https://github.com/Antonio-Tresol/gemma3-refusal-axis"
            target="_blank"
            rel="noopener noreferrer"
            className="nav-link"
            style={{
              fontFamily: fonts.mono,
              fontSize: "11.5px",
              textDecoration: "none",
              whiteSpace: "nowrap",
              letterSpacing: "0.02em",
            }}
          >
            GitHub ↗
          </a>
        </div>
      </div>
    </nav>
  );
}
