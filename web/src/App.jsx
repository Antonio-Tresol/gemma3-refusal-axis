import { useEffect, useState } from "react";
import { palette, fonts, cssVars } from "./tokens.js";
import { TopNav } from "./components/TopNav.jsx";
import { LandingPage } from "./pages/Landing.jsx";
import { RefusalAxisPage } from "./pages/RefusalAxis.jsx";
import { FeatureHierarchyPage } from "./pages/FeatureHierarchy.jsx";
import { MethodologyPage } from "./pages/Methodology.jsx";
import { ReferencesPage } from "./pages/References.jsx";
import { GlossaryPage } from "./pages/Glossary.jsx";

// Top-level route slug for hash routes like `#/glossary` or `#/glossary#refusal`.
function routeOf(hash) {
  if (!hash || hash === "#" || hash === "#/") return "#/";
  const after = hash.slice(2); // drop leading "#/"
  const stop = after.indexOf("#");
  const route = stop >= 0 ? after.slice(0, stop) : after;
  return "#/" + route;
}

function useHashRoute() {
  const get = () => (typeof window === "undefined" ? "#/" : window.location.hash || "#/");
  const [hash, setHash] = useState(get);
  useEffect(() => {
    const handler = () => {
      const next = get();
      const prevRoute = routeOf(hash);
      const nextRoute = routeOf(next);
      setHash(next);
      // Scroll to top only when the top-level route changes (not when the
      // user follows an in-page anchor like `#/glossary#refusal-axis`).
      if (prevRoute !== nextRoute) {
        window.scrollTo({ top: 0, behavior: "instant" });
      }
    };
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return routeOf(hash);
}

function useScrolled() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 12);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);
  return scrolled;
}

export default function App() {
  const view = useHashRoute();
  const scrolled = useScrolled();

  let page;
  switch (view) {
    case "#/refusal-axis":
      page = <RefusalAxisPage />;
      break;
    case "#/feature-hierarchy":
      page = <FeatureHierarchyPage />;
      break;
    case "#/methodology":
      page = <MethodologyPage />;
      break;
    case "#/references":
      page = <ReferencesPage />;
      break;
    case "#/glossary":
      page = <GlossaryPage />;
      break;
    default:
      page = <LandingPage />;
  }

  return (
    <div
      style={{
        ...cssVars,
        background: palette.bg,
        color: palette.body,
        fontFamily: fonts.body,
        minHeight: "100vh",
      }}
    >
      <link
        href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300;1,8..60,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
        rel="stylesheet"
      />
      <style>{`
        @keyframes fadeIn { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:translateY(0); } }
        * { margin:0; padding:0; box-sizing:border-box; }
        html { scroll-behavior: smooth; }
        body { background: ${palette.bg}; color: ${palette.body}; -webkit-font-smoothing: antialiased; }
        ::selection { background: #E5E5E5; color: ${palette.text}; }

        /* Plain inline links */
        .link-plain { text-decoration: none; transition: color 0.15s ease; }
        .link-plain svg { transition: transform 0.18s ease; }
        .link-plain:hover { color: ${palette.text} !important; text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
        .link-plain:hover svg { transform: translate(2px, -2px); }

        /* Resource / pre-work link rows */
        .link-a { transition: color 0.15s ease; }
        .link-a em { transition: text-decoration 0.15s ease; }
        .link-a svg { transition: transform 0.18s ease; }
        .link-a:hover { color: ${palette.text} !important; }
        .link-a:hover em { text-decoration: underline; text-underline-offset: 3px; text-decoration-thickness: 1px; }
        .link-a:hover svg { transform: translate(2px, -2px); }

        /* Hoverable cards */
        .hover-card { transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease; }
        .hover-card svg { transition: transform 0.2s ease; }
        .hover-card:hover { border-color: ${palette.text} !important; background: ${palette.surfaceAlt} !important; transform: translateY(-1px); }
        .hover-card:hover svg { transform: translate(2px, -2px); }

        /* TopNav links */
        .nav-link {
          color: ${palette.muted};
          transition: color 0.15s ease;
        }
        .nav-link:hover { color: ${palette.text}; text-decoration: underline; text-underline-offset: 4px; text-decoration-thickness: 1px; }
        .nav-link--active { color: ${palette.text}; text-decoration: underline; text-underline-offset: 4px; text-decoration-thickness: 1px; }

        /* Math (KaTeX) */
        .katex { font-size: 1.05em; }

        /* Headings tighter on mobile */
        @media (max-width: 640px) {
          .nav-link { font-size: 10.5px !important; }
        }

        /* Print friendliness */
        @media print {
          nav, .no-print { display: none !important; }
          .hover-card { transform: none !important; }
        }
      `}</style>

      <TopNav view={view} scrolled={scrolled} />
      <main key={view} style={{ animation: "fadeIn 0.25s ease", paddingBottom: "120px" }}>
        {page}
      </main>
      <footer
        style={{
          maxWidth: "820px",
          margin: "0 auto",
          padding: "32px 28px 64px",
          fontFamily: fonts.mono,
          fontSize: "11px",
          color: palette.muted,
          lineHeight: 1.7,
          borderTop: `1px solid ${palette.border}`,
        }}
      >
        <p>
          By <span style={{ color: palette.text }}>Antonio Badilla-Olivas</span>. MIT licensed.{" "}
          <a
            className="link-plain"
            href="https://github.com/Antonio-Tresol/gemma3-refusal-axis"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: palette.text }}
          >
            Source on GitHub
          </a>
          .
        </p>
      </footer>
    </div>
  );
}
