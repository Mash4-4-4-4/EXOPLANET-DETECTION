import { Orbit } from "lucide-react";

export default function Header() {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">
          <Orbit size={20} strokeWidth={1.7} />
        </div>
        <div>
          <div className="brand-name">
            EXO <span>DETECTOR</span>
          </div>
          <div className="brand-sub">AI EXOPLANET CLASSIFICATION</div>
        </div>
      </div>

      <nav>
        <a className="active" href="#detect">Detect</a>
        <a href="#about">About</a>
        <a href="#docs">Docs</a>
      </nav>
    </header>
  );
}