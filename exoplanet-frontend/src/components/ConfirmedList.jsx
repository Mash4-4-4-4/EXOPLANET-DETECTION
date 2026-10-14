import { ChevronRight } from "lucide-react";

function confidence(item) {
  const values = item.probabilities || {};
  return Math.max(
    values.CONFIRMED || 0,
    values.CANDIDATE || 0,
    values["FALSE POSITIVE"] || 0
  );
}

export default function ConfirmedList({ results, visibleCount, selected, onSelect }) {
  const confirmed = (results || [])
    .slice(0, visibleCount)
    .filter((item) => item.prediction === "CONFIRMED");

  return (
    <aside className="panel results-panel">
      <div className="list-heading">
        <div>
          <span className="status-dot confirmed" />
          <span>Confirmed Exoplanets</span>
        </div>
        <b>{confirmed.length}</b>
      </div>

      <div className="confirmed-list">
        {confirmed.length === 0 ? (
          <div className="empty-list">
            Confirmed objects will appear here as the scan progresses.
          </div>
        ) : (
          confirmed.map((item) => (
            <button
              key={item.index}
              className={`planet-row ${selected === item.index ? "selected" : ""}`}
              onClick={() => onSelect(item.index)}
            >
              <div className="planet-orb" />
              <div className="planet-copy">
                <strong>{item.name || `KOI-${item.index}`}</strong>
                <span>
                  KepID: {item.kepid ?? "not available"} · {(confidence(item) * 100).toFixed(1)}%
                </span>
              </div>
              <ChevronRight size={17} />
            </button>
          ))
        )}
      </div>
    </aside>
  );
}