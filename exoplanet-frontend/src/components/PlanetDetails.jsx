function pct(value) {
  return `${((value || 0) * 100).toFixed(2)}%`;
}

function value(v, suffix = "") {
  return v === null || v === undefined || v === "" || Number.isNaN(Number(v))
    ? "N/A"
    : `${Number(v).toLocaleString(undefined, { maximumFractionDigits: 3 })}${suffix}`;
}

export default function PlanetDetails({ item }) {
  if (!item) {
    return (
      <section className="detail-card placeholder-detail">
        <span>Select a confirmed object</span>
      </section>
    );
  }

  const p = item.probabilities || {};

  return (
    <section className="detail-card">
      <div className="detail-head">
        <div>
          <span className="detail-kicker">SELECTED OBJECT</span>
          <h2>{item.name || `KOI-${item.index}`}</h2>
        </div>
        <span className="badge confirmed-badge">CONFIRMED</span>
      </div>

      <div className="detail-grid">
        <div><span>KepID</span><strong>{item.kepid ?? "N/A"}</strong></div>
        <div><span>Confidence</span><strong>{pct(p.CONFIRMED)}</strong></div>
        <div><span>Orbital Period</span><strong>{value(item.features?.koi_period, " d")}</strong></div>
        <div><span>Planet Radius</span><strong>{value(item.features?.koi_prad, " R⊕")}</strong></div>
        <div><span>Transit Duration</span><strong>{value(item.features?.koi_duration, " h")}</strong></div>
        <div><span>Equilibrium Temp.</span><strong>{value(item.features?.koi_teq, " K")}</strong></div>
        <div><span>Stellar Radius</span><strong>{value(item.features?.koi_srad, " R☉")}</strong></div>
        <div><span>Model SNR</span><strong>{value(item.features?.koi_model_snr)}</strong></div>
      </div>

      <div className="probabilities">
        <div className="prob-title">Classification probabilities</div>

        {[
          ["CONFIRMED", p.CONFIRMED, "confirmed"],
          ["CANDIDATE", p.CANDIDATE, "candidate"],
          ["FALSE POSITIVE", p["FALSE POSITIVE"], "false"],
        ].map(([label, prob, tone]) => (
          <div className="prob-row" key={label}>
            <span>{label}</span>
            <div className="prob-track">
              <div className={tone} style={{ width: `${(prob || 0) * 100}%` }} />
            </div>
            <b>{pct(prob)}</b>
          </div>
        ))}
      </div>
    </section>
  );
}