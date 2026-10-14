import { useMemo } from "react";

const palette = {
  CONFIRMED: "#ff3038",
  CANDIDATE: "#ffbf24",
  "FALSE POSITIVE": "#e7e9ec",
};

function seeded(index) {
  const x = Math.sin(index * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

export default function RadarScanner({
  results,
  visibleCount,
  scanning,
  progress,
}) {
  const visible = results?.slice(0, visibleCount) || [];

  const points = useMemo(
    () =>
      visible.map((item, index) => {
        const angle = seeded(index + 1) * Math.PI * 2;
        const radius = 18 + seeded(index + 33) * 36;
        const x = 50 + Math.cos(angle) * radius;
        const y = 50 + Math.sin(angle) * radius;
        const size =
          item.prediction === "CONFIRMED"
            ? 8
            : item.prediction === "CANDIDATE"
              ? 6
              : 5;

        return {
          id: item.index,
          x,
          y,
          size,
          color: palette[item.prediction],
        };
      }),
    [visible]
  );

  return (
    <section className="scanner-wrap">
      <div className={`radar ${scanning ? "is-scanning" : "is-complete"}`}>
        <div className="radar-grid" />
        <div className="radar-cross horizontal" />
        <div className="radar-cross vertical" />
        <div className="radar-sweep" />

        <div className="radar-ring ring-1" />
        <div className="radar-ring ring-2" />
        <div className="radar-ring ring-3" />
        <div className="radar-ring ring-4" />

        {points.map((point) => (
          <span
            key={point.id}
            className="radar-point"
            style={{
              left: `${point.x}%`,
              top: `${point.y}%`,
              width: point.size,
              height: point.size,
              background: point.color,
              boxShadow: `0 0 12px ${point.color}`,
            }}
          />
        ))}

        <div className="radar-center" />
      </div>

      <div className="scan-progress">
        <div className="progress-track">
          <div style={{ width: `${progress}%` }} />
        </div>
        <span>{Math.round(progress)}%</span>
      </div>

      <div className="scan-status">
        {scanning
          ? `Scanning ${results?.length?.toLocaleString() || 0} objects...`
          : results
            ? `Scan complete · ${results.length.toLocaleString()} objects`
            : "Waiting for data"}
      </div>

      <div className="legend">
        <span><i className="legend-dot confirmed" />Confirmed</span>
        <span><i className="legend-dot candidate" />Candidate</span>
        <span><i className="legend-dot false" />False Positive</span>
      </div>
    </section>
  );
}