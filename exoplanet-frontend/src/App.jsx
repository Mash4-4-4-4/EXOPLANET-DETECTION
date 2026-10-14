import { useEffect, useMemo, useState } from "react";
import Header from "./components/Header";
import UploadPanel from "./components/UploadPanel";
import RadarScanner from "./components/RadarScanner";
import ConfirmedList from "./components/ConfirmedList";
import PlanetDetails from "./components/PlanetDetails";
import { predictCsv } from "./lib/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [rowCount, setRowCount] = useState(0);
  const [results, setResults] = useState(null);
  const [visibleCount, setVisibleCount] = useState(0);
  const [selected, setSelected] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");

  const confirmed = useMemo(
    () => (results || []).filter((r) => r.prediction === "CONFIRMED").length,
    [results]
  );

  useEffect(() => {
    if (!scanning || !results) return;

    let cursor = 0;
    const total = results.length;
    const timer = setInterval(() => {
      cursor += Math.max(1, Math.ceil(total / 90));
      if (cursor >= total) {
        cursor = total;
        clearInterval(timer);
        setScanning(false);
      }
      setVisibleCount(cursor);
      setProgress((cursor / total) * 100);
    }, 45);

    return () => clearInterval(timer);
  }, [scanning, results]);

  const handleFile = async (nextFile) => {
    if (!nextFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please choose a CSV file.");
      return;
    }

    setFile(nextFile);
    setResults(null);
    setVisibleCount(0);
    setSelected(null);
    setProgress(0);
    setError("");

    try {
      const text = await nextFile.text();
      const lines = text
        .split(/\r?\n/)
        .filter((line) => line.trim() && !line.trim().startsWith("#"));
      setRowCount(Math.max(0, lines.length - 1));
    } catch {
      setRowCount(0);
      setError("Could not read this CSV.");
    }
  };

  const startDetection = async () => {
    if (!file) return;

    setError("");
    setScanning(true);
    setProgress(2);
    setVisibleCount(0);
    setSelected(null);

    try {
      const payload = await predictCsv(file);
      const predictions = payload.predictions || [];

      if (!predictions.length) {
        throw new Error("The backend returned no predictions.");
      }

      setResults(predictions);
    } catch (err) {
      setScanning(false);
      setProgress(0);
      setError(err.message || "Detection failed.");
    }
  };

  const clearFile = () => {
    setFile(null);
    setRowCount(0);
    setResults(null);
    setVisibleCount(0);
    setSelected(null);
    setProgress(0);
    setError("");
  };

  const selectedItem = results?.find((r) => r.index === selected) || null;

  return (
    <div className="app-shell" id="detect">
      <Header />

      <main className="workspace">
        <UploadPanel
          file={file}
          rowCount={rowCount}
          onFile={handleFile}
          onClear={clearFile}
          onStart={startDetection}
          disabled={scanning}
        />

        <RadarScanner
          results={results}
          visibleCount={visibleCount}
          scanning={scanning}
          progress={progress}
        />

        <ConfirmedList
          results={results}
          visibleCount={visibleCount}
          selected={selected}
          onSelect={setSelected}
        />
      </main>

      {error && <div className="error-toast">{error}</div>}

      {results && !scanning && selectedItem && (
        <div className="details-shell">
          <PlanetDetails item={selectedItem} />
        </div>
      )}

      {results && !scanning && !selectedItem && confirmed > 0 && (
        <div className="scan-complete">
          <span className="complete-dot" />
          Scan complete · {results.length.toLocaleString()} objects · {confirmed.toLocaleString()} confirmed
        </div>
      )}

      <section className="model-strip" id="about">
        <span>V9 FINAL</span>
        <span>35 FEATURES</span>
        <span>84.85% ACCURACY</span>
        <span>93.52% ROC-AUC</span>
        <span>FP FLAGS EXCLUDED</span>
      </section>

      <section className="sr-only" id="docs">
        Upload a Kepler-like CSV and run the V9 final classification model.
      </section>
    </div>
  );
}