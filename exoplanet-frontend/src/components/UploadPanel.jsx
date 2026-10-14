import { FileUp, X, Play } from "lucide-react";

export default function UploadPanel({
  file,
  rowCount,
  onFile,
  onClear,
  onStart,
  disabled,
}) {
  const handleDrop = (event) => {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0];
    if (dropped) onFile(dropped);
  };

  return (
    <section className="panel upload-panel">
      <div className="panel-title">1. Upload CSV</div>

      <label
        className={`dropzone ${file ? "has-file" : ""}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <FileUp size={28} strokeWidth={1.4} />
        <span className="drop-main">
          {file ? file.name : "Drag & drop your CSV"}
        </span>
        <span className="drop-sub">
          {file ? `${rowCount.toLocaleString()} rows detected` : "or click to browse"}
        </span>

        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />

        {!file && <span className="choose-button">Choose File</span>}
      </label>

      {file && (
        <div className="file-chip">
          <div>
            <strong>{file.name}</strong>
            <span>{rowCount.toLocaleString()} rows</span>
          </div>
          <button onClick={onClear} aria-label="Remove file">
            <X size={17} />
          </button>
        </div>
      )}

      <button
        className="primary-button"
        disabled={!file || disabled}
        onClick={onStart}
      >
        {disabled ? "Scanning..." : "Start Detection"}
        <Play size={16} fill="currentColor" />
      </button>
    </section>
  );
}