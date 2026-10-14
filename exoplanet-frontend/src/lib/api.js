const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function predictCsv(file) {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body,
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || "The backend rejected the CSV.");
  }

  return payload;
}

export async function getHealth() {
  const response = await fetch(`${API_URL}/health`);
  if (!response.ok) throw new Error("Backend unavailable");
  return response.json();
}