import React, { useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    setErrorMsg("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Server Error (${res.status}): ${text}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (error) {
      setErrorMsg(error.message || "Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>🧠 WBAN Anomaly Detection Assistant</h1>

      <div style={styles.card}>
        <label style={styles.label}>🧾 Upload your sensor CSV file:</label>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
          style={styles.input}
        />
        {file && <p style={styles.fileInfo}>📂 Selected: {file.name}</p>}

        <button
          onClick={upload}
          disabled={!file || loading}
          style={styles.button}
        >
          {loading ? "⏳ Analyzing..." : "📤 Upload & Analyze"}
        </button>

        {errorMsg && <p style={styles.error}>❌ {errorMsg}</p>}

        {result && (
          <div style={styles.resultBox}>
            <h3 style={styles.resultHeading}>📊 Analysis Result</h3>
            <p style={styles.resultLine}>
              <strong>Status:</strong>{" "}
              <span
                style={{ color: result.summary === "Anomalous" ? "#dc2626" : "#16a34a" }}
              >
                {result.summary}
              </span>
            </p>

            {result.explanation && (
              <p style={styles.resultExplain}>{result.explanation}</p>
            )}

            {result.timeline && (
              <Bar
                data={{
                  labels: result.timeline.map((_, i) => `t${i}`),
                  datasets: [
                    {
                      label: "Anomaly Score",
                      data: result.timeline,
                      backgroundColor: "rgba(75,192,192,0.6)",
                    },
                  ],
                }}
                options={{
                  responsive: true,
                  plugins: {
                    legend: { position: "top" },
                    title: { display: true, text: "Anomaly Score Over Time" },
                  },
                }}
              />
            )}
          </div>
        )}
      </div>

      <div style={styles.footer}>
        🤖 Ask: <i>“What happened to my body signals?”</i> in the chatbot below.
      </div>
    </div>
  );
}

const styles = {
  container: {
    fontFamily: "Segoe UI, sans-serif",
    padding: 40,
    background: "#f3f4f6",
    minHeight: "100vh",
  },
  heading: {
    textAlign: "center",
    fontSize: 28,
    marginBottom: 30,
    color: "#1f2937",
  },
  card: {
    background: "#fff",
    maxWidth: 700,
    margin: "0 auto",
    padding: 30,
    borderRadius: 12,
    boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
  },
  label: {
    fontWeight: "bold",
    fontSize: 16,
    marginBottom: 10,
  },
  input: {
    fontSize: 15,
    marginBottom: 10,
  },
  fileInfo: {
    fontSize: 14,
    color: "#4b5563",
    marginBottom: 10,
  },
  button: {
    padding: "10px 25px",
    fontSize: 16,
    background: "#4f46e5",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
    marginTop: 10,
  },
  error: {
    color: "#dc2626",
    marginTop: 15,
    fontWeight: "bold",
  },
  resultBox: {
    marginTop: 30,
    background: "#ecfccb",
    padding: 20,
    borderRadius: 10,
    border: "1px solid #84cc16",
  },
  resultHeading: {
    fontSize: 18,
    fontWeight: 600,
    marginBottom: 10,
  },
  resultLine: {
    fontSize: 16,
    color: "#1f2937",
  },
  resultExplain: {
    fontStyle: "italic",
    fontSize: 15,
    marginTop: 10,
    color: "#374151",
  },
  footer: {
    marginTop: 40,
    textAlign: "center",
    fontSize: 14,
    color: "#4b5563",
  },
};

export default App;
