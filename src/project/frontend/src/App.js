import React, { useState } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";

// Fix API key access
const API_KEY = process.env.REACT_APP_GEMINI_API_KEY;

// Initialize with explicit API version (v1 instead of v1beta)
const genAI = new GoogleGenerativeAI(API_KEY, {
  apiVersion: "v1" // Use stable v1 instead of beta
});

// Direct API call as fallback approach
const directGeminiRequest = async (prompt) => {
  try {
    // Direct API call as absolute last resort
    const response = await fetch('https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': API_KEY,
      },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.4,
          topP: 0.8,
          maxOutputTokens: 1024,
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`Direct API call failed: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.candidates[0]?.content?.parts[0]?.text || "No response text available";
  } catch (err) {
    console.error("Direct Gemini API call failed:", err);
    throw err;
  }
};

// Update the tryMultipleModels function
const tryMultipleModels = async (prompt) => {
  // We'll determine which models are available first
  let availableModels = [];
  let modelPaths = [];
  
  try {
    const models = await genAI.listModels();
    console.log("Available models:", models);
    
    // Store full model paths
    if (models && models.models) {
      modelPaths = models.models.map(m => m.name);
      availableModels = models.models.map(m => m.name.split('/').pop());
      console.log("Available model paths:", modelPaths);
      console.log("Available model names:", availableModels);
    }
  } catch (error) {
    console.error("Error listing models:", error);
    // If we can't list models, try these common formats with v1 path
    modelPaths = [
      "gemini-pro",
      "gemini-1.0-pro",
      "gemini-1.5-flash",
      "gemini-1.5-pro",
      "models/gemini-pro",
      "models/gemini-1.0-pro",
      "models/gemini-1.5-flash", 
      "models/gemini-1.5-pro"
    ];
  }
  
  let lastError = null;
  
  // First try the full paths returned by the API
  for (const modelPath of modelPaths) {
    try {
      console.log(`Attempting to use model path: ${modelPath}`);
      const model = genAI.getGenerativeModel({ 
        model: modelPath,
        generationConfig: {
          temperature: 0.4,
          topP: 0.8,
          maxOutputTokens: 1024,
        }
      });
      const result = await model.generateContent(prompt);
      const response = await result.response;
      const text = await response.text();
      console.log(`Success with model: ${modelPath}`);
      return text;
    } catch (err) {
      console.error(`Error with model ${modelPath}:`, err.message);
      lastError = err;
    }
  }
  
  // Try direct API call as last resort
  try {
    console.log("Attempting direct API call as last resort");
    return await directGeminiRequest(prompt);
  } catch (directErr) {
    console.error("Direct API call failed:", directErr);
    // If we get here, all methods failed
    throw lastError || directErr || new Error("Failed to generate content with any method");
  }
};

// List available models for debugging
const listAvailableModels = async () => {
  try {
    const models = await genAI.listModels();
    console.log("Available models:", models);
    return models;
  } catch (error) {
    console.error("Error listing models:", error);
    return [];
  }
};

// Call this once when the app loads
console.log("Attempting to list available models...");
listAvailableModels();

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [llmResponse, setLlmResponse] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [chatResponse, setChatResponse] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [quotaResetTime, setQuotaResetTime] = useState(null);

  // Parse API errors to provide user-friendly messages
  const parseGeminiError = (error) => {
    const errorMessage = error.message || "Unknown error";
    console.error("Gemini API error:", errorMessage);
    
    if (errorMessage.includes("quota") || errorMessage.includes("429")) {
      // Set quota reset time to 60 minutes from now (approximate)
      setQuotaResetTime(new Date(Date.now() + 60 * 60 * 1000));
      
      return {
        isQuotaError: true,
        message: "Gemini API quota limit reached. Please try again later.",
        details: "The free tier of Gemini API has limits on requests per minute and per day."
      };
    } else if (errorMessage.includes("404") || errorMessage.includes("not found")) {
      return {
        isQuotaError: false,
        message: "Gemini API model not available.",
        details: "The requested model is not available with your current API key or region."
      };
    } else if (errorMessage.includes("403") || errorMessage.includes("permission")) {
      return {
        isQuotaError: false,
        message: "API key permission issue.",
        details: "Your API key may not have permission to use this model or service."
      };
    }
    
    return {
      isQuotaError: false,
      message: errorMessage,
      details: "There was an issue connecting to the Gemini API."
    };
  };

  // Provide default explanations when API is unavailable
  const getDefaultExplanation = (summary, types) => {
    if (summary === "Normal") {
      return "Your sensor data appears to be within normal parameters. No anomalies were detected in the body sensor network data. This suggests that the monitored physiological signals are following expected patterns.";
    } else {
      const typeText = types && types.length > 0 
        ? `The system detected the following types of anomalies: ${types.join(', ')}. `
        : "";
      
      return `Your sensor data shows anomalous patterns that differ from normal baseline activity. ${typeText}These deviations could indicate potential issues that might require attention. Consider consulting with a healthcare professional for further evaluation.`;
    }
  };

  // Update the model references in fetchLLMExplanation
  const fetchLLMExplanation = async (summary, types) => {
    try {
      console.log("Attempting to use Gemini API with key available:", !!API_KEY);
      
      if (!API_KEY) {
        throw new Error("Gemini API key is missing. Check your environment variables.");
      }
      
      const prompt = `
You are a medical AI assistant analyzing human sensor signals. 
Please explain the following anomaly detection result in clear, user-friendly terms.

Summary: ${summary}
Types: ${types && types.length > 0 ? types.join(', ') : 'None'}

Include relevant advice if appropriate.
`;
      
      // Try multiple models with fallback
      try {
        const text = await tryMultipleModels(prompt);
        setLlmResponse(text);
      } catch (modelErr) {
        // Fallback to direct API call if model finder fails
        const text = await directGeminiRequest(prompt);
        setLlmResponse(text);
      }
    } catch (err) {
      console.error("Gemini explanation error:", err);
      const error = parseGeminiError(err);
      
      if (error.isQuotaError) {
        // Provide fallback explanation when quota is exceeded
        setLlmResponse(`⚠️ ${error.message}\n\n${error.details}\n\nIn the meantime, here's a standard explanation:\n\n${getDefaultExplanation(summary, types)}`);
      } else {
        setLlmResponse(`⚠️ AI explanation unavailable: ${error.message}`);
      }
    }
  };

  // Update the model references in askGemini
  const askGemini = async () => {
    if (!chatInput.trim()) return;
    
    // Add user question to chat history
    const newHistory = [...chatHistory, { role: "user", content: chatInput }];
    setChatHistory(newHistory);
    
    // Clear input and show loading state
    const question = chatInput;
    setChatInput("");
    setChatResponse("Thinking...");
    
    try {
      if (!API_KEY) {
        throw new Error("Gemini API key is missing. Check your environment variables.");
      }
      
      // Try multiple models with fallback
      try {
        const text = await tryMultipleModels(question);
        setChatHistory([...newHistory, { role: "ai", content: text }]);
      } catch (modelErr) {
        // Fallback to direct API call if model finder fails
        const text = await directGeminiRequest(question);
        setChatHistory([...newHistory, { role: "ai", content: text }]);
      }
      
      setChatResponse("");
    } catch (err) {
      console.error("Gemini chat error:", err);
      const error = parseGeminiError(err);
      
      setChatHistory([...newHistory, { 
        role: "ai", 
        content: error.isQuotaError 
          ? `⚠️ ${error.message}\n\n${error.details}\n\nPlease try again later.` 
          : `⚠️ Unable to answer: ${error.message}`
      }]);
      setChatResponse("");
    }
  };

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    setErrorMsg("");
    setResult(null);
    setLlmResponse("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      setResult(data);
      await fetchLLMExplanation(data.summary, data.types);
    } catch (error) {
      setErrorMsg(error.message || "Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.heading}>WBAN Anomaly Detection</h1>
        <div style={styles.logo}>
          <span style={styles.geminiDot}></span>
          <span style={styles.geminiText}>Powered by Gemini</span>
        </div>
      </div>

      {quotaResetTime && (
        <div style={styles.quotaMessage}>
          API quota limit reached. Try again after approximately: {quotaResetTime.toLocaleTimeString()}
        </div>
      )}

      <div style={styles.mainContent}>
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>Upload Sensor Data</h2>
          <p style={styles.cardDescription}>
            Upload your WBAN sensor data file to detect anomalies in body sensor networks.
          </p>
          
          <div style={styles.uploadSection}>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files[0])}
              style={styles.fileInput}
              id="file-upload"
              className="hidden"
            />
            <label htmlFor="file-upload" style={styles.fileInputLabel}>
              {file ? file.name : "Choose CSV file"}
            </label>
            
            <button 
              onClick={upload} 
              disabled={!file || loading} 
              style={file && !loading ? styles.buttonPrimary : styles.buttonDisabled}
            >
              {loading ? "Analyzing..." : "Analyze Data"}
            </button>
          </div>

          {errorMsg && <div style={styles.errorMsg}>{errorMsg}</div>}

          {result && (
            <div style={result.summary === "Anomalous" ? styles.resultBoxAnomalous : styles.resultBoxNormal}>
              <div style={styles.resultHeader}>
                <h3 style={styles.resultTitle}>Analysis Result</h3>
                <span style={result.summary === "Anomalous" ? styles.statusBadgeError : styles.statusBadgeSuccess}>
                  {result.summary}
                </span>
              </div>
              
              {result.types && result.types.length > 0 && (
                <div style={styles.anomalyTypes}>
                  {result.types.map((type, index) => (
                    <span key={index} style={styles.anomalyType}>{type}</span>
                  ))}
                </div>
              )}
              
              <div style={styles.llmExplanation}>
                <h4 style={styles.explanationTitle}>Gemini Analysis</h4>
                {llmResponse ? (
                  <p style={styles.explanationText}>{llmResponse}</p>
                ) : (
                  <div style={styles.loadingExplanation}>Requesting AI explanation...</div>
                )}
              </div>
            </div>
          )}
        </div>

        <div style={styles.chatCard}>
          <h2 style={styles.cardTitle}>Ask Gemini</h2>
          <p style={styles.cardDescription}>
            Have questions about your results or anomaly detection? Ask Gemini for help.
          </p>
          
          <div style={styles.chatHistory}>
            {chatHistory.map((msg, index) => (
              <div 
                key={index} 
                style={msg.role === "user" ? styles.userMessage : styles.aiMessage}
              >
                <div style={styles.messageContent}>{msg.content}</div>
              </div>
            ))}
          </div>
          
          <div style={styles.chatInputArea}>
            <textarea
              placeholder="Ask something about anomaly detection..."
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              style={styles.chatTextarea}
              rows={2}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  askGemini();
                }
              }}
            />
            <button 
              onClick={askGemini} 
              disabled={!chatInput.trim() || quotaResetTime !== null} 
              style={chatInput.trim() && quotaResetTime === null ? styles.sendButton : styles.sendButtonDisabled}
            >
              <span style={styles.sendIcon}>→</span>
            </button>
          </div>
          {chatResponse && <div style={styles.typingIndicator}>{chatResponse}</div>}
        </div>
      </div>
    </div>
  );
}

// Modern Gemini-inspired styling
const styles = {
  container: {
    fontFamily: "'Google Sans', 'Roboto', sans-serif",
    backgroundColor: "#f8f9fa",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    padding: "20px",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "16px 0",
    marginBottom: "24px",
  },
  heading: {
    fontSize: "28px",
    fontWeight: "500",
    color: "#202124",
    margin: 0,
  },
  logo: {
    display: "flex",
    alignItems: "center",
  },
  geminiDot: {
    width: "16px",
    height: "16px",
    borderRadius: "50%",
    background: "linear-gradient(135deg, #8e44ad, #1a73e8)",
    marginRight: "8px",
  },
  geminiText: {
    color: "#5f6368",
    fontSize: "14px",
    fontWeight: "500",
  },
  quotaMessage: {
    backgroundColor: "#fff8e1",
    color: "#bf360c",
    padding: "12px 16px",
    borderRadius: "8px",
    fontSize: "14px",
    marginBottom: "16px",
    border: "1px solid #ffe082",
    maxWidth: "1200px",
    margin: "0 auto 24px auto",
    width: "100%",
    textAlign: "center",
    fontWeight: "500",
  },
  mainContent: {
    display: "flex",
    flexDirection: "column",
    gap: "24px",
    maxWidth: "1200px",
    margin: "0 auto",
    width: "100%",
  },
  card: {
    backgroundColor: "white",
    borderRadius: "16px",
    padding: "32px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)",
  },
  cardTitle: {
    fontSize: "24px",
    fontWeight: "500",
    margin: "0 0 8px 0",
    color: "#202124",
  },
  cardDescription: {
    fontSize: "16px",
    color: "#5f6368",
    marginBottom: "24px",
    lineHeight: "1.5",
  },
  uploadSection: {
    display: "flex",
    flexDirection: "column",
    gap: "16px",
    marginBottom: "24px",
  },
  fileInput: {
    display: "none",
  },
  fileInputLabel: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "12px 16px",
    backgroundColor: "#f1f3f4",
    color: "#202124",
    borderRadius: "8px",
    cursor: "pointer",
    fontWeight: "500",
    border: "1px solid #dadce0",
    transition: "all 0.2s ease",
  },
  buttonPrimary: {
    backgroundColor: "#1a73e8",
    color: "white",
    border: "none",
    borderRadius: "8px",
    padding: "12px 24px",
    fontSize: "16px",
    fontWeight: "500",
    cursor: "pointer",
    transition: "all 0.2s ease",
  },
  buttonDisabled: {
    backgroundColor: "#dadce0",
    color: "#5f6368",
    border: "none",
    borderRadius: "8px",
    padding: "12px 24px",
    fontSize: "16px",
    fontWeight: "500",
    cursor: "not-allowed",
  },
  errorMsg: {
    backgroundColor: "#fdeded",
    color: "#d93025",
    padding: "12px 16px",
    borderRadius: "8px",
    marginBottom: "16px",
    fontWeight: "500",
  },
  resultBoxNormal: {
    backgroundColor: "#e6f4ea",
    borderRadius: "12px",
    padding: "24px",
    border: "1px solid #ceead6",
  },
  resultBoxAnomalous: {
    backgroundColor: "#fce8e6",
    borderRadius: "12px",
    padding: "24px",
    border: "1px solid #f6bbb8",
  },
  resultHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "16px",
  },
  resultTitle: {
    fontSize: "20px",
    fontWeight: "500",
    margin: 0,
    color: "#202124",
  },
  statusBadgeSuccess: {
    backgroundColor: "#137333",
    color: "white",
    padding: "4px 12px",
    borderRadius: "16px",
    fontSize: "14px",
    fontWeight: "500",
  },
  statusBadgeError: {
    backgroundColor: "#c5221f",
    color: "white",
    padding: "4px 12px",
    borderRadius: "16px",
    fontSize: "14px",
    fontWeight: "500",
  },
  anomalyTypes: {
    display: "flex",
    flexWrap: "wrap",
    gap: "8px",
    marginBottom: "16px",
  },
  anomalyType: {
    backgroundColor: "rgba(0,0,0,0.08)",
    padding: "4px 12px",
    borderRadius: "16px",
    fontSize: "14px",
    color: "#202124",
  },
  llmExplanation: {
    backgroundColor: "white",
    borderRadius: "8px",
    padding: "16px",
    marginTop: "16px",
  },
  explanationTitle: {
    fontSize: "16px",
    fontWeight: "500",
    marginTop: 0,
    marginBottom: "8px",
    color: "#202124",
  },
  explanationText: {
    fontSize: "15px",
    lineHeight: "1.5",
    color: "#202124",
    margin: 0,
    whiteSpace: "pre-wrap",
  },
  loadingExplanation: {
    color: "#5f6368",
    fontStyle: "italic",
  },
  chatCard: {
    backgroundColor: "white",
    borderRadius: "16px",
    padding: "32px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)",
  },
  chatHistory: {
    display: "flex",
    flexDirection: "column",
    gap: "16px",
    marginBottom: "24px",
    maxHeight: "400px",
    overflowY: "auto",
    padding: "8px 0",
  },
  userMessage: {
    alignSelf: "flex-end",
    backgroundColor: "#e8f0fe",
    borderRadius: "18px 18px 4px 18px",
    padding: "12px 16px",
    maxWidth: "80%",
  },
  aiMessage: {
    alignSelf: "flex-start",
    backgroundColor: "#f1f3f4",
    borderRadius: "18px 18px 18px 4px",
    padding: "12px 16px",
    maxWidth: "80%",
  },
  messageContent: {
    fontSize: "15px",
    lineHeight: "1.5",
    whiteSpace: "pre-wrap",
  },
  chatInputArea: {
    display: "flex",
    gap: "12px",
    alignItems: "flex-end",
    position: "relative",
  },
  chatTextarea: {
    flex: 1,
    padding: "12px 16px",
    borderRadius: "24px",
    border: "1px solid #dadce0",
    fontSize: "15px",
    resize: "none",
    outline: "none",
    fontFamily: "inherit",
  },
  sendButton: {
    backgroundColor: "#1a73e8",
    color: "white",
    border: "none",
    borderRadius: "50%",
    width: "40px",
    height: "40px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
  },
  sendButtonDisabled: {
    backgroundColor: "#dadce0",
    color: "#5f6368",
    border: "none",
    borderRadius: "50%",
    width: "40px",
    height: "40px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "not-allowed",
  },
  sendIcon: {
    fontSize: "20px",
    fontWeight: "bold",
  },
  typingIndicator: {
    color: "#5f6368",
    fontSize: "14px",
    fontStyle: "italic",
    marginTop: "8px",
  },
};

export default App;