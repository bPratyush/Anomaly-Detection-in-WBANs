import React, { useState, useEffect, useRef } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";
import "./App.css";

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const API_KEY = process.env.REACT_APP_GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(API_KEY, {
  apiVersion: "v1"
});

const validateCSV = async (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const rows = text.split('\n');
      if (rows.length < 100) {
        reject(new Error("CSV file must contain at least 100 rows of data for accurate analysis."));
      } else {
        resolve(true);
      }
    };
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsText(file);
  });
};

const directGeminiRequest = async (prompt) => {
  try {
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

const tryMultipleModels = async (prompt) => {
  let availableModels = [];
  let modelPaths = [];
  
  try {
    const models = await genAI.listModels();
    console.log("Available models:", models);
    if (models && models.models) {
      modelPaths = models.models.map(m => m.name);
      availableModels = models.models.map(m => m.name.split('/').pop());
      console.log("Available model paths:", modelPaths);
      console.log("Available model names:", availableModels);
    }
  } catch (error) {
    console.error("Error listing models:", error);
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
  
  try {
    console.log("Attempting direct API call as last resort");
    return await directGeminiRequest(prompt);
  } catch (directErr) {
    console.error("Direct API call failed:", directErr);
    throw lastError || directErr || new Error("Failed to generate content with any method");
  }
};

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
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [showExamples, setShowExamples] = useState(false);
  const [validatingFile, setValidatingFile] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [fileTooSmallError, setFileTooSmallError] = useState(false);
  const [showPrivacyModal, setShowPrivacyModal] = useState(false);
  const [showTermsModal, setShowTermsModal] = useState(false); 
  const [showHelpModal, setShowHelpModal] = useState(false);
  const chatHistoryRef = useRef(null);
  const fileInputRef = useRef(null);
  const exampleQuestions = [
    "Explain these results in simple terms",
    "What do these anomalies mean for my health?",
    "How serious are these results?",
    "What should I do about these results?",
    "How reliable is this analysis?"
  ];
  const handleFileSelect = async (e) => {
    const selectedFile = e.target.files ? e.target.files[0] : null;
    if (!selectedFile) return;
    
    setValidatingFile(true);
    setFileTooSmallError(false);
    setErrorMsg("");
    
    try {
      await validateCSV(selectedFile);
      setFile(selectedFile);
    } catch (error) {
      setFileTooSmallError(true);
      setErrorMsg(error.message);
      setFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } finally {
      setValidatingFile(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.csv')) {
        handleFileSelect({ target: { files: [droppedFile] } });
      } else {
        setErrorMsg("Please upload a CSV file.");
      }
    }
  };

  useEffect(() => {
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const parseGeminiError = (error) => {
    const errorMessage = error.message || "Unknown error";
    console.error("Gemini API error:", errorMessage);
    
    if (errorMessage.includes("quota") || errorMessage.includes("429")) {
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

const fetchLLMExplanation = async (summary, types) => {
  try {
    setExplanationLoading(true);
    console.log("Attempting to use Gemini API with key available:", !!API_KEY);
    
    if (!API_KEY) {
      throw new Error("Gemini API key is missing. Check your environment variables.");
    }
    const prompt = `You're providing analysis of WBAN (Wireless Body Area Network) sensor data. Your tone should be warm and conversational, but knowledgeable.

Analysis summary: ${summary}
${types && types.length > 0 ? `Patterns detected: ${types.join(', ')}` : 'No unusual patterns detected.'}

Provide a brief, natural explanation (3-4 sentences) that:
1. Explains what these results likely mean in everyday language
2. Mentions any detected patterns naturally in conversation
3. Suggests what might be good to know or do next

Your response should sound like a knowledgeable friend explaining results - conversational but informed. No bullet points, no stars, no formatting symbols. Just natural conversation.`;
    
    console.log("Explanation prompt:", prompt)
    try {
      console.log("Attempting direct API call for explanation");
      await sleep(500); // Small delay to avoid rate limiting
      const text = await directGeminiRequest(prompt);
      console.log("Successfully received explanation from direct API call");
      const finalResponse = `${text.trim()}\n\nNeed more details? Ask specific questions in the chat below.`;
      setLlmResponse(finalResponse);
    } catch (directErr) {
      console.error("Direct API call failed for explanation:", directErr);
      await sleep(1000);
      try {
        console.log("Falling back to tryMultipleModels for explanation");
        const text = await tryMultipleModels(prompt);
        console.log("Successfully received explanation from multiple models approach");

        const finalResponse = `${text.trim()}\n\n Need more details? Ask specific questions in the chat below.`;
        setLlmResponse(finalResponse);
      } catch (modelErr) {
        console.error("Multiple models approach failed for explanation:", modelErr);
        throw modelErr;
      }
    }
  } catch (err) {
    console.error("Gemini explanation error:", err);
    const error = parseGeminiError(err);
    
    if (error.isQuotaError) {
      setLlmResponse(`⚠️ ${error.message}\n\n${error.details}\n\nIn the meantime, here's a brief explanation:\n\n${getDefaultExplanation(summary, types)}\n\n_Ask for more details in the chat below._`);
    } else {
      setLlmResponse(`⚠️ AI explanation unavailable: ${error.message}\n\nTry using the chat feature below to ask about your results.`);
    }
  } finally {
    setExplanationLoading(false);
  }
};

  const askGemini = async () => {
    if (!chatInput.trim()) return;
    const newHistory = [...chatHistory, { role: "user", content: chatInput }];
    setChatHistory(newHistory);
    const question = chatInput;
    setChatInput("");
    setChatResponse("Thinking...");
    
    try {
      if (!API_KEY) {
        throw new Error("Gemini API key is missing. Check your environment variables.");
      }

      let contextualPrompt = question;

if (result) {
  contextualPrompt = `You're a knowledgeable assistant analyzing WBAN (Wireless Body Area Network) sensor data. Your tone should be conversational but informative.

The sensor data analysis shows:
- Overall assessment: ${result.summary}
- ${result.types && result.types.length > 0 ? `Patterns identified: ${result.types.join(', ')}` : 'No unusual patterns identified.'}
${llmResponse ? `- Earlier, you mentioned: ${llmResponse}` : ''}

The person just asked: "${question}"

Respond naturally as if you're having a conversation. Be helpful, clear, and approachable. Provide accurate information but avoid overly technical medical jargon unless necessary. Don't use formatting like bullet points or headers - your response should flow like natural conversation. Balance being informative with being reassuring.`;
} else {
 
  contextualPrompt = `You're a knowledgeable assistant specializing in WBAN (Wireless Body Area Network) technology and sensor data analysis. 

The person just asked: "${question}"

Respond in a natural, conversational way while sharing accurate information. Avoid technical jargon when possible and explain concepts clearly. Don't use formatting like bullet points or headers - your response should flow like a natural conversation. Be informative without being overly formal.`;
}
      
      console.log("Sending contextual prompt:", contextualPrompt);
      try {
        console.log("Attempting direct API call for chat");
        const text = await directGeminiRequest(contextualPrompt);
        setChatHistory([...newHistory, { role: "ai", content: text }]);
      } catch (directErr) {
        console.error("Direct API call failed for chat:", directErr);
        
        await sleep(1000);
        
        try {
          console.log("Falling back to tryMultipleModels for chat");
          const text = await tryMultipleModels(contextualPrompt);
          setChatHistory([...newHistory, { role: "ai", content: text }]);
        } catch (modelErr) {
          console.error("Multiple models approach failed for chat:", modelErr);
          throw modelErr;
        }
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

      setChatHistory([]);
      
      // Add a small delay before making the Gemini API call. This helps prevent rate limiting issues
      await sleep(1000);
      
      await fetchLLMExplanation(data.summary, data.types);
    } catch (error) {
      setErrorMsg(error.message || "Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      {/* App header with logo and title */}
      <div className="header">
        <div className="header-left">
          <div className="logo-icon"></div>
          <h1 className="heading">WBAN Anomaly Detection</h1>
        </div>
        <div className="gemini-powered">
          <span className="gemini-dot"></span>
          <span className="gemini-text">Powered by Gemini</span>
        </div>
      </div>

      {/* Quota limit message */}
      {quotaResetTime && (
        <div className="quota-message">
          <span className="quota-icon">⚠️</span> API quota limit reached. Try again after approximately: {quotaResetTime.toLocaleTimeString()}
        </div>
      )}

      <div className="main-content">
        {/* Upload card */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Upload Sensor Data</h2>
            <div className="card-subtitle">
              Upload your WBAN sensor data file containing at least 100 entries to detect anomalies in body sensor networks.
            </div>
          </div>
          
          {/* Drag and drop file upload area */}
          <div 
            className={dragActive ? "drag-drop-area drag-active" : "drag-drop-area"}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="file-input"
              id="file-upload"
              ref={fileInputRef}
              disabled={loading || validatingFile}
            />
            
            <div className="upload-inner">
              <div className="upload-icon">
                {file ? '📄' : '📊'}
              </div>
              
              <div className="upload-text">
                {validatingFile ? (
                  <div className="validating-text">
                    <div className="small-spinner"></div>
                    Validating file...
                  </div>
                ) : file ? (
                  <div className="file-info">
                    <div className="file-name">{file.name}</div>
                    <div className="file-size">{(file.size / 1024).toFixed(1)} KB</div>
                  </div>
                ) : (
                  <>
                    <div className="upload-title">Drag and drop your CSV file or</div>
                    <label htmlFor="file-upload" className="browse-button">
                      Browse files
                    </label>
                  </>
                )}

                {file && !loading && !validatingFile && (
                  <div className="file-actions">
                    <label htmlFor="file-upload" className="change-file-btn">
                      Change file
                    </label>
                    <button 
                      onClick={() => {
                        setFile(null);
                        if (fileInputRef.current) fileInputRef.current.value = "";
                      }}
                      className="remove-file-btn"
                    >
                      Remove
                    </button>
                  </div>
                )}
              </div>
            </div>
            
            {file && (
              <button 
                onClick={upload} 
                disabled={loading || validatingFile} 
                className={loading || validatingFile ? "button-disabled" : "analyze-button"}
              >
                {loading ? (
                  <div className="loading-container">
                    <div className="loading-spinner"></div>
                    <span>Analyzing...</span>
                  </div>
                ) : "Analyze Data"}
              </button>
            )}
          </div>
          
          {fileTooSmallError && (
            <div className="validation-error">
              <span className="error-icon">⚠️</span> {errorMsg}
            </div>
          )}
          
          {errorMsg && !fileTooSmallError && (
            <div className="error-msg">
              <span className="error-icon">❌</span> {errorMsg}
            </div>
          )}

          {/* Analysis result section */}
          {result && (
            <div className="result-section">
              <div className={result.summary === "Anomalous" ? "result-box-anomalous" : "result-box-normal"}>
                <div className="result-header">
                  <div className="result-header-left">
                    <h3 className="result-title">Analysis Result</h3>
                    <div className="result-timestamp">
                      {new Date().toLocaleString()}
                    </div>
                  </div>
                  <span className={result.summary === "Anomalous" ? "status-badge-error" : "status-badge-success"}>
                    {result.summary}
                  </span>
                </div>
                
                {result.types && result.types.length > 0 && (
                  <div className="anomaly-types">
                    {result.types.map((type, index) => (
                      <span key={index} className="anomaly-type">
                        <span className="anomaly-type-icon">⚠️</span> {type}
                      </span>
                    ))}
                  </div>
                )}
                
                <div className="llm-explanation">
                  <div className="explanation-header">
                    <h4 className="explanation-title">
                      <span className="explanation-icon">🔍</span> Gemini Analysis
                    </h4>
                    {explanationLoading && (
                      <div className="explanation-status">
                        <div className="small-spinner"></div>
                        <span>Analyzing...</span>
                      </div>
                    )}
                  </div>
                  
                  {llmResponse ? (
                    <div className={llmResponse.includes("⚠️") ? "explanation-error" : "explanation-text"}>
                      {llmResponse}
                    </div>
                  ) : (
                    <div className="loading-explanation">
                      {explanationLoading ? (
                        <>
                          <div className="loading-spinner"></div>
                          <div>
                            Requesting AI explanation...
                            <div style={{ marginTop: "8px", fontSize: "12px" }}>
                              This may take a moment. You can also use the chat below for questions.
                            </div>
                          </div>
                        </>
                      ) : (
                        "Waiting for analysis..."
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Visual cue to connect results to chat */}
          {result && llmResponse && (
            <div className="ask-about-results">
              <div className="ask-about-results-arrow"></div>
              <div className="ask-about-results-content">
                <span className="ask-about-results-icon">💡</span>
                <div>
                  <div className="ask-about-results-title">Have questions about your results?</div>
                  <div className="ask-about-results-subtitle">Use the chat below to ask Gemini about your specific analysis results</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Chat card */}
        <div className="chat-card">
          <div className="card-header">
            <div className="chat-header-top">
              <h2 className="card-title">Ask Gemini</h2>
              {result && <span className="context-badge">Results in context</span>}
            </div>
            <div className="card-subtitle">
              {result ? 
                "Ask questions about your analysis results or general anomaly detection topics." :
                "Have questions about anomaly detection? Ask Gemini for help."}
            </div>
          </div>
          
          {/* Example questions */}
          {result && (
            <div className="examples-section">
              <button 
                onClick={() => setShowExamples(!showExamples)}
                className="examples-button"
                aria-expanded={showExamples}
              >
                <span className="examples-button-icon">{showExamples ? "−" : "+"}</span>
                {showExamples ? "Hide Examples" : "Show Example Questions"}
              </button>
              
              {showExamples && (
                <div className="examples-list">
                  {exampleQuestions.map((q, index) => (
                    <div 
                      key={index} 
                      className="example-question"
                      onClick={() => setChatInput(q)}
                    >
                      {q}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          
          {/* Chat history */}
          <div className="chat-history" ref={chatHistoryRef}>
            {chatHistory.length > 0 ? (
              chatHistory.map((msg, index) => (
                <div 
                  key={index} 
                  className={msg.role === "user" ? "user-message" : "ai-message"}
                >
                  <div className="message-content">{msg.content}</div>
                </div>
              ))
            ) : (
              <div className="empty-chat-message">
                <div className="empty-chat-icon">💬</div>
                {result ? (
                  <p>Ask questions about your analysis results</p>
                ) : (
                  <p>Ask about WBAN Technology and Anomaly Detection</p>
                )}
              </div>
            )}
          </div>
          
          {/* Chat input area */}
          <div className="chat-input-area">
            <textarea
              placeholder={result ? 
                "Ask about your analysis results (e.g., 'Explain these results in simple terms')" : 
                "Ask something about anomaly detection..."}
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              className="chat-textarea"
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
              className={chatInput.trim() && quotaResetTime === null ? "send-button" : "send-button-disabled"}
              aria-label="Send message"
              title="Send message (or press Enter)"
            >
              <span className="send-icon">→</span>
            </button>
          </div>

          {/* Typing indicator */}
          {chatResponse && (
            <div className="typing-indicator">
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
              {chatResponse}
            </div>
          )}
        </div>
      </div>
      
      {/* Enhanced Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-section">
  <h3 className="footer-title">About</h3>
  <div className="footer-links">
    {/* Replace anchors with proper hrefs */}
    <a 
      href="https://en.wikipedia.org/wiki/Wireless_body_area_network"
      className="footer-link" 
      target="_blank"
      rel="noopener noreferrer"
    >
      About WBANs
    </a>
    <a 
      href="https://en.wikipedia.org/wiki/Anomaly_detection"
      className="footer-link" 
      target="_blank"
      rel="noopener noreferrer"
    >
      Anomaly Detection
    </a>
    <a 
      href="https://ai.google/discover/generativeai/"
      className="footer-link" 
      target="_blank"
      rel="noopener noreferrer"
    >
      About Gemini AI
    </a>
  </div>
</div>

<div className="footer-section">
  <h3 className="footer-title">Resources</h3>
  <div className="footer-links">
    {/* Replace with button for modal triggers */}
    <button 
      className="footer-link-button" 
      onClick={() => setShowHelpModal(true)}
    >
      Help Center
    </button>
  </div>
</div>

<div className="footer-section">
  <h3 className="footer-title">Legal</h3>
  <div className="footer-links">
    {/* Replace with buttons for modal triggers */}
    <button 
      className="footer-link-button" 
      onClick={() => setShowTermsModal(true)}
    >
      Terms of Service
    </button>
    <button 
      className="footer-link-button" 
      onClick={() => setShowPrivacyModal(true)}
    >
      Privacy Policy
    </button>
    <button 
      className="footer-link-button" 
      onClick={() => window.alert('© ' + new Date().getFullYear() + ' WBAN Anomaly Detection - All rights reserved')}
    >
      Copyright Information
    </button>
  </div>
</div>
          </div>
        
        <div className="footer-bottom">
          <div className="footer-copyright">
            © {new Date().getFullYear()} WBAN Anomaly Detection System. All rights reserved.
          </div>
          <div className="footer-made-with-love">
    Made with <span className="heart-icon">❤️</span> by Pratyush Bindal at BITS Pilani, Hyderabad Campus
  </div>
        </div>
      </footer>
      
      {/* Modal for Privacy Policy */}
      {showPrivacyModal && (
        <div className="modal-overlay" onClick={() => setShowPrivacyModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Privacy Policy</h2>
              <button className="modal-close" onClick={() => setShowPrivacyModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <p>Last updated: June 2025</p>
              
              <p>This Privacy Policy describes how the WBAN Anomaly Detection System ("we", "us", or "our") collects, uses, and discloses your information when you use our service.</p>
              
              <h3>Information We Collect</h3>
              <p>When you use our service, we may collect the following types of information:</p>
              <ul>
                <li><strong>CSV Data Files:</strong> We process the sensor data files you upload for analysis.</li>
                <li><strong>Chat Interactions:</strong> We store the conversations you have with our AI assistant.</li>
                <li><strong>Usage Data:</strong> We collect information on how you interact with our application.</li>
              </ul>
              
              <h3>How We Use Your Information</h3>
              <p>We use the information we collect to:</p>
              <ul>
                <li>Provide and maintain our service</li>
                <li>Detect and analyze anomalies in your uploaded data</li>
                <li>Improve our anomaly detection algorithms</li>
                <li>Respond to your queries through our AI assistant</li>
                <li>Monitor the usage of our service</li>
              </ul>
              
              <h3>Data Security</h3>
              <p>The security of your data is important to us. Your uploaded files are processed locally and not stored on our servers after analysis is complete. We implement appropriate security measures to protect against unauthorized access, alteration, disclosure, or destruction of your data.</p>
              
              <h3>Third-Party Services</h3>
              <p>We use Google's Gemini AI API to provide AI-powered explanations and chat functionality. Your interactions with the AI are subject to Google's privacy policies as well.</p>
              
              <h3>Changes to This Privacy Policy</h3>
              <p>We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page.</p>
              
              <h3>Contact Us</h3>
              <p>If you have any questions about this Privacy Policy, please contact us at privacy@wbananomaly.example.com</p>
            </div>
            <div className="modal-footer">
              <button className="modal-btn" onClick={() => setShowPrivacyModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
      
      {/* Modal for Terms of Service */}
      {showTermsModal && (
        <div className="modal-overlay" onClick={() => setShowTermsModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Terms of Service</h2>
              <button className="modal-close" onClick={() => setShowTermsModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <p>Last updated: June 2025</p>
              
              <p>Please read these Terms of Service carefully before using the WBAN Anomaly Detection System.</p>
              
              <h3>Acceptance of Terms</h3>
              <p>By accessing or using our service, you agree to be bound by these Terms. If you disagree with any part of the terms, you may not access the service.</p>
              
              <h3>Use of Service</h3>
              <p>Our service provides anomaly detection for Wireless Body Area Network (WBAN) sensor data. You may upload CSV files containing sensor data for analysis. You agree to use the service only for lawful purposes and in accordance with these Terms.</p>
              
              <h3>Medical Disclaimer</h3>
              <p>The WBAN Anomaly Detection System is not a medical device and is not intended for diagnosis, treatment, or prevention of any disease or health condition. The analysis and information provided should not be considered medical advice. Always consult with qualified healthcare professionals regarding any health concerns.</p>
              
              <h3>User Data</h3>
              <p>You retain all rights to the data you upload. We do not claim ownership of your uploaded files or the content of your interactions with our AI assistant.</p>
              
              <h3>Limitation of Liability</h3>
              <p>To the maximum extent permitted by law, we shall not be liable for any indirect, incidental, special, consequential, or punitive damages, or any loss of profits or revenues, whether incurred directly or indirectly, or any loss of data, use, goodwill, or other intangible losses resulting from your use of our service.</p>
              
              <h3>Changes to Terms</h3>
              <p>We reserve the right to modify or replace these Terms at any time. If a revision is material, we will provide at least 30 days' notice prior to any new terms taking effect.</p>
              
              <h3>Contact Us</h3>
              <p>If you have any questions about these Terms, please contact us</p>
            </div>
            <div className="modal-footer">
              <button className="modal-btn" onClick={() => setShowTermsModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
      {/* Modal for Help */}
      {showHelpModal && (
        <div className="modal-overlay" onClick={() => setShowHelpModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 className="modal-title">Help Center</h2>
              <button className="modal-close" onClick={() => setShowHelpModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <h3>Getting Started</h3>
              <p>The WBAN Anomaly Detection System helps you analyze sensor data from Wireless Body Area Networks to identify potential anomalies.</p>
              
              <h3>How to Use</h3>
              <ol>
                <li><strong>Upload Data:</strong> Drag and drop a CSV file or click "Browse files" to upload your sensor data.</li>
                <li><strong>Analyze:</strong> Click the "Analyze Data" button to process your file.</li>
                <li><strong>Review Results:</strong> The system will display whether your data is normal or contains anomalies.</li>
                <li><strong>Ask Questions:</strong> Use the chat feature to ask Gemini AI about your results or general questions about WBAN technology.</li>
              </ol>
              
              <h3>CSV File Format</h3>
              <p>Your CSV file should contain at least 100 rows of sensor data. Each row represents a timestamp, and columns should contain sensor readings from different nodes in your WBAN setup.</p>
              
              <h3>Understanding Results</h3>
              <p>After analysis, you'll see one of two results:</p>
              <ul>
                <li><strong>Normal:</strong> No unusual patterns detected in your sensor data.</li>
                <li><strong>Anomalous:</strong> Unusual patterns detected that may indicate potential issues. The specific types of anomalies detected will be listed.</li>
              </ul>
              
              <h3>Frequently Asked Questions</h3>
              <p><strong>Q: What is a WBAN?</strong><br/>
              A: A Wireless Body Area Network (WBAN) is a network of wearable computing devices that monitor body functions and the surrounding environment.</p>
              
              <p><strong>Q: How accurate is the anomaly detection?</strong><br/>
              A: Our system uses advanced machine learning algorithms to detect anomalies with high accuracy, but results should always be verified by healthcare professionals.</p>
            
              
              <h3>Contact Support</h3>
              <p>If you need further assistance, please contact us</p>
            </div>
            <div className="modal-footer">
              <button className="modal-btn" onClick={() => setShowHelpModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;




