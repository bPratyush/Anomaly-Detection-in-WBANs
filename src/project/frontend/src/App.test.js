import { render, screen } from '@testing-library/react';
import App from './App';

test('renders learn react link', () => {
  render(<App />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
// Mock the GoogleGenerativeAI library
jest.mock("@google/generative-ai", () => {
  return {
    GoogleGenerativeAI: jest.fn().mockImplementation(() => {
      return {
        getGenerativeModel: jest.fn().mockImplementation(() => ({
          generateContent: jest.fn().mockResolvedValue({
            response: {
              text: jest.fn().mockResolvedValue("This is a test AI response")
            }
          })
        })),
        listModels: jest.fn().mockResolvedValue({
          models: [{ name: "gemini-pro" }]
        })
      };
    })
  };
});

// Mock fetch for API calls
global.fetch = jest.fn();

describe('WBAN Anomaly Detection App', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    global.fetch.mockClear();
  });

  test('renders application title', () => {
    render(<App />);
    expect(screen.getByText(/WBAN Anomaly Detection/i)).toBeInTheDocument();
  });

  test('renders upload section', () => {
    render(<App />);
    expect(screen.getByText(/Upload Sensor Data/i)).toBeInTheDocument();
    expect(screen.getByText(/Choose CSV file/i)).toBeInTheDocument();
  });

  test('renders chat section', () => {
    render(<App />);
    expect(screen.getByText(/Ask Gemini/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Ask something about anomaly detection/i)).toBeInTheDocument();
  });

  test('file selection updates state', () => {
    render(<App />);
    const file = new File(['test data'], 'test.csv', { type: 'text/csv' });
    const fileInput = screen.getByLabelText(/Choose CSV file/i);
    
    // Mock FileList
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });
    
    fireEvent.change(fileInput);
    
    expect(screen.getByText('test.csv')).toBeInTheDocument();
  });

  test('upload button is disabled without file', () => {
    render(<App />);
    const uploadButton = screen.getByText(/Analyze Data/i);
    expect(uploadButton).toBeDisabled();
  });

  test('successful file upload shows results', async () => {
    // Mock successful API response
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ 
        summary: "Normal", 
        types: [] 
      })
    });

    render(<App />);
    
    // Set file
    const file = new File(['test data'], 'test.csv', { type: 'text/csv' });
    const fileInput = screen.getByLabelText(/Choose CSV file/i);
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });
    fireEvent.change(fileInput);
    
    // Click upload button
    const uploadButton = screen.getByText(/Analyze Data/i);
    fireEvent.click(uploadButton);
    
    // Wait for results to appear
    await waitFor(() => {
      expect(screen.getByText(/Analysis Result/i)).toBeInTheDocument();
      expect(screen.getByText(/Normal/i)).toBeInTheDocument();
    });
  });

  test('failed file upload shows error message', async () => {
    // Mock failed API response
    global.fetch.mockResolvedValueOnce({
      ok: false,
      text: async () => "Server error"
    });

    render(<App />);
    
    // Set file
    const file = new File(['test data'], 'test.csv', { type: 'text/csv' });
    const fileInput = screen.getByLabelText(/Choose CSV file/i);
    Object.defineProperty(fileInput, 'files', {
      value: [file]
    });
    fireEvent.change(fileInput);
    
    // Click upload button
    const uploadButton = screen.getByText(/Analyze Data/i);
    fireEvent.click(uploadButton);
    
    // Wait for error message
    await waitFor(() => {
      expect(screen.getByText(/Server error/i)).toBeInTheDocument();
    });
  });

  test('shows anomalous results correctly', async () => {
    // Mock anomalous result
    global.fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ 
        summary: "Anomalous", 
        types: ["Temperature", "Heart Rate"] 
      })
    });

    render(<App />);
    
    // Set file and upload
    const file = new File(['test data'], 'test.csv', { type: 'text/csv' });
    const fileInput = screen.getByLabelText(/Choose CSV file/i);
    Object.defineProperty(fileInput, 'files', { value: [file] });
    fireEvent.change(fileInput);
    
    const uploadButton = screen.getByText(/Analyze Data/i);
    fireEvent.click(uploadButton);
    
    // Check results
    await waitFor(() => {
      expect(screen.getByText(/Anomalous/i)).toBeInTheDocument();
      expect(screen.getByText(/Temperature/i)).toBeInTheDocument();
      expect(screen.getByText(/Heart Rate/i)).toBeInTheDocument();
    });
  });

  test('chat input sends message and displays response', async () => {
    render(<App />);
    
    // Type message
    const chatInput = screen.getByPlaceholderText(/Ask something about anomaly detection/i);
    fireEvent.change(chatInput, { target: { value: 'What is an anomaly?' } });
    
    // Send message
    const sendButton = screen.getByText('→');
    fireEvent.click(sendButton);
    
    // Check message appears in chat
    expect(screen.getByText('What is an anomaly?')).toBeInTheDocument();
    
    // Check for response (after async operation)
    await waitFor(() => {
      expect(screen.getByText('This is a test AI response')).toBeInTheDocument();
    });
  });

  test('quota limit displays reset message', async () => {
    // Mock quota error
    const quotaError = new Error("Quota exceeded");
    quotaError.message = "quota exceeded";
    
    global.fetch.mockRejectedValueOnce(quotaError);

    render(<App />);
    
    // Set file and upload to trigger quota error
    const file = new File(['test data'], 'test.csv', { type: 'text/csv' });
    const fileInput = screen.getByLabelText(/Choose CSV file/i);
    Object.defineProperty(fileInput, 'files', { value: [file] });
    fireEvent.change(fileInput);
    
    const uploadButton = screen.getByText(/Analyze Data/i);
    fireEvent.click(uploadButton);
    
    // Check for quota message
    await waitFor(() => {
      expect(screen.getByText(/API quota limit reached/i)).toBeInTheDocument();
    });
  });

  test('pressing Enter in chat input sends message', async () => {
    render(<App />);
    
    // Type message
    const chatInput = screen.getByPlaceholderText(/Ask something about anomaly detection/i);
    fireEvent.change(chatInput, { target: { value: 'What is WBAN?' } });
    
    // Press Enter
    fireEvent.keyDown(chatInput, { key: 'Enter', code: 'Enter' });
    
    // Check message appears in chat
    expect(screen.getByText('What is WBAN?')).toBeInTheDocument();
    
    // Check for response (after async operation)
    await waitFor(() => {
      expect(screen.getByText('This is a test AI response')).toBeInTheDocument();
    });
  });
});