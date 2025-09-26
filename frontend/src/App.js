import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { text: 'Hello! I am your RAG chatbot. Ask me anything about your documents!', sender: 'bot' }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [modelProvider, setModelProvider] = useState('openai');
  const messagesEndRef = useRef(null);

  // Scroll to bottom of chat
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check if the knowledge base is initialized
  useEffect(() => {
    const checkInitialization = async () => {
      try {
        // This is a simple check - in a real app, you might want a dedicated endpoint for this
        const response = await axios.get('/');
        setIsInitialized(true);
      } catch (error) {
        console.log('Knowledge base not initialized yet');
      }
    };
    checkInitialization();
  }, []);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = { text: inputMessage, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('/chat/', {
        message: inputMessage
      });

      const botMessage = {
        text: response.data.response,
        sender: 'bot',
        sources: response.data.sources || []
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { 
        text: 'Sorry, I encountered an error. Please try again later.', 
        sender: 'bot',
        isError: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);
    setIsUploading(true);

    try {
      await axios.post('/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // After uploading, initialize the knowledge base
      await initializeKnowledgeBase();
      alert('Document uploaded and processed successfully!');
      setSelectedFile(null);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const initializeKnowledgeBase = async () => {
    try {
      await axios.post('/initialize/');
      setIsInitialized(true);
      setMessages(prev => [...prev, { 
        text: 'Knowledge base initialized! You can now ask questions about your documents.', 
        sender: 'bot' 
      }]);
    } catch (error) {
      console.error('Error initializing knowledge base:', error);
      alert('Error initializing knowledge base. Please check the console for details.');
    }
  };

  const switchModelProvider = async (provider) => {
    try {
      await axios.post('/switch-model/', { provider });
      setModelProvider(provider);
      setMessages(prev => [...prev, { 
        text: `Switched to ${provider} model. You can now chat with the new model.`,
        sender: 'bot'
      }]);
    } catch (error) {
      console.error('Error switching model:', error);
      alert(`Error switching to ${provider} model. Please check the console for details.`);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>RAG Chatbot</h1>
        <div className="model-switcher">
          <button 
            className={`model-btn ${modelProvider === 'openai' ? 'active' : ''}`}
            onClick={() => switchModelProvider('openai')}
            disabled={isLoading}
          >
            OpenAI
          </button>
          <button 
            className={`model-btn ${modelProvider === 'ollama' ? 'active' : ''}`}
            onClick={() => switchModelProvider('ollama')}
            disabled={isLoading}
          >
            Ollama
          </button>
        </div>
      </header>

      <div className="chat-container">
        <div className="messages">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              <div className="message-content">
                {message.text}
                {message.sources && message.sources.length > 0 && (
                  <div className="sources">
                    <p>Sources:</p>
                    <ul>
                      {message.sources.map((source, i) => (
                        <li key={i}>
                          {source.metadata?.source || 'Document'} (Page {source.metadata?.page || 'N/A'})
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
        ))}
        <div ref={messagesEndRef} />
        </div>

        {!isInitialized && (
          <div className="initialization-prompt">
            <p>Please upload and process a document to start chatting.</p>
            <div className="file-upload">
              <input type="file" onChange={handleFileChange} disabled={isUploading} />
              <button 
                onClick={handleUpload} 
                disabled={!selectedFile || isUploading}
              >
                {isUploading ? 'Uploading...' : 'Upload & Process'}
              </button>
            </div>
          </div>
        )}

        <form onSubmit={handleSendMessage} className="message-form">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Type your message..."
            disabled={!isInitialized || isLoading}
          />
          <button 
            type="submit" 
            disabled={!inputMessage.trim() || isLoading || !isInitialized}
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
