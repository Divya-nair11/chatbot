import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [chatSessions, setChatSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);

  // Load chat sessions from localStorage on mount, but don't load the last session
  useEffect(() => {
    const savedSessions = localStorage.getItem('chatSessions');
    if (savedSessions) {
      setChatSessions(JSON.parse(savedSessions));
    }
    // Start with a new chat (empty chatHistory and no currentSessionId)
    setChatHistory([]);
    setCurrentSessionId(null);
  }, []);

  // Save chat sessions to localStorage whenever they change
  useEffect(() => {
    if (chatSessions.length > 0) {
      localStorage.setItem('chatSessions', JSON.stringify(chatSessions));
    }
  }, [chatSessions]);

  const handleSendQuery = async () => {
    if (!query.trim()) return;

    const newMessage = { sender: 'user', message: query };
    const updatedHistory = [...chatHistory, newMessage];
    setChatHistory(updatedHistory);

    // Clear the input field immediately
    setQuery('');

    try {
      const response = await axios.post('http://127.0.0.1:8000/chat', { query });
      let botResponse = response.data.response;

      // Remove labels if they exist
      botResponse = botResponse.replace(/^Answer:\s*/, '').replace(/\[(KNOWLEDGE BASE|WEB SEARCH|GENERAL KNOWLEDGE|Fallback Response)\]/g, '').trim();

      const updatedHistoryWithBot = [...updatedHistory, { sender: 'bot', message: botResponse }];
      setChatHistory(updatedHistoryWithBot);

      // Update or create a chat session
      if (currentSessionId) {
        setChatSessions((prevSessions) =>
          prevSessions.map((session) =>
            session.id === currentSessionId
              ? { ...session, history: updatedHistoryWithBot }
              : session
          )
        );
      } else {
        const newSession = {
          id: Date.now().toString(),
          title: query.length > 30 ? query.substring(0, 30) + '...' : query,
          history: updatedHistoryWithBot,
        };
        setChatSessions((prevSessions) => [...prevSessions, newSession]);
        setCurrentSessionId(newSession.id);
      }
    } catch (error) {
      console.error('Error sending query:', error);
      const updatedHistoryWithError = [...updatedHistory, { sender: 'bot', message: 'Error: Could not get a response from the server.' }];
      setChatHistory(updatedHistoryWithError);

      if (currentSessionId) {
        setChatSessions((prevSessions) =>
          prevSessions.map((session) =>
            session.id === currentSessionId
              ? { ...session, history: updatedHistoryWithError }
              : session
          )
        );
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendQuery();
    }
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const startNewChat = () => {
    setChatHistory([]);
    setCurrentSessionId(null);
  };

  const loadChatSession = (sessionId) => {
    const session = chatSessions.find((s) => s.id === sessionId);
    if (session) {
      setCurrentSessionId(sessionId);
      setChatHistory(session.history);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      {isSidebarOpen && (
        <div className="sidebar">
          <button className="sidebar-toggle" onClick={toggleSidebar}>
            ◄
          </button>
          <div className="sidebar-content">
            <button className="new-chat-button" onClick={startNewChat}>
              New Chat
            </button>
            <h3>Chat History</h3>
            <ul className="chat-history-list">
              {chatSessions.map((session) => (
                <li
                  key={session.id}
                  className={session.id === currentSessionId ? 'active' : ''}
                  onClick={() => loadChatSession(session.id)}
                >
                  {session.title}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Show Button (appears when sidebar is hidden) */}
      {!isSidebarOpen && (
        <button className="show-sidebar-button" onClick={toggleSidebar}>
          ►
        </button>
      )}

      {/* Chatbot Container */}
      <div className={`chatbot-container ${isSidebarOpen ? 'with-sidebar' : 'full-width'}`}>
        <h1>Chatbot</h1>
        <div className="chat-container">
          {chatHistory.map((chat, index) => (
            <div key={index} className={`chat-message ${chat.sender}`}>
              <strong>{chat.sender === 'user' ? 'You' : 'Bot'}:</strong>
              <p>{chat.message}</p>
            </div>
          ))}
        </div>

        {/* Input Container */}
        <div className="input-container">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your query..."
          />
          <button onClick={handleSendQuery}>Send</button>
        </div>
      </div>
    </div>
  );
}

export default App;