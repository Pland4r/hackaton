import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';


const ChatPage = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'ai',
      text: "Hello! I'm your RH Agent AI assistant. How can I help you today?",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      text: inputText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputText;
    setInputText('');
    setIsTyping(true);

    try {
      // Send message to backend API
      const response = await axios.post('http://localhost:5000/api/chat', {
        prompt: currentInput
      });

      const aiResponse = {
        id: Date.now() + 1,
        type: 'ai',
        text: response.data.response,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, aiResponse]);
    } catch (error) {
      console.error('Error sending message to API:', error);
      
      // Fallback to local response if API fails
      const aiResponse = {
        id: Date.now() + 1,
        type: 'ai',
        text: "I'm sorry, I'm having trouble connecting to the server right now. Please try again later.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, aiResponse]);
    } finally {
      setIsTyping(false);
    }
  };

  const generateAIResponse = (userInput) => {
    const responses = [
      "What salary should we offer a Senior DevOps Engineer?.",
      "hould we advance this candidate's resume?",
      "Which candidate has the highest priority?",
      "Show me all available candidates",
      "Show me all available jobs",
   
    ];
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const quickActions = [
"What salary should we offer a Senior DevOps Engineer?.",
      "hould we advance this candidate's resume?",
      "Which candidate has the highest priority?",
      "Show me all available candidates",
      "Show me all available jobs",
  ];

  const handleQuickAction = (action) => {
    setInputText(action);
    inputRef.current?.focus();
  };

  return (
    <div className="chat-page">
      {/* Main Chat Interface */}
      <div className="chat-container">
        {/* Chat Header */}
        <div className="chat-header">
          <div className="chat-title">
            <div className="ai-avatar">🤖</div>
            <div>
              <h2>RH Agent AI</h2>
              <div className="status-indicator">
                <div className="status-dot"></div>
                <span>Online</span>
              </div>
            </div>
          </div>
          <button 
            className="close-btn"
            onClick={() => setIsModalOpen(true)}
            title="Open in modal"
          >
            ⚡
          </button>
        </div>

        {/* Chat Messages */}
        <div className="chat-messages">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.type}`}>
              <div className="message-avatar">
                {message.type === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-content">
                <div className="message-text">{message.text}</div>
                <div className="message-time">{message.timestamp}</div>
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div className="message ai">
              <div className="message-avatar">🤖</div>
              <div className="typing-indicator">
                <div className="typing-dots">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input */}
        <div className="chat-input-area">
          <div className="chat-input-container">
            <textarea
              ref={inputRef}
              className="chat-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              rows="1"
            />
            <button
              className="send-button"
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isTyping}
            >
              ➤
            </button>
          </div>
          
          {/* Quick Actions */}
          <div className="quick-actions">
            {quickActions.map((action, index) => (
              <button
                key={index}
                className="quick-action-btn"
                onClick={() => handleQuickAction(action)}
              >
                {action}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Modal Chat Interface */}
      {isModalOpen && (
        <div className="chat-modal" onClick={() => setIsModalOpen(false)}>
          <div className="chat-modal-content" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="chat-header">
              <div className="chat-title">
                <div className="ai-avatar">🤖</div>
                <div>
                  <h2>RH Agent AI</h2>
                  <div className="status-indicator">
                    <div className="status-dot"></div>
                    <span>Online</span>
                  </div>
                </div>
              </div>
              <button 
                className="close-btn"
                onClick={() => setIsModalOpen(false)}
                title="Close modal"
              >
                ✕
              </button>
            </div>

            {/* Modal Messages */}
            <div className="chat-messages">
              {messages.map((message) => (
                <div key={message.id} className={`message ${message.type}`}>
                  <div className="message-avatar">
                    {message.type === 'user' ? '👤' : '🤖'}
                  </div>
                  <div className="message-content">
                    <div className="message-text">{message.text}</div>
                    <div className="message-time">{message.timestamp}</div>
                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="message ai">
                  <div className="message-avatar">🤖</div>
                  <div className="typing-indicator">
                    <div className="typing-dots">
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                      <div className="typing-dot"></div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Modal Input */}
            <div className="chat-input-area">
              <div className="chat-input-container">
                <textarea
                  className="chat-input"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message here..."
                  rows="1"
                />
                <button
                  className="send-button"
                  onClick={handleSendMessage}
                  disabled={!inputText.trim() || isTyping}
                >
                  ➤
                </button>
              </div>
              
              {/* Quick Actions */}
              <div className="quick-actions">
                {quickActions.map((action, index) => (
                  <button
                    key={index}
                    className="quick-action-btn"
                    onClick={() => handleQuickAction(action)}
                  >
                    {action}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatPage;
