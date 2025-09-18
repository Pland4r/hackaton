import React, { useState } from "react";

const Nav = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="nav-container">
      <div className="nav-content">
        {/* Logo */}
        <div className="nav-logo">
          <div className="logo-icon">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="14" fill="url(#logoGradient)" />
              <path d="M12 10h8v2h-8v-2zm0 4h8v2h-8v-2zm0 4h6v2h-6v-2z" fill="white" />
              <defs>
                <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#667eea" />
                  <stop offset="100%" stopColor="#764ba2" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <span className="logo-text">RH Agent</span>
        </div>

        {/* Desktop Navigation */}
        <div className="nav-links">
          <a href="#home" className="nav-link">
            <span className="nav-link-icon">🏠</span>
            Home
          </a>
          <a href="#about" className="nav-link">
            <span className="nav-link-icon">ℹ️</span>
            About Us
          </a>
        </div>

        {/* Login Button */}
        <div className="nav-actions">
          <button className="login-btn">
            <span className="login-icon">👤</span>
            Login
          </button>
        </div>

        {/* Mobile Menu Button */}
        <button className="mobile-menu-btn" onClick={toggleMenu}>
          <span className={`hamburger ${isMenuOpen ? 'active' : ''}`}>
            <span></span>
            <span></span>
            <span></span>
          </span>
        </button>
      </div>

      {/* Mobile Menu */}
      <div className={`mobile-menu ${isMenuOpen ? 'active' : ''}`}>
        <div className="mobile-menu-content">
          <a href="#home" className="mobile-nav-link" onClick={() => setIsMenuOpen(false)}>
            <span className="nav-link-icon">🏠</span>
            Home
          </a>
          <a href="#about" className="mobile-nav-link" onClick={() => setIsMenuOpen(false)}>
            <span className="nav-link-icon">ℹ️</span>
            About Us
          </a>
          <button className="mobile-login-btn" onClick={() => setIsMenuOpen(false)}>
            <span className="login-icon">👤</span>
            Login
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Nav;
