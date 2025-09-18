import "./App.css";
import { Routes, BrowserRouter, Route } from "react-router-dom";
import NavBar from "./components/Nav";
import ChatPage from "./pages/ChatPage";
import React from "react";
function App() {
  return <>
  <BrowserRouter>
  <NavBar/>
  <Routes>
<Route path="*" element={<ChatPage/>} />
  </Routes>
  </BrowserRouter>
  
  
  </>;
}

export default App;
