import React from 'react'
import './App.css'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/navbar/Navbar';
import Login from './pages/login/Login'
import Register from './pages/register/Register'
import LandingPage from './pages/landing page/LandingPage'
import HomePage from './pages/home page/HomePage';
import Snackbar from './components/snackbar/Snackbar';
import { UserAuthContextProvider } from 'D:/UI/src/ContextAuth.js'
import Videopage from  'D:/UI/src/templates/Videopage.js'

function App() {
  return (
    <div className='app'>
      <BrowserRouter>
            <Navbar />
              <Snackbar />
              <UserAuthContextProvider>
                  <Routes>
                    <Route path="/" element={<LandingPage />}/>
                    <Route path="/login" element={<Login />}/>
                    <Route path="/register" element={<Register />}/>
                      <Route path="/homepage" element={<HomePage />} />
                      <Route path="/Video" element={<Videopage />} />
                  </Routes>
              </UserAuthContextProvider>
      </BrowserRouter>
    </div>
  )
}

export default App