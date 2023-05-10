import React from 'react'
import './landing.css'
import HeroImage from '../../assets/pngwing.png'
import { Link } from 'react-router-dom'


function LandingPage() {
    return (
    <div className='landing-page'>
        <div className="landing-page-container">

            <div className="landing-page-head">
                <p>Smart City<span> Era</span></p >
                <p>with Deep</p>
                <p>Artificial Inteligence</p>

              <Link to='/register'><button>Get Started</button></Link>
            </div >
            <div className="landing-page-img">
                <img src={HeroImage} alt="hero-cover" />
            </div>
        </div >
    </div >



    )
}

export default LandingPage