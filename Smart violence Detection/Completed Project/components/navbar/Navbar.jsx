import React from 'react'
import './navbar.css'
import { Link } from 'react-router-dom'

function Navbar() {
    return (
        <div className='navbar'>
            <div className="nav-links">
                <ul>
                    <Link to='/'><li>Home</li></Link>
                    <Link to='/login'><li>Login</li></Link>
                    <Link to='/register'><li>Register</li></Link>
                </ul>
            </div>
        </div>
    )
}

export default Navbar