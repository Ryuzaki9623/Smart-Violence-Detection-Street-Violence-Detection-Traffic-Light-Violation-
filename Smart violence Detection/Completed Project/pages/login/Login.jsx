import React from 'react'
import './login.css'
import TextField from '@mui/material/TextField';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from 'D:/UI/src/ContextAuth.js';
import { useState,useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';


const Login = () => {
    const { UserLogin } = useAuth()
    const [err, setError] = useState("")
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [user, setUser] = useState({
        email: "",
        password: "",
    })
    const navigate = useNavigate()

    useEffect(() => {
        if (!isLoggedIn) {
            window.history.pushState(null, '', '/')
        }
    }, [isLoggedIn])


    const UserHandler = (e) => {
        const { name, value } = e.target;
        setUser((pre) => {
            return {
                ...pre,
                [name]: value
            }
        })
    }
    const SubmitHandler = async (e) => {
        e.preventDefault()
        const { email, password } = user
        if (email === "" || password === "") {
            setInterval(() => {
                setError("")
            }, 5000)
            return setError("Fill All the Fields")
        }
        try {
            await UserLogin(email, password)
            const response = await axios.post('/login', { email, password });
            setIsLoggedIn(true);
            navigate("/homepage")
            notify()
        } catch (error) {

            if (error.code === "auth/user-not-found") {
                setInterval(() => {
                    setError("")
                }, 5000)
                user_notify()
                return setError("User Not Found")
            }
            else if (error.code === "auth/wrong-password") {
                setInterval(() => {
                    setError("")
                }, 5000)
                wpass_notify()
                return setError("Wrong Password")
            }
            else {
                setInterval(() => {
                    setError("")
                }, 5000)
                return setError(`${error.message}`)
            }
        }

    }


    const notify = () => toast.success('Logged In Succesfully', {
        position: "bottom-right",
        autoClose: 2000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "light",
    });
    const user_notify = () => toast.error('User Not Found', {
        position: "bottom-right",
        autoClose: 2000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "light",
    });
    const wpass_notify = () => toast.error('Wrong Pass', {
        position: "bottom-right",
        autoClose: 2000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "light",
    });

    return (
        <div className='login'>

            <div className="login-container">
                <div className="login-header">
                    L<span>o</span>gin
                </div>

                <div className="login-body">
                    <form onSubmit={SubmitHandler} className="form">
                        <TextField id="outlined-basic" value={user.email} name='email' onChange={UserHandler} label="Email" variant="outlined" sx={{ marginBottom: '1rem', fontSize: '10px', width: '90%' }} size='small' />
                        <TextField id="outlined-basic" value={user.password} name='password' onChange={UserHandler} label="Password" variant="outlined" sx={{ marginBottom: '1rem', fontSize: '10px', width: '90%' }} type ='password' size='small' />
                        <div className="login-button">
                            <button>Login</button>
                        </div>
                    </form>
                </div>
                <p className="forget">Dont Have An Account ?<Link to='/register'> Sign Up</Link></p>
            </div>
        </div>
    )
}

export default Login