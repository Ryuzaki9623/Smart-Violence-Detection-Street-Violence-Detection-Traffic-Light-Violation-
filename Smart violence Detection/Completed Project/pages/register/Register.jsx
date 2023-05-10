import React from 'react'
import './register.css'
import TextField from '@mui/material/TextField';
import { Box } from '@mui/material';
import MenuItem from '@mui/material/MenuItem';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { useAuth } from 'D:/UI/src/ContextAuth.js';

    const Signup = () => {
        const { error, SignUp, currentuser } = useAuth()
        const [err, setError] = useState("")
        const [backError, setBackError] = useState("")
        const [user, setUser] = useState({
            uname: "",
            mail: "",
            pass: "",
            fname: "",
            lname: "",
        })
        useEffect(() => {
            console.log("i am in")
            if (error) {
                setInterval(() => {
                    setBackError("")
                }, 5000)
                setBackError(error)
            }
        }, [error, currentuser])
        const UserHandler = (e) => {
            const { name, value } = e.target;
            console.log(name + "::::::::::" + value)
            setUser((pre) => {
                return {
                    ...pre,
                    [name]: value
                }
            })
        }

        const SubmitHandler = async (e) => {
            e.preventDefault()
            const { mail, pass, uname,fname,lname } = user
            if (pass === "" || mail === "" || uname === "" || fname === "" || lname === "" ) {
                setInterval(() => {
                    setError("")
                }, 5000)
                fld_notify()
                return setError("please fill All the fields ")
            }
            else if (!pass.length >= 6) {
                setInterval(() => {
                    setError("")
                }, 5000)
                pass_err()
                return setError("Password Must be Greater then 6 Length")
            }
            else {

                SignUp(mail, pass, uname,fname,lname)
                {
                    currentuser && setUser({
                        uname: "",
                        mail: "",
                        pass: "",
                        fname: "",
                        lname: "",
                    })
                }
            }
        }
    // toastify
    const fld_notify = () => toast.error('Please Fill All The Fields', {
        position: "bottom-right",
        autoClose: 2000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "light",
    });
    const pass_err = () => toast.error('Password Must Be Atleast 6 Characters', {
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
        <div className='register'>

            <div className="register-container">
                <div className="register-header">
                    Si<span>g</span>n Up
                </div>

                <div className="register-body">
                    <form onSubmit={SubmitHandler}>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', width: '90%' }} >
                            <TextField id="FName" name="fname" value={user.fname} onChange={UserHandler} label="First Name" variant="outlined" sx={{ marginBottom: '1rem', fontSize: '10px', width: '50%' }} size='small' />
                            <TextField id="LName" name="lname" value={user.lname} onChange={UserHandler} label="Last Name" variant="outlined" sx={{ marginBottom: '1rem', fontSize: '10px', width: '50%' }} size='small' />
                        </Box>
                        <TextField id="Email" name="mail" label="Email" value={user.mail} onChange={UserHandler} variant="outlined" sx={{ marginBottom: '1rem', fontSize: '10px', width: '90%' }} size='small'/>
                        <TextField id="UName" name="uname" label="User Name" variant="outlined" value={user.uname} onChange={UserHandler} sx={{ marginBottom: '1rem', fontSize: '10px', width: '90%' }} size='small'/>
                        <TextField id="pass" name="pass" label="Password" variant="outlined" value={user.pass} onChange={UserHandler} sx={{ marginBottom: '1rem', fontSize: '10px', width: '90%' }} size='small' type='password'/>

                        <div className="register-button">
                           <button variant="primary" type="Submit">Create Account</button>
                        </div>
                     </form>
                </div>
                <p className="forget">Already Have An Account ?<Link to = '/Login'> Sign In</Link></p>
            </div>
        </div>
    )
}

export default Signup