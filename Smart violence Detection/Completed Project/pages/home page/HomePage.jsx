import React from 'react'
import { useRef } from 'react';
import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from 'D:/UI/src/ContextAuth.js';
import { getAuth } from "firebase/auth";
import { toast } from 'react-toastify';
import { db } from "D:/UI/src/firebase.js";
import { doc, getDoc, updateDoc } from "firebase/firestore";
import Table from '../../components/table/Table'
import './homepage.css'

function HomePage() {
    const [file, setFile] = useState('')
    const inputRef = useRef(null);

    const auth = getAuth();
    const user = auth.currentUser;

    const [username, setUserName] = useState('');
    const [role, setrole] = useState('');
    const [email, setemail] = useState('');
    const [fname, setfname] = useState('');
    const [lname, setlname] = useState('');

    const [isLoggedIn, setIsLoggedIn] = useState(true);

    useEffect(() => {
        // add a new state to the browser history
        window.history.pushState(null, null, window.location.pathname);
        // listen to the popstate event (back/forward button)
        window.addEventListener('popstate', onBackButtonEvent);
        return () => {
            // remove the popstate listener when the component unmounts
            window.removeEventListener('popstate', onBackButtonEvent);
        };
    }, []);

    const onBackButtonEvent = (event) => {
        // prevent the default behavior (going back to the previous state)
        event.preventDefault();
        // replace the current state with the same state (i.e., the current page)
        window.history.pushState(null, null, window.location.pathname);
        if (!isLoggedIn) {
            navigate('/Login');
        }
    };

    useEffect(() => {
        if (user && user.uid) {
            // fetch the user data from Firestore
            const docRef = doc(db, "users", user.uid);
            getDoc(docRef).then((doc) => {
                if (doc.exists()) {
                    const userData = doc.data();
                    setUserName(userData.UName);
                    setrole(userData.rl);
                    setemail(userData.email);
                    setfname(userData.fname);
                    setlname(userData.lname);
                }
            }).catch((error) => {
                console.log("Error getting user data:", error);
            });
        }
    }, [user])
    const handleClick = () => {
        //  open file input box on click of other element
        inputRef.current.click();
    };

    const { logout } = useAuth()
    const navigate = useNavigate()
    const userlogout = async () => {

        try {
            await logout()
            setIsLoggedIn(false);

            window.history.pushState(null, '', '/login');

            navigate("/Login")
            notify()
        } catch (error) {
            console.log(error)
        }

    }

    const handleFileChange = async (event) => {
        const fileObj = event.target.files;
        if (!fileObj) {
            return;
        }

        console.log('fileObj is', fileObj);

        const formData = new FormData();
        for (let i = 0; i < fileObj.length; i++) {
            formData.append('file', fileObj[i]);
        }

        const fileNamesArr = [];
        for (let i = 0; i < fileObj.length; i++) {
            fileNamesArr.push(fileObj[i].name);
        }

        setFile(fileNamesArr.join(', '));

        // reset file input
        event.target.value = null;

        // is now empty
        console.log(event.target.files);

        const response = await fetch('/home', {
            method: 'POST',
            body: formData,
        });

        if (response.ok) {
            console.log('File uploaded successfully');

            //Display in a new window
            window.open('/Video','_blank');
        }
        else {
            console.error('Error uploading file:', response.status);
        }
    };


    // toast logout
    const notify = () => toast.success('Logged Out', {
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
        <div className='home-page'>
            <div className="home-page-container">
                <div className="home-page-left-sec">
                    <div className="video-upload">
                        <h4>Upload File</h4>
                        <input
                            style={{ display: 'none' }}
                            ref={inputRef}
                            type="file"
                            multiple
                            onChange={handleFileChange}
                        />
                        <button onClick={handleClick}>Upload Video</button>
                        <div className='fileName'>{file}</div>
                    </div>

                    <div className="analytics">
                        <p>History</p>
                        <Table />
                    </div>
                </div>


                <div className="home-page-right-sec">
                    <div className="profile">
                        <h4>Profile</h4>
                        <div className="profile-header">
                            <div className="profile-img">
                                <img src="https://w7.pngwing.com/pngs/722/101/png-transparent-computer-icons-user-profile-circle-abstract-miscellaneous-rim-account-thumbnail.png" alt="" />
                            </div>

                            <div className="profile-details">
                                <p>{fname} {lname}</p>
                                <p>User</p>
                                <p>{username}</p>
                                <p>{email}</p>
                            </div>
                        </div>
                        <button onClick={userlogout}>Logout</button>
                    </div>   
                </div>
            </div>
        </div>
    )
}

export default HomePage