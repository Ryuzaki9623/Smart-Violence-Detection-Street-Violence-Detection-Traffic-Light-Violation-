import React, {createContext, useContext, useEffect,useState} from 'react'
import {
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signOut,
    getAuth,
    onAuthStateChanged,
    AuthErrorCodes
} from "firebase/auth";
import { toast } from 'react-toastify';
import { auth, db } from "D:/UI/src/firebase.js";
import {doc, setDoc } from "firebase/firestore";

const userAuthContext = createContext();
export const useAuth = () => { return useContext(userAuthContext) }

const  UserAuthContextProvider = ({ children }) => {
    const [error, setError] = useState("")
    const [currentuser, setUser] = useState({});
    useEffect(() => {
        const Unsubscribe = onAuthStateChanged(auth, (user) => {
            if (user) {
                setUser(user)
            }

        });
        return () => {
            Unsubscribe();
        }
    }, [currentuser])

    const UserLogin = (email, password) => {
        return signInWithEmailAndPassword(auth, email, password)
    }

    //logout Functionllity
    const logout = () => {
        return signOut(auth)
    }
    const SignUp = async (email, password, UName,fname,lname) => {
        setError("");
        createUserWithEmailAndPassword(auth, email, password).then(
            async (result) => {
                console.log(result)
                try {
                    // const docRef = await addDoc(collection(db, "users"), {
                    //   FullName,
                    //   userId: `${result.user.uid}`
                    // });
                    const ref = doc(db, "users", result.user.uid)
                    const docRef = await setDoc(ref, { UName, email, fname, lname})
                    notify()
                    alert("Welcome new User created successfully")
                    console.log("Document written with ID: ", docRef.id);
                } catch (e) {
                    console.error("Error adding document: ", e);
                }
            }
        ).catch(err => {
            if (err.code === "auth/email-already-in-use") {

                setInterval(() => {
                    setError("")
                }, 5000)
                em_exsts()
                setError("email already in use try another email")
            }
            else if (err.code === AuthErrorCodes.WEAK_PASSWORD) {

                setInterval(() => {
                    setError("")
                }, 5000)
                passw_err()
                setError("Password Must be 6 charecter")
            }

            else {
                setError(err.message)
            }
        })
    }
    const passw_err = () => toast.error('Password Must Be Atleast 6 Characters', {
        position: "bottom-right",
        autoClose: 2000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "light",
    });
    const em_exsts = () => toast.warning('Email Already Exists Please Use Another Mail', {
        position: "bottom-right",
        autoClose: 2000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "light",
    });
    const notify = () => toast.success('Account Created Succesfully', {
        position: "bottom-right",
        autoClose: 2000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "light",
    });
    const value = {
        SignUp,
        error,
        currentuser,
        UserLogin,
        logout
    }
    return (
        <userAuthContext.Provider value={value} >
            {children}
        </userAuthContext.Provider>
    );
}

export { UserAuthContextProvider }