// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth"
import { getFirestore } from "firebase/firestore"

const firebaseConfig = {
    apiKey: "AIzaSyDh611fIuUMOnjD7OLjqjc126FQwLf_ocw",
    authDomain: "smart-violence-detection.firebaseapp.com",
    projectId: "smart-violence-detection",
    storageBucket: "smart-violence-detection.appspot.com",
    messagingSenderId: "773496712238",
    appId: "1:773496712238:web:0942aacc789c05436703bf",
    measurementId: "G-HZZ5GPH9SY"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app)
const auth = getAuth(app)
export {
    db,
    auth
}