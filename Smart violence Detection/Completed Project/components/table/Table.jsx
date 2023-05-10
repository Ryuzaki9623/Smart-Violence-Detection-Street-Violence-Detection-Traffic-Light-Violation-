import React, { useState, useEffect } from 'react';
import { db } from "D:/UI/src/firebase.js";
import { doc, getDoc, updateDoc, collection, onSnapshot } from "firebase/firestore";
import { getAuth } from "firebase/auth";
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';

function TableComponent() {

    const [loading, setLoading] = useState(true);
    const [rows, setRows] = useState([]);
    const [lastUpdatedAt, setLastUpdatedAt] = useState(null);

    const fetchData = async () => {
        const auth = getAuth();
        const user = auth.currentUser;
        if (user) {
            const userId = user.uid;
            const docRef = doc(db, 'users', userId);
            const docSnap = await getDoc(docRef);
            if (docSnap.exists()) {
                const userFiles = docSnap.data().files;
                const data = userFiles.map((file, index) => ({
                    id: index + 1,
                    date: file.dateAdded || '',
                    fileName: file.name || '',
                    Details: file.status || '',
                }));
                const updatedAt = docSnap.data().updatedAt;
                setRows(data);
                setLastUpdatedAt(updatedAt);
            }
        }
    };

    useEffect(() => {
        const checkForUpdates = async () => {
            const auth = getAuth();
            const user = auth.currentUser;
            if (user) {
                const userId = user.uid;
                const docRef = doc(db, 'users', userId);
                const docSnap = await getDoc(docRef);
                if (docSnap.exists()) {
                    const updatedAt = docSnap.data().updatedAt;
                    if (updatedAt !== lastUpdatedAt) {
                        const userFiles = docSnap.data().files;
                        const data = userFiles.map((file, index) => ({
                            id: index + 1,
                            date: file.date || '',
                            fileName: file.filename || '',
                            Details: file.status || ''
                        }));
                        setRows(data);
                        setLastUpdatedAt(updatedAt);
                    }
                }
            }
        };

        const intervalId = setInterval(checkForUpdates, 5000);

        return () => {
            clearInterval(intervalId);
        };
    }, [lastUpdatedAt]);



    return (
        <div>
            <TableContainer component={Paper}>
                <Table sx={{ minWidth: 650 }} aria-label="caption table">
                    <caption>All your history will charted above</caption>
                    <TableHead>
                        <TableRow>
                            <TableCell>No</TableCell>
                            <TableCell align="center">Date</TableCell>
                            <TableCell align="center">File Name</TableCell>
                            <TableCell align="center">Status</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {rows.map((row, val) => (
                            <TableRow key={val}>
                                <TableCell component="th" scope="row">
                                    {row.id}
                                </TableCell>
                                <TableCell align="center">{row.date}</TableCell>
                                <TableCell align="center">{row.fileName}</TableCell>
                                <TableCell align="center">{row.Details}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </div>
    )
}

export default TableComponent
