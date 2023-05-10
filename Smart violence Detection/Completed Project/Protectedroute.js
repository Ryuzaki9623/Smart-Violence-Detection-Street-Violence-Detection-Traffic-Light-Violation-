import React from 'react';
import { Route, Navigate } from 'react-router-dom';
import { useAuth } from 'D:/UI/src/ContextAuth.js';

const ProtectedRoute = ({ component: Component, ...rest }) => {
    const { isAuthenticated } = useAuth();

    return (
        <Route {...rest} render={(props) => (
            isAuthenticated
                ? <Component {...props} />
                : <Navigate to='/login' />
        )} />
    );
}

export default ProtectedRoute;
