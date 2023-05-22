import React, { useEffect, useState } from 'react';
import 'D:/UI/src/templates/videopage.css'

function VideoPlayer() {
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        window.addEventListener('beforeunload', handleBeforeUnload);

        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
        };
    }, []);

    const handleBeforeUnload = (event) => {
        event.preventDefault();
        event.returnValue = '';
    };

    const handleImageLoad = () => {
        setIsLoading(false);
    };

    return (
        <div className="Apk">
            <header className="Apk-header">
                {isLoading ? (
                    <div className="loading-container">
                        <p>Loading...</p>
                    </div>
                ) : null}
                <img src="http://127.0.0.1:5000/video" onLoad={handleImageLoad} />
            </header>
        </div>
    );
}

export default VideoPlayer;

