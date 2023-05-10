import React, { useEffect } from 'react';

function VideoPlayer() {
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

    return (
        <div className="Apk">
            <header className="Apk-header">
                <img src={"http://127.0.0.1:5000/video"} />
            </header>
        </div>
    );
}

export default VideoPlayer;

