import { useEffect, useRef } from 'react';
import io from 'socket.io-client';

export default function LiveTryOn({ category, customImage }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    // Socket.IO Connection with Python Backend
    const socket = io('http://localhost:8000');

    // Custom Uploaded Image ya Preset URL Handle karne ki logic
    if (customImage) {
      if (typeof customImage === 'string') {
        // High-resolution image canvas buffer conversion for presets
        const img = new Image();
        img.crossOrigin = 'Anonymous';
        img.src = customImage;
        img.onload = () => {
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = img.width;
          tempCanvas.height = img.height;
          const ctx = tempCanvas.getContext('2d');
          ctx.drawImage(img, 0, 0);
          const base64Data = tempCanvas.toDataURL('image/png');
          
          socket.emit('update_overlay', {
            image_data: base64Data,
            category: category
          });
        };
      } else if (customImage instanceof File) {
        // Direct File upload handler (AVIF, WEBP, PNG, JPG)
        const reader = new FileReader();
        reader.onload = (e) => {
          socket.emit('update_overlay', {
            image_data: e.target.result,
            category: category
          });
        };
        reader.readAsDataURL(customImage);
      }
    } else {
      socket.emit('change_category', { category: category });
    }

    // Live Webcam Frame Streaming to Python Server
    const interval = setInterval(() => {
      if (videoRef.current && canvasRef.current) {
        const context = canvasRef.current.getContext('2d');
        context.drawImage(videoRef.current, 0, 0, 640, 480);
        const frame = canvasRef.current.toDataURL('image/jpeg', 0.6);
        socket.emit('stream_frame', { image: frame, productType: category });
      }
    }, 100);

    // AI Processed Frame Receive Handler
    socket.on('processed_frame', (data) => {
      const img = new Image();
      img.src = data.image;
      img.onload = () => {
        if (canvasRef.current) {
          const context = canvasRef.current.getContext('2d');
          context.drawImage(img, 0, 0, 640, 480);
        }
      };
    });

    // Request & Start Browser Webcam
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((err) => {
        console.error("Webcam Access Error: ", err);
      });

    return () => {
      clearInterval(interval);
      socket.disconnect();
    };
  }, [category, customImage]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '480px', backgroundColor: '#0f172a' }}>
      <video ref={videoRef} autoPlay playsInline style={{ display: 'none' }} />
      <canvas ref={canvasRef} width={640} height={480} style={{ width: '100%', height: '100%', borderRadius: '20px' }} />
    </div>
  );
}