import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from deepface import DeepFace

st.set_page_config(page_title="Emotion Mirror", page_icon="🪞", layout="centered")

st.markdown("""
<style>
    /* Styling to make it look premium */
    .stApp {
        background-color: #0e1117;
    }
    h1 {
        text-align: center;
        color: #00FFCA;
        font-family: 'Inter', sans-serif;
    }
    .instruction {
        text-align: center;
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🪞 Emotion Mirror")
st.markdown('<p class="instruction">Real-time facial expression analysis. Please allow camera access.</p>', unsafe_allow_html=True)

# Load Haar cascade for fast initial face detection (bounding box)
@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_cascade()

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Convert incoming WebRTC frame to BGR format for OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for the Haar Cascade face detector
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (fast)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) == 0:
            # Edge Case: No Face Detected
            text = "No face detected"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            # Display text centered near top
            cv2.putText(img, text, (text_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Edge Case: Multiple Faces handled dynamically
            for (x, y, w, h) in faces:
                # Add padding to the face bounding box for better DeepFace accuracy
                padding = 20
                start_y = max(0, y - padding)
                start_x = max(0, x - padding)
                end_y = min(img.shape[0], y + h + padding)
                end_x = min(img.shape[1], x + w + padding)
                
                # Extract the Region of Interest (ROI) containing just the face
                face_roi = img[start_y:end_y, start_x:end_x]
                
                try:
                    # Predict emotion on just the ROI
                    # enforce_detection=False prevents crash if DeepFace's internal detector disagrees with Haar
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                    
                    if isinstance(result, list):
                        result = result[0]
                        
                    emotion = result['dominant_emotion']
                    
                    # Choose a premium color mapped to the emotion
                    color_map = {
                        'happy': (0, 255, 0),       # Green
                        'sad': (255, 0, 0),         # Blue
                        'angry': (0, 0, 255),       # Red
                        'surprise': (0, 255, 255),  # Yellow
                        'neutral': (255, 255, 255), # White
                        'fear': (128, 0, 128),      # Purple
                        'disgust': (0, 128, 0)      # Dark Green
                    }
                    color = color_map.get(emotion, (255, 255, 255))
                    
                    # Capitalize emotion text
                    emotion_text = emotion.capitalize()
                    
                    # Draw bounding box and label for each face independently
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                    
                except Exception as e:
                    # Human-like fallback if analysis fails for a specific frame (e.g. motion blur)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 200), 2)
                    cv2.putText(img, "Scanning...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # Return the processed frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Streamer initialized
webrtc_streamer(
    key="emotion-mirror",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown('<p class="footer">Built for Bipolar Factory Assignment.</p>', unsafe_allow_html=True)
