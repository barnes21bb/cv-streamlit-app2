import cv2
import streamlit as st


def load_video(video_path):
    """Load video from path and initialize session state"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Failed to open video file")
        cap.release()
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        st.error("Video contains no frames")
        return 0

    st.session_state.video_path = video_path
    st.session_state.total_frames = total_frames
    st.session_state.current_frame_num = 0

    return total_frames


def get_frame(video_path, frame_number):
    """Extract specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None
