import streamlit as st
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import os
import tempfile
from datetime import datetime
import json
import sqlite3
import shutil
from pathlib import Path
import pickle
import re
import io
import zipfile
from streamlit_drawable_canvas import st_canvas

try:
    from models.detection import YOLODetector
except Exception:
    YOLODetector = None

# Database setup
def init_database():
    """Initialize SQLite database for users and projects"""
    conn = sqlite3.connect('video_annotation.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Projects table
    c.execute('''CREATE TABLE IF NOT EXISTS projects
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  name TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users(id),
                  UNIQUE(user_id, name))''')
    
    # Annotations table
    c.execute('''CREATE TABLE IF NOT EXISTS annotations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  project_id INTEGER NOT NULL,
                  video_name TEXT NOT NULL,
                  frame_num INTEGER NOT NULL,
                  annotations_data TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (project_id) REFERENCES projects(id))''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

# Create directories for file storage
STORAGE_DIR = Path("annotation_storage")
STORAGE_DIR.mkdir(exist_ok=True)

def validate_email(email):
    """Basic email validation"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def get_or_create_user(email):
    """Get user ID or create new user"""
    conn = sqlite3.connect('video_annotation.db')
    c = conn.cursor()
    
    c.execute("SELECT id FROM users WHERE email = ?", (email,))
    result = c.fetchone()
    
    if result:
        user_id = result[0]
    else:
        c.execute("INSERT INTO users (email) VALUES (?)", (email,))
        user_id = c.lastrowid
        conn.commit()
    
    conn.close()
    return user_id

def get_all_users():
    """Get all registered users"""
    conn = sqlite3.connect('video_annotation.db')
    c = conn.cursor()
    c.execute("SELECT email FROM users ORDER BY email")
    users = [row[0] for row in c.fetchall()]
    conn.close()
    return users

def create_project(user_id, project_name):
    """Create a new project for user"""
    conn = sqlite3.connect('video_annotation.db')
    c = conn.cursor()
    
    try:
        c.execute("INSERT INTO projects (user_id, name) VALUES (?, ?)", 
                  (user_id, project_name))
        project_id = c.lastrowid
        conn.commit()
        
        # Create project directory
        project_dir = STORAGE_DIR / f"user_{user_id}" / f"project_{project_id}"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        return project_id
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_projects(user_id):
    """Get all projects for a user"""
    conn = sqlite3.connect('video_annotation.db')
    c = conn.cursor()
    c.execute("""SELECT id, name, created_at, updated_at 
                 FROM projects 
                 WHERE user_id = ? 
                 ORDER BY updated_at DESC""", (user_id,))
    projects = c.fetchall()
    conn.close()
    return projects

def save_video_to_project(user_id, project_id, video_file):
    """Save video file to project directory"""
    project_dir = STORAGE_DIR / f"user_{user_id}" / f"project_{project_id}"
    video_path = project_dir / video_file.name
    
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    return str(video_path)

def save_annotations(project_id, video_name, frame_num, annotations):
    """Save annotations to database"""
    conn = sqlite3.connect('video_annotation.db')
    c = conn.cursor()
    
    # Check if annotation exists
    c.execute("""SELECT id FROM annotations 
                 WHERE project_id = ? AND video_name = ? AND frame_num = ?""",
              (project_id, video_name, frame_num))
    result = c.fetchone()
    
    annotations_json = json.dumps(annotations)
    
    if result:
        # Update existing
        c.execute("""UPDATE annotations 
                     SET annotations_data = ? 
                     WHERE id = ?""",
                  (annotations_json, result[0]))
    else:
        # Insert new
        c.execute("""INSERT INTO annotations 
                     (project_id, video_name, frame_num, annotations_data) 
                     VALUES (?, ?, ?, ?)""",
                  (project_id, video_name, frame_num, annotations_json))
    
    # Update project timestamp
    c.execute("UPDATE projects SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
              (project_id,))
    
    conn.commit()
    conn.close()

def load_project_annotations(project_id, video_name):
    """Load all annotations for a video in a project"""
    conn = sqlite3.connect('video_annotation.db')
    c = conn.cursor()
    c.execute("""SELECT frame_num, annotations_data 
                 FROM annotations 
                 WHERE project_id = ? AND video_name = ?""",
              (project_id, video_name))
    
    annotations = {}
    for frame_num, data in c.fetchall():
        annotations[frame_num] = json.loads(data)
    
    conn.close()
    return annotations

def get_project_videos(user_id, project_id):
    """Get all videos in a project directory"""
    project_dir = STORAGE_DIR / f"user_{user_id}" / f"project_{project_id}"
    if project_dir.exists():
        return [f for f in project_dir.iterdir() 
                if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]
    return []

# Initialize session state for user management
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None
if 'current_project_name' not in st.session_state:
    st.session_state.current_project_name = None

# Original session state variables
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}
if 'classes' not in st.session_state:
    st.session_state.classes = ['good-cup', 'bad-cup', 'no-cup']
if 'current_class' not in st.session_state:
    st.session_state.current_class = 'good-cup'
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0
if 'current_frame_num' not in st.session_state:
    st.session_state.current_frame_num = 0
if 'detector_conf' not in st.session_state:
    st.session_state.detector_conf = 0.25
if 'detector' not in st.session_state:
    try:
        from models.detection import YOLODetector
        st.session_state.detector = YOLODetector(conf=st.session_state.detector_conf)
    except Exception:
        st.session_state.detector = None
if 'detection_counts' not in st.session_state:
    st.session_state.detection_counts = {}

def load_video(video_path):
    """Load video from path"""
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
    st.session_state.detection_counts = {}

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

def draw_annotations(image, annotations):
    """Draw bounding boxes on image"""
    img_with_boxes = image.copy()
    
    colors = {
        'good-cup': (0, 255, 0),
        'bad-cup': (255, 0, 0),
        'no-cup': (255, 255, 0)
    }
    
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        class_name = ann['class']
        color = colors.get(class_name, (0, 0, 255))
        
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        label = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img_with_boxes, (x1, y1 - text_height - 4), (x1 + text_width + 4, y1), color, -1)
        cv2.putText(img_with_boxes, label, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), thickness)
    
    return img_with_boxes

def generate_pascal_voc_xml(annotations_dict, video_name, video_shape):
    """Return PASCAL VOC XML strings per frame.

    Parameters
    ----------
    annotations_dict : dict
        Mapping of frame number to a list of annotations.
    video_name : str
        Base name of the annotated video.
    video_shape : tuple
        (height, width, channels) of the video frame.

    Returns
    -------
    dict
        Dictionary mapping frame numbers to XML strings, each containing
        a single ``<annotation>`` root element.
    """

    output = {}
    h, w, c = video_shape

    for frame_num, frame_annotations in annotations_dict.items():
        if not frame_annotations:
            continue

        annotation = ET.Element("annotation")

        folder = ET.SubElement(annotation, "folder")
        folder.text = "frames"

        filename = ET.SubElement(annotation, "filename")
        filename.text = f"{video_name}_frame_{frame_num}.jpg"

        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Custom Video Annotation"

        size = ET.SubElement(annotation, "size")
        width = ET.SubElement(size, "width")
        width.text = str(w)
        height = ET.SubElement(size, "height")
        height.text = str(h)
        depth = ET.SubElement(size, "depth")
        depth.text = str(c)

        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        for ann in frame_annotations:
            obj = ET.SubElement(annotation, "object")

            name = ET.SubElement(obj, "name")
            name.text = ann['class']

            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"

            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"

            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(ann['bbox'][0])
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(ann['bbox'][1])
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(ann['bbox'][2])
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(ann['bbox'][3])

        from xml.dom import minidom
        rough_string = ET.tostring(annotation, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        lines = pretty_xml.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        pretty_xml = '\n'.join(non_empty_lines)

        output[frame_num] = pretty_xml

    return output


def run_model_detection(video_path, conf):
    """Run YOLOv8 detection on the entire video."""
    if st.session_state.detector is None:
        st.error("YOLOv8 not available. Please install ultralytics.")
        return {}, {}
    st.session_state.detector.set_conf(conf)
    annotations, counts = st.session_state.detector.detect_video(video_path)
    return annotations, counts

# Streamlit UI
st.set_page_config(page_title="Video Annotation Tool", layout="wide")

# User authentication section
if st.session_state.user_id is None:
    st.title("🎥 Video Annotation Tool - Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### Welcome! Please enter your workspace")
        
        # Check for existing users
        existing_users = get_all_users()
        
        if existing_users:
            workspace_option = st.radio(
                "Choose an option:",
                ["Select existing workspace", "Create new workspace"]
            )
            
            if workspace_option == "Select existing workspace":
                selected_email = st.selectbox(
                    "Select your workspace:",
                    existing_users
                )
                if st.button("Enter Workspace", type="primary"):
                    st.session_state.user_id = get_or_create_user(selected_email)
                    st.session_state.user_email = selected_email
                    st.rerun()
            else:
                email = st.text_input("Enter your email address:")
                if st.button("Create Workspace", type="primary"):
                    if email and validate_email(email):
                        if email not in existing_users:
                            st.session_state.user_id = get_or_create_user(email)
                            st.session_state.user_email = email
                            st.rerun()
                        else:
                            st.error("This email already has a workspace!")
                    else:
                        st.error("Please enter a valid email address")
        else:
            email = st.text_input("Enter your email address to create your first workspace:")
            if st.button("Create Workspace", type="primary"):
                if email and validate_email(email):
                    st.session_state.user_id = get_or_create_user(email)
                    st.session_state.user_email = email
                    st.rerun()
                else:
                    st.error("Please enter a valid email address")

else:
    # Main application
    st.title("🎥 Video Annotation Tool")
    
    # User info in header
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**Workspace:** {st.session_state.user_email}")
    with col2:
        if st.session_state.current_project_name:
            st.markdown(f"**Current Project:** {st.session_state.current_project_name}")
    with col3:
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("🗂️ Project Management")
        
        # Get user projects
        projects = get_user_projects(st.session_state.user_id)
        
        # Project selection
        if projects:
            project_names = ["Select a project..."] + [p[1] for p in projects]
            selected_project = st.selectbox("Select Project", project_names)
            
            if selected_project != "Select a project...":
                project_data = next(p for p in projects if p[1] == selected_project)
                if st.button("Open Project"):
                    st.session_state.current_project_id = project_data[0]
                    st.session_state.current_project_name = project_data[1]
                    st.session_state.annotations = {}
                    st.session_state.video_path = None
                    st.rerun()
        
        # Create new project
        st.subheader("Create New Project")
        new_project_name = st.text_input("Project Name")
        if st.button("Create Project"):
            if new_project_name:
                project_id = create_project(st.session_state.user_id, new_project_name)
                if project_id:
                    st.session_state.current_project_id = project_id
                    st.session_state.current_project_name = new_project_name
                    st.success(f"Created project: {new_project_name}")
                    st.rerun()
                else:
                    st.error("Project name already exists!")
        
        st.divider()
        
        # Video management (only show if project is selected)
        if st.session_state.current_project_id:
            st.header("📹 Video Management")
            
            # Upload new video
            video_file = st.file_uploader("Upload new video", type=['mp4', 'avi', 'mov', 'mkv'])
            if video_file is not None:
                if st.button("Add to Project"):
                    video_path = save_video_to_project(
                        st.session_state.user_id,
                        st.session_state.current_project_id,
                        video_file
                    )
                    st.success(f"Added {video_file.name} to project")
                    st.rerun()
            
            # List project videos
            st.subheader("Project Videos")
            videos = get_project_videos(st.session_state.user_id, st.session_state.current_project_id)
            
            if videos:
                for video_path in videos:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(video_path.name)
                    with col2:
                        if st.button("Load", key=f"load_{video_path.name}"):
                            if load_video(str(video_path)) > 0:
                                st.session_state.annotations = load_project_annotations(
                                    st.session_state.current_project_id,
                                    video_path.name
                                )
                                st.rerun()
                            else:
                                st.error("Unable to load video")
            else:
                st.info("No videos in project yet")
            
            st.divider()
            
            # Class management
            st.header("🏷️ Class Management")
            
            # Current classes
            st.subheader("Current Classes")
            for i, cls in enumerate(st.session_state.classes):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(cls)
                with col2:
                    if st.button("❌", key=f"del_{i}"):
                        if len(st.session_state.classes) > 1:
                            st.session_state.classes.pop(i)
                            st.rerun()
            
            # Add new class
            new_class = st.text_input("Add new class")
            if st.button("➕ Add Class") and new_class:
                if new_class not in st.session_state.classes:
                    st.session_state.classes.append(new_class)
                    st.rerun()
            
            # Select current class
            st.session_state.current_class = st.selectbox(
                "Current annotation class",
                st.session_state.classes
            )

            st.divider()

            # Detection with YOLOv8
            st.header("🧠 Run Detection")
            st.session_state.detector_conf = st.slider(
                "Detection Confidence",
                0.0,
                1.0,
                st.session_state.detector_conf,
                step=0.05,
            )
            if st.button("Run YOLOv8 Detection"):
                if st.session_state.video_path:
                    with st.spinner("Running detection..."):
                        anns, counts = run_model_detection(
                            st.session_state.video_path,
                            st.session_state.detector_conf,
                        )
                    st.session_state.annotations = anns
                    st.session_state.detection_counts = counts
                    st.success("Detection complete")
                    st.rerun()
                else:
                    st.warning("Load a video first")

            st.divider()

            # Export annotations
            st.header("💾 Export Annotations")
            if st.button("Export as PASCAL VOC XML"):
                if st.session_state.annotations and st.session_state.video_path:
                    video_name = Path(st.session_state.video_path).stem
                    frame = get_frame(st.session_state.video_path, 0)
                    if frame is not None:
                        xml_map = generate_pascal_voc_xml(
                            st.session_state.annotations,
                            video_name,
                            frame.shape
                        )
                        if xml_map:
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                                for frame_num, xml_str in xml_map.items():
                                    xml_filename = f"{video_name}_frame_{frame_num}.xml"
                                    zipf.writestr(xml_filename, xml_str)
                            zip_buffer.seek(0)

                            st.download_button(
                                label="📥 Download XML ZIP",
                                data=zip_buffer.getvalue(),
                                file_name=f"{video_name}_annotations.zip",
                                mime="application/zip"
                            )
                else:
                    st.warning("No annotations to export!")
    
    # Main content area
    if st.session_state.current_project_id is None:
        st.info("👈 Please select or create a project to begin")
    elif st.session_state.video_path and st.session_state.total_frames > 0:
        video_name = Path(st.session_state.video_path).name
        st.subheader(f"Annotating: {video_name}")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Frame slider
            frame_num = st.slider(
                "Select Frame",
                0,
                st.session_state.total_frames - 1,
                st.session_state.current_frame_num,
                key="frame_slider"
            )
            st.session_state.current_frame_num = frame_num
        
        with col2:
            st.metric("Frame", f"{frame_num + 1} / {st.session_state.total_frames}")
        
        # Get current frame
        current_frame = get_frame(st.session_state.video_path, frame_num)
        
        if current_frame is not None:
            # --- Interactive bounding box annotation ---
            st.subheader("Draw Bounding Boxes (Interactive)")
            pil_img = Image.fromarray(current_frame)
            canvas_height, canvas_width = pil_img.height, pil_img.width
            frame_annotations = st.session_state.annotations.get(frame_num, [])
            # Prepare initial rectangles for st_canvas
            initial_rects = []
            for ann in frame_annotations:
                x1, y1, x2, y2 = ann['bbox']
                initial_rects.append({
                    "type": "rect",
                    "left": x1,
                    "top": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "stroke": "#000000",
                    "fill": "rgba(255, 165, 0, 0.3)",
                    "name": ann['class']
                })
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#000000",
                background_image=pil_img,
                drawing_mode="rect",
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                key=f"canvas_{frame_num}",
                initial_drawing={"version": "4.4.0", "objects": initial_rects, "background": {}}
            )
            # Parse rectangles from canvas
            new_annotations = []
            if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                for i, obj in enumerate(canvas_result.json_data["objects"]):
                    if obj["type"] == "rect":
                        x1 = int(obj["left"])
                        y1 = int(obj["top"])
                        x2 = int(obj["left"] + obj["width"])
                        y2 = int(obj["top"] + obj["height"])
                        # Assign class per box (default to current_class, or use previous if available)
                        label_key = f"label_{frame_num}_{i}"
                        default_class = obj.get("name", st.session_state.current_class)
                        box_class = st.selectbox(
                            f"Class for Box {i+1}",
                            st.session_state.classes,
                            index=st.session_state.classes.index(default_class) if default_class in st.session_state.classes else 0,
                            key=label_key
                        )
                        new_annotations.append({
                            'class': box_class,
                            'bbox': [x1, y1, x2, y2]
                        })
            # Save annotations if changed
            if new_annotations != frame_annotations:
                st.session_state.annotations[frame_num] = new_annotations
                save_annotations(
                    st.session_state.current_project_id,
                    video_name,
                    frame_num,
                    new_annotations
                )
            # Clear all boxes button
            if st.button("Clear All Boxes", key=f"clear_{frame_num}"):
                st.session_state.annotations[frame_num] = []
                save_annotations(
                    st.session_state.current_project_id,
                    video_name,
                    frame_num,
                    []
                )
                st.rerun()
            
            # Statistics
            st.divider()
            total_annotations = sum(len(anns) for anns in st.session_state.annotations.values())
            annotated_frames = len(st.session_state.annotations)
            current_detections = st.session_state.detection_counts.get(frame_num, 0)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Annotations", total_annotations)
            with col2:
                st.metric("Annotated Frames", annotated_frames)
            with col3:
                st.metric("Remaining Frames", st.session_state.total_frames - annotated_frames)
            with col4:
                st.metric("Detections (Frame)", current_detections)
    
    elif st.session_state.current_project_id:
        st.info("👈 Please select a video from the sidebar to begin annotation")