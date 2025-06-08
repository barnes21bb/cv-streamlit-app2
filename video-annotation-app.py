import streamlit as st
from PIL import Image
from pathlib import Path
import io
import zipfile
from streamlit_drawable_canvas import st_canvas

from src.database import (
    init_database,
    get_or_create_user,
    get_all_users,
    create_project,
    get_user_projects,
    save_video_to_project,
    save_annotations,
    load_project_annotations,
    get_project_videos,
)
from src.video_utils import load_video, get_frame
from src.annotation_utils import draw_annotations, generate_pascal_voc_xml
from src.utils import validate_email

# Initialize database on startup
init_database()

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
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Annotations", total_annotations)
            with col2:
                st.metric("Annotated Frames", annotated_frames)
            with col3:
                st.metric("Remaining Frames", st.session_state.total_frames - annotated_frames)
    
    elif st.session_state.current_project_id:
        st.info("👈 Please select a video from the sidebar to begin annotation")