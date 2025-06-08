import streamlit as st
import cv2
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import os
import tempfile
from datetime import datetime
import json

# Initialize session state
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}
if 'classes' not in st.session_state:
    st.session_state.classes = ['good-cup', 'bad-cup', 'no-cup']
if 'current_class' not in st.session_state:
    st.session_state.current_class = 'good-cup'
if 'drawing' not in st.session_state:
    st.session_state.drawing = False
if 'start_point' not in st.session_state:
    st.session_state.start_point = None
if 'temp_box' not in st.session_state:
    st.session_state.temp_box = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'total_frames' not in st.session_state:
    st.session_state.total_frames = 0
if 'current_frame_num' not in st.session_state:
    st.session_state.current_frame_num = 0

def load_video(video_file):
    """Load video and save to temporary file"""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    st.session_state.video_path = tfile.name
    st.session_state.total_frames = total_frames
    st.session_state.current_frame_num = 0
    st.session_state.annotations = {}
    
    return tfile.name

def get_frame(video_path, frame_number):
    """Extract specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None

def draw_annotations(image, annotations):
    """Draw bounding boxes on image"""
    img_with_boxes = image.copy()
    
    # Define colors for different classes
    colors = {
        'good-cup': (0, 255, 0),
        'bad-cup': (255, 0, 0),
        'no-cup': (255, 255, 0)
    }
    
    for ann in annotations:
        x1, y1, x2, y2 = ann['bbox']
        class_name = ann['class']
        color = colors.get(class_name, (0, 0, 255))
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img_with_boxes, (x1, y1 - text_height - 4), (x1 + text_width + 4, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_with_boxes, label, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), thickness)
    
    return img_with_boxes

def generate_pascal_voc_xml(annotations_dict, video_name, output_path):
    """Generate PASCAL VOC format XML"""
    root = ET.Element("annotations")
    root.set("version", "1.0")
    
    for frame_num, frame_annotations in annotations_dict.items():
        if not frame_annotations:
            continue
            
        annotation = ET.SubElement(root, "annotation")
        
        # Add folder
        folder = ET.SubElement(annotation, "folder")
        folder.text = "frames"
        
        # Add filename
        filename = ET.SubElement(annotation, "filename")
        filename.text = f"{video_name}_frame_{frame_num}.jpg"
        
        # Add source
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Custom Video Annotation"
        
        # Add size (we'll use the first frame's size for all)
        if st.session_state.video_path:
            frame = get_frame(st.session_state.video_path, 0)
            if frame is not None:
                h, w, c = frame.shape
                size = ET.SubElement(annotation, "size")
                width = ET.SubElement(size, "width")
                width.text = str(w)
                height = ET.SubElement(size, "height")
                height.text = str(h)
                depth = ET.SubElement(size, "depth")
                depth.text = str(c)
        
        # Add segmented
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"
        
        # Add objects
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
    
    # Create properly formatted XML string
    from xml.dom import minidom
    rough_string = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove extra blank lines
    lines = pretty_xml.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    pretty_xml = '\n'.join(non_empty_lines)
    
    return pretty_xml

# Streamlit UI
st.set_page_config(page_title="Video Annotation Tool", layout="wide")

st.title("🎥 Video Annotation Tool")
st.markdown("Upload a video, navigate to frames, and manually annotate objects with bounding boxes")

# Sidebar for controls
with st.sidebar:
    st.header("📁 Video Upload")
    video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if video_file is not None:
        if 'current_video_name' not in st.session_state or st.session_state.current_video_name != video_file.name:
            st.session_state.current_video_name = video_file.name
            load_video(video_file)
            st.success(f"Loaded: {video_file.name}")
            st.info(f"Total frames: {st.session_state.total_frames}")
    
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
        if st.session_state.annotations:
            video_name = os.path.splitext(st.session_state.current_video_name)[0]
            xml_content = generate_pascal_voc_xml(
                st.session_state.annotations,
                video_name,
                "annotations.xml"
            )
            
            st.download_button(
                label="📥 Download XML",
                data=xml_content,
                file_name=f"{video_name}_annotations.xml",
                mime="text/xml"
            )
        else:
            st.warning("No annotations to export!")

# Main content area
if st.session_state.video_path and st.session_state.total_frames > 0:
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
        # Get annotations for current frame
        frame_annotations = st.session_state.annotations.get(frame_num, [])
        
        # Draw existing annotations
        annotated_frame = draw_annotations(current_frame, frame_annotations)
        
        # Display frame
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.image(annotated_frame, use_column_width=True)
        
        with col2:
            st.subheader("Frame Annotations")
            
            if frame_annotations:
                for i, ann in enumerate(frame_annotations):
                    with st.container():
                        st.text(f"{i+1}. {ann['class']}")
                        st.text(f"   Box: {ann['bbox']}")
                        if st.button(f"Delete", key=f"del_ann_{frame_num}_{i}"):
                            frame_annotations.pop(i)
                            st.session_state.annotations[frame_num] = frame_annotations
                            st.rerun()
            else:
                st.info("No annotations yet")
            
            st.divider()
            
            # Manual input for precise annotation
            st.subheader("Add Annotation")
            with st.form(key=f"annotation_form_{frame_num}"):
                st.write(f"Class: {st.session_state.current_class}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    x1 = st.number_input("X1", min_value=0, max_value=current_frame.shape[1], value=0)
                    y1 = st.number_input("Y1", min_value=0, max_value=current_frame.shape[0], value=0)
                
                with col_b:
                    x2 = st.number_input("X2", min_value=0, max_value=current_frame.shape[1], value=100)
                    y2 = st.number_input("Y2", min_value=0, max_value=current_frame.shape[0], value=100)
                
                if st.form_submit_button("Add Box"):
                    new_annotation = {
                        'class': st.session_state.current_class,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    }
                    
                    if frame_num not in st.session_state.annotations:
                        st.session_state.annotations[frame_num] = []
                    
                    st.session_state.annotations[frame_num].append(new_annotation)
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

else:
    st.info("👆 Please upload a video file to begin annotation")