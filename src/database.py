import sqlite3
import json
from pathlib import Path

STORAGE_DIR = Path("annotation_storage")
STORAGE_DIR.mkdir(exist_ok=True)


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
        return [
            f
            for f in project_dir.iterdir()
            if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.3gp']
        ]
    return []
