# migrate_db.py
import sqlite3

conn = sqlite3.connect('posture_detection.db')
cursor = conn.cursor()

# Add missing columns to posture_detections table
try:
    cursor.execute('ALTER TABLE posture_detections ADD COLUMN user_id INTEGER')
    cursor.execute('ALTER TABLE posture_detections ADD COLUMN session_id INTEGER')
    cursor.execute('ALTER TABLE posture_detections ADD COLUMN detection_type TEXT DEFAULT "image"')
    cursor.execute('ALTER TABLE posture_detections ADD COLUMN bbox_x1 REAL')
    cursor.execute('ALTER TABLE posture_detections ADD COLUMN bbox_y1 REAL')
    cursor.execute('ALTER TABLE posture_detections ADD COLUMN bbox_x2 REAL')
    cursor.execute('ALTER TABLE posture_detections ADD COLUMN bbox_y2 REAL')
    conn.commit()
    print("Migration successful!")
except sqlite3.OperationalError as e:
    print(f"Migration error (columns may already exist): {e}")
finally:
    conn.close()