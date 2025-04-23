import sqlite3
import os
from datetime import datetime

def save_initial_profile(name, language_level, hobbies, db_path="./student_data.db"):
    """
    Save a student profile to a SQLite database.
    
    Parameters:
    - name: Student's name (string)
    - language_level: Student's language level (string, e.g., "Beginner A1")
    - hobbies: List of up to 3 hobbies (list of strings)
    - db_path: Path where to save the SQLite database (default: "./student_data.db")
    
    Returns:
    - student_id: ID of the newly inserted student
    """
    # Create directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        language_level TEXT NOT NULL,
        registration_date DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS hobbies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        hobby TEXT NOT NULL,
        FOREIGN KEY (student_id) REFERENCES students(id)
    )
    ''')
    
    # Insert student
    cursor.execute(
        'INSERT INTO students (name, language_level) VALUES (?, ?)',
        (name, language_level)
    )
    student_id = cursor.lastrowid
    
    # Insert hobbies
    for hobby in hobbies[:3]:
        if hobby.strip():
            cursor.execute(
                'INSERT INTO hobbies (student_id, hobby) VALUES (?, ?)',
                (student_id, hobby)
            )
    
    # Save and close
    conn.commit()
    conn.close()
    
    return student_id