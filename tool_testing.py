import os
import csv
import glob
db_path="./student_data"

if not os.path.exists(db_path):
    print(f"Directory not found: {db_path}")


# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(db_path, "*.csv"))

if not csv_files:
    print(f"No student data files found in {db_path}")


student_count = 0

# Process each file
for file_path in csv_files:
    filename = os.path.basename(file_path)
    print(f"\n===== Student File: {filename} =====")
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                student_count += 1
                print(f"\nStudent ID: {row['id']}")
                print(f"Name: {row['name']}")
                print(f"Language Level: {row['language_level']}")
                print(f"Registration Date: {row['registration_date']}")
                
                # Handle hobbies - convert from pipe-separated to list
                hobbies = row['hobbies'].split('|') if row['hobbies'] else []
                if hobbies:
                    print(f"Hobbies: {', '.join(hobbies)}")
                else:
                    print("Hobbies: None")
    except Exception as e:
        print(f"Error reading file {filename}: {e}")

print(f"\nTotal students found: {student_count}")
