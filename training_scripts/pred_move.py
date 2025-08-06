"""
pred_move.py

Moves all but the two most recently modified files from the prediction directory to a dated archive folder.
This helps keep the prediction folder clean while preserving past prediction results.

Author: Lance Santerre
"""

import os
import shutil
from datetime import datetime, timedelta


def move_old_files_except_latest_two():
    """
    Moves all files from the prediction directory except the two most recent ones
    into a dated subfolder within the old_pred directory.
    """
    folder_path = "/Users/lancesanterre/so_predict/app/data"
    dest_base_path = "/Users/lancesanterre/so_predict/data/old_pred"

    # Get yesterday's date in YYYY-MM-DD format
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Destination folder path
    dest_folder = os.path.join(dest_base_path, yesterday_str)
    os.makedirs(dest_folder, exist_ok=True)

    # Get list of all files and their modified time
    files = [
        (f, os.path.getmtime(os.path.join(folder_path, f)))
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        or os.path.isdir(os.path.join(folder_path, f))
    ]

    # Sort files by last modified time (newest last)
    files.sort(key=lambda x: x[1], reverse=True)

    # Keep the two newest files
    files_to_move = files[2:]

    for filename, _ in files_to_move:
        file_path = os.path.join(folder_path, filename)
        dest_path = os.path.join(dest_folder, filename)
        try:
            shutil.move(file_path, dest_path)
            print(f"✅ Moved: {filename} ➡️ {dest_path}")
        except Exception as e:
            print(f"⚠️ Failed to move {filename}: {e}")

    print(f"\n📁 Older files moved to: {dest_folder}")


# Run the cleanup
move_old_files_except_latest_two()
