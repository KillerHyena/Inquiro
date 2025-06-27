import json
import os
import tempfile
import shutil
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('feedback_manager')

FEEDBACK_DIR = "data/feedback"
JSONL_FILE = os.path.join(FEEDBACK_DIR, "feedback_{date}.jsonl")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file

def save_feedback(feedback_data):
    """Save feedback data to JSONL files with rotation and atomic writes."""
    try:
        # Add timestamp if not provided
        feedback_data.setdefault('timestamp', datetime.now().isoformat())
        
        # Ensure directory exists
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        
        # Get current date-based filename
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = JSONL_FILE.format(date=current_date)
        
        # Rotate file if it's too large
        if os.path.exists(filename) and os.path.getsize(filename) > MAX_FILE_SIZE:
            rotate_file(filename)
        
        # Create JSONL line
        json_line = json.dumps(feedback_data, ensure_ascii=False)
        
        # Atomic write to prevent corruption
        write_atomic(filename, json_line + '\n')
        
        logger.info(f"Feedback saved successfully for request: {feedback_data.get('request_id', 'unknown')}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save feedback: {str(e)}")
        # Fallback to temp file if primary save fails
        save_to_fallback(json_line)
        return False

def write_atomic(filename, content):
    """Write to file atomically to prevent partial writes on failure"""
    try:
        # Write to temp file first
        with tempfile.NamedTemporaryFile(mode='w', 
                                         dir=os.path.dirname(filename),
                                         delete=False) as tmp_file:
            tmp_file.write(content)
        
        # Atomic rename (works on Unix and Windows)
        os.replace(tmp_file.name, filename)
        
    except OSError as e:
        logger.error(f"Atomic write failed: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)
        raise

def rotate_file(filename):
    """Rotate files when they become too large"""
    try:
        base_name = os.path.basename(filename).replace('.jsonl', '')
        dir_name = os.path.dirname(filename)
        
        # Find next rotation number
        counter = 1
        while True:
            rotated_name = os.path.join(dir_name, f"{base_name}_{counter}.jsonl")
            if not os.path.exists(rotated_name):
                break
            counter += 1
        
        # Rotate the file
        shutil.move(filename, rotated_name)
        logger.info(f"Rotated feedback file to: {rotated_name}")
        
    except Exception as e:
        logger.error(f"File rotation failed: {str(e)}")

def save_to_fallback(json_line):
    """Save feedback to fallback location when primary fails"""
    try:
        fallback_file = os.path.join(FEEDBACK_DIR, "feedback_fallback.jsonl")
        with open(fallback_file, 'a', encoding='utf-8') as f:
            f.write(json_line + '\n')
        logger.warning(f"Saved feedback to fallback file: {fallback_file}")
    except Exception as e:
        logger.error(f"Critical: Failed to save to fallback: {str(e)}")
        # Last resort - print to stderr
        print(f"CRITICAL FEEDBACK LOSS: {json_line}")

def load_feedback():
    """Load all feedback data from files (for admin/reporting)"""
    feedback = []
    try:
        # Get all feedback files
        files = [f for f in os.listdir(FEEDBACK_DIR) 
                if f.startswith('feedback_') and f.endswith('.jsonl')]
        
        for filename in sorted(files):
            full_path = os.path.join(FEEDBACK_DIR, filename)
            with open(full_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        feedback.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in {filename}: {line}")
    
    except Exception as e:
        logger.error(f"Failed to load feedback: {str(e)}")
    
    return feedback