import os
import uuid
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_session import Session
from dotenv import load_dotenv
import openai
from prompts import get_system_prompt
from feedback_manager import save_feedback
import threading
import queue
import random

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey')

# Configure server-side sessions
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.addHandler(logging.StreamHandler())

# Initialize OpenAI clients with multiple keys if available
API_KEYS = os.getenv('OPENAI_API_KEYS', os.getenv('OPENAI_API_KEY', '')).split(',')
clients = [openai.OpenAI(api_key=key.strip()) for key in API_KEYS if key.strip()]
current_key_index = 0

# Use GPT-3.5-turbo as default since it has higher rate limits
MODEL_NAME = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
MAX_INPUT_LENGTH = 5000  # Character limit for user input
MAX_CONCURRENT_REQUESTS = 3  # Limit simultaneous API calls
MAX_QUEUE_SIZE = 10  # Maximum allowed queue size

# Request queue and worker thread
request_queue = queue.Queue()
stop_worker = threading.Event()

# Define available functions with metadata
FUNCTIONS = [
    {"id": "summarize", "name": "Summarize", "icon": "bi-file-text", 
     "description": "Condense long texts into key points"},
    {"id": "explain", "name": "Explain", "icon": "bi-journal-text", 
     "description": "Get clear explanations of concepts"},
    {"id": "translate", "name": "Translate", "icon": "bi-translate", 
     "description": "Translate text to another language"},
    {"id": "code", "name": "Code Assistant", "icon": "bi-code-slash", 
     "description": "Generate or explain code snippets"},
    {"id": "creative", "name": "Creative Writer", "icon": "bi-pencil", 
     "description": "Generate stories, poems, and more"}
]

# --- Rate Limit Handling System ---
class RateLimitManager:
    def __init__(self):
        self.last_request_time = 0
        self.retry_delay = 5  # Base delay in seconds
        self.max_retry_delay = 60
        self.consecutive_failures = 0
        self.lock = threading.Lock()
        
    def should_delay(self):
        with self.lock:
            elapsed = time.time() - self.last_request_time
            required_delay = min(self.retry_delay * (2 ** self.consecutive_failures), self.max_retry_delay)
            return max(0, required_delay - elapsed)
        
    def record_success(self):
        with self.lock:
            self.last_request_time = time.time()
            self.consecutive_failures = 0
            # Reset delay after success
            self.retry_delay = max(5, self.retry_delay // 2)
            
    def record_failure(self):
        with self.lock:
            self.consecutive_failures += 1
            self.retry_delay = min(self.retry_delay * 1.5, self.max_retry_delay)
            self.last_request_time = time.time()

rate_limit_manager = RateLimitManager()

def api_worker():
    """Background worker to process API requests with rate limit handling"""
    while not stop_worker.is_set():
        try:
            task = request_queue.get(timeout=1)
            if task is None:
                continue
                
            session_id, function_id, user_input, callback = task
            
            # Check rate limits before processing
            required_delay = rate_limit_manager.should_delay()
            if required_delay > 0:
                time.sleep(required_delay)
                
            try:
                start_time = time.time()
                system_prompt = get_system_prompt(function_id)
                
                # Rotate API keys
                global current_key_index
                with threading.Lock():
                    client = clients[current_key_index % len(clients)]
                    current_key_index = (current_key_index + 1) % len(clients)
                
                # Make API request
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                ai_response = response.choices[0].message.content.strip()
                processing_time = round(time.time() - start_time, 2)
                rate_limit_manager.record_success()
                
                # Prepare result
                func_meta = next((f for f in FUNCTIONS if f["id"] == function_id), None)
                result = {
                    'status': 'success',
                    'session_id': session_id,
                    'func_id': function_id,
                    'func_name': func_meta["name"] if func_meta else "Unknown",
                    'func_icon': func_meta["icon"] if func_meta else "bi-question",
                    'model_used': MODEL_NAME,
                    'user_input': user_input,
                    'ai_response': ai_response,
                    'processing_time': processing_time
                }
                
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                rate_limit_manager.record_failure()
                result = {
                    'status': 'retry',
                    'session_id': session_id,
                    'error': f"API rate limit exceeded. Please try again in {rate_limit_manager.retry_delay} seconds."
                }
            except openai.NotFoundError:
                # Handle model not found error
                result = {
                    'status': 'error',
                    'session_id': session_id,
                    'error': f"The model '{MODEL_NAME}' is not available. Please try a different model."
                }
            except openai.APIError as e:
                result = {
                    'status': 'error',
                    'session_id': session_id,
                    'error': f"OpenAI API error: {str(e)}"
                }
            except Exception as e:
                result = {
                    'status': 'error',
                    'session_id': session_id,
                    'error': f"Unexpected error: {str(e)}"
                }
                
            # Send result back to main thread
            callback(result)
            request_queue.task_done()
            
        except queue.Empty:
            continue

# Start worker thread
worker_thread = threading.Thread(target=api_worker, daemon=True)
worker_thread.start()

@app.route('/')
def index():
    """Render the homepage with function cards."""
    feedback_success = request.args.get('feedback') == 'success'
    queue_size = request_queue.qsize()
    return render_template('index.html', 
                          functions=FUNCTIONS, 
                          feedback_success=feedback_success,
                          model_name=MODEL_NAME,
                          queue_size=queue_size)

@app.route('/process', methods=['POST'])
def process_request():
    """Queue user request for processing with OpenAI API."""
    selected_functions = request.form.getlist('function')
    user_input = request.form.get('user_input', '').strip()
    
    # Validate input
    if len(selected_functions) != 1:
        flash('Please select exactly one function', 'danger')
        return redirect(url_for('index'))
    
    function_id = selected_functions[0]
    
    if not user_input:
        flash('Please enter your request', 'danger')
        return redirect(url_for('index'))
    
    if len(user_input) > MAX_INPUT_LENGTH:
        flash(f'Input too long. Maximum {MAX_INPUT_LENGTH} characters allowed.', 'danger')
        return redirect(url_for('index'))
    
    # Check queue size
    if request_queue.qsize() >= MAX_QUEUE_SIZE:
        flash('Our servers are busy. Please try again in a few minutes.', 'warning')
        return redirect(url_for('index'))
    
    # Create unique session ID for this request
    session_id = str(uuid.uuid4())
    queue_position = request_queue.qsize() + 1
    
    # Store minimal session data
    session[session_id] = {
        'function': function_id,
        'user_input': user_input,
        'timestamp': datetime.now().isoformat(),
        'status': 'queued',
        'queue_position': queue_position
    }
    
    # Add to processing queue
    request_queue.put((session_id, function_id, user_input, lambda result: process_result(result)))
    
    # Render a simple processing page with auto-refresh
    return render_processing_page(session_id, queue_position)

def render_processing_page(session_id, queue_position):
    """Render a simple processing page with auto-refresh"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Processing - Inquiro AI Assistant</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
        <meta http-equiv="refresh" content="3; url=/result/{session_id}">
        <style>
            body {{
                background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
            }}
            .processing-card {{
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                background: white;
                padding: 30px;
                text-align: center;
            }}
            .spinner {{
                width: 4rem;
                height: 4rem;
                margin: 20px auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="processing-card">
                        <h2><i class="bi bi-cpu"></i> Processing Your Request</h2>
                        <div class="spinner-border text-primary spinner" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p>We're working on your request with our AI engine</p>
                        <div class="progress mt-4">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                role="progressbar" style="width: 75%"></div>
                        </div>
                        <div class="mt-4">
                            <span class="badge bg-primary">
                                <i class="bi bi-arrow-left-right"></i> Queue Position: {queue_position}
                            </span>
                        </div>
                        <p class="mt-4 text-muted">
                            <i class="bi bi-info-circle"></i> 
                            This page will refresh automatically. Please keep it open.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def process_result(result):
    """Callback to handle API results and update session"""
    session_id = result['session_id']
    
    if result['status'] == 'success':
        # Store full session data
        session[session_id] = {
            **session.get(session_id, {}),
            'ai_response': result['ai_response'],
            'model_used': result['model_used'],
            'status': 'completed',
            'processing_time': result['processing_time']
        }
    elif result['status'] == 'retry':
        session[session_id] = {
            **session.get(session_id, {}),
            'status': 'retry',
            'error': result['error']
        }
    else:
        session[session_id] = {
            **session.get(session_id, {}),
            'status': 'error',
            'error': result['error']
        }

@app.route('/result/<session_id>')
def show_result(session_id):
    """Show processing result for a session"""
    request_data = session.get(session_id, {})
    
    if not request_data:
        flash('Session not found. Please submit a new request.', 'danger')
        return redirect(url_for('index'))
    
    status = request_data.get('status', 'unknown')
    
    if status == 'queued':
        # Re-render processing page with current position
        queue_position = request_data.get('queue_position', request_queue.qsize() + 1)
        return render_processing_page(session_id, queue_position)
    
    if status == 'retry':
        flash(request_data.get('error', 'API rate limit exceeded. Please try again later.'), 'danger')
        return redirect(url_for('index'))
    
    if status == 'error':
        error_message = request_data.get('error', 'An unknown error occurred.')
        return render_template('result.html', 
                              request_id=session_id,
                              func_name="Error",
                              func_icon="bi-exclamation-circle",
                              model_used=MODEL_NAME,
                              user_input=request_data.get('user_input', ''),
                              ai_response=error_message)
    
    # Successful result
    func_meta = next((f for f in FUNCTIONS if f["id"] == request_data.get('function')), None)
    
    return render_template('result.html', 
                          request_id=session_id,
                          func_id=request_data.get('function'),
                          func_name=func_meta["name"] if func_meta else "Unknown",
                          func_icon=func_meta["icon"] if func_meta else "bi-question",
                          model_used=request_data.get('model_used', MODEL_NAME),
                          user_input=request_data.get('user_input', ''),
                          ai_response=request_data.get('ai_response', ''),
                          processing_time=request_data.get('processing_time', 0))

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """Process user feedback submissions."""
    request_id = request.form.get('request_id')
    
    # Retrieve original request data from session
    request_data = session.get(request_id, {})
    
    if not request_data:
        flash('Could not find the original request. Feedback not saved.', 'warning')
        return redirect(url_for('index'))
    
    # Prepare feedback data
    feedback_data = {
        'request_id': request_id,
        'function': request_data.get('function'),
        'user_input': request_data.get('user_input'),
        'ai_response': request_data.get('ai_response'),
        'model_used': request_data.get('model_used', 'unknown'),
        'rating': request.form.get('rating'),
        'comments': request.form.get('comments', ''),
        'timestamp': request_data.get('timestamp')
    }
    
    # Save feedback to file
    save_feedback(feedback_data)
    
    # Clean up session data
    session.pop(request_id, None)
    
    # Redirect with success message
    return redirect(url_for('index') + '?feedback=success')

@app.teardown_appcontext
def shutdown_worker(exception=None):
    """Clean up worker thread on shutdown"""
    stop_worker.set()
    try:
        worker_thread.join(timeout=5.0)
    except RuntimeError:
        pass

@app.route('/api/status')
def api_status():
    """Endpoint to get current API status"""
    return {
        'queue_size': request_queue.qsize(),
        'consecutive_failures': rate_limit_manager.consecutive_failures,
        'retry_delay': rate_limit_manager.retry_delay,
        'status': 'normal' if rate_limit_manager.consecutive_failures == 0 else 'delayed'
    }

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode, threaded=True)