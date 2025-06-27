import os
import uuid
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
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
MAX_CONCURRENT_REQUESTS = 5  # Increased simultaneous API calls
MAX_QUEUE_SIZE = 15  # Increased allowed queue size

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
        self.retry_delay = 3  # Reduced base delay
        self.max_retry_delay = 30  # Reduced max delay
        self.consecutive_failures = 0
        self.lock = threading.Lock()
        
    def should_delay(self):
        with self.lock:
            elapsed = time.time() - self.last_request_time
            required_delay = min(self.retry_delay * (1.5 ** self.consecutive_failures), self.max_retry_delay)
            return max(0, required_delay - elapsed)
        
    def record_success(self):
        with self.lock:
            self.last_request_time = time.time()
            self.consecutive_failures = 0
            # Reset delay after success
            self.retry_delay = max(3, self.retry_delay // 1.5)
            
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
            task = request_queue.get(timeout=0.5)  # Shorter timeout
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
    # Initialize chat histories if not present
    if 'chats' not in session:
        session['chats'] = {
            'summarize': [{
                'role': 'ai',
                'content': "Hello! I'm your Summarize assistant. Paste any text and I'll create a concise summary for you."
            }],
            'explain': [{
                'role': 'ai',
                'content': "Hi there! I'm your Explain assistant. Give me any concept or topic and I'll explain it clearly."
            }],
            'translate': [{
                'role': 'ai',
                'content': "Bonjour! I'm your Translation assistant. I can translate between 50+ languages with cultural context."
            }],
            'code': [{
                'role': 'ai',
                'content': "Hello, developer! I'm your Code assistant. I can help with code generation, debugging, and explanations."
            }],
            'creative': [{
                'role': 'ai',
                'content': "Hello creative mind! I'm your Creative assistant. Let's brainstorm ideas, write stories, or create art concepts."
            }]
        }
    
    feedback_success = request.args.get('feedback') == 'success'
    queue_size = request_queue.qsize()
    return render_template('index.html', 
                          functions=FUNCTIONS, 
                          feedback_success=feedback_success,
                          model_name=MODEL_NAME,
                          queue_size=queue_size,
                          chats=session.get('chats', {}))

@app.route('/process', methods=['POST'])
def process_request():
    """Queue user request for processing with OpenAI API."""
    function_id = request.form.get('function')
    user_input = request.form.get('user_input', '').strip()
    
    # Validate input
    if not function_id or function_id not in [f['id'] for f in FUNCTIONS]:
        return jsonify({'error': 'Invalid function selection'}), 400
    
    if not user_input:
        return jsonify({'error': 'Please enter your request'}), 400
    
    if len(user_input) > MAX_INPUT_LENGTH:
        return jsonify({'error': f'Input too long. Maximum {MAX_INPUT_LENGTH} characters allowed.'}), 400
    
    # Check queue size
    if request_queue.qsize() >= MAX_QUEUE_SIZE:
        return jsonify({'error': 'Our servers are busy. Please try again in a few minutes.'}), 429
    
    # Create unique session ID for this request
    session_id = str(uuid.uuid4())
    queue_position = request_queue.qsize() + 1
    
    # Add user message to chat history
    if 'chats' not in session:
        session['chats'] = {}
    
    if function_id not in session['chats']:
        session['chats'][function_id] = []
    
    session['chats'][function_id].append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id
    })
    session.modified = True
    
    # Add to processing queue
    request_queue.put((session_id, function_id, user_input, lambda result: process_result(result)))
    
    return jsonify({
        'session_id': session_id,
        'queue_position': queue_position,
        'status': 'queued'
    })

def process_result(result):
    """Callback to handle API results and update session"""
    session_id = result['session_id']
    function_id = result.get('func_id')
    
    if not function_id or 'chats' not in session or function_id not in session['chats']:
        return
    
    if result['status'] == 'success':
        # Add AI response to chat history
        session['chats'][function_id].append({
            'role': 'ai',
            'content': result['ai_response'],
            'model_used': result['model_used'],
            'processing_time': result['processing_time'],
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id  # 👈 important
        })
    else:
        # Add error message to chat history
        session['chats'][function_id].append({
            'role': 'error',
            'content': result.get('error', 'An unknown error occurred'),
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        })
    
    session.modified = True

@app.route('/get_result/<session_id>')
def get_result(session_id):
    for func_id, chats in session.get('chats', {}).items():
        for chat in reversed(chats):  # Latest first
            if chat.get('session_id') == session_id:
                if chat['role'] == 'ai':
                    return jsonify({
                        'status': 'completed',
                        'message': chat['content']
                    })
                elif chat['role'] == 'error':
                    return jsonify({
                        'status': 'error',
                        'message': chat['content']
                    })
    return jsonify({'status': 'pending'})


@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """Process user feedback submissions."""
    request_id = request.form.get('request_id')
    rating = request.form.get('rating')
    comments = request.form.get('comments', '')
    
    # Validate input
    if not rating or rating not in ['good', 'neutral', 'poor']:
        return jsonify({'error': 'Invalid feedback rating'}), 400
    
    # Save feedback
    save_feedback({
        'request_id': request_id,
        'rating': rating,
        'comments': comments,
        'timestamp': datetime.now().isoformat()
    })
    
    return jsonify({'status': 'success'})

@app.teardown_appcontext
def shutdown_worker(exception=None):
    """Clean up worker thread on shutdown"""
    stop_worker.set()
    try:
        worker_thread.join(timeout=2.0)
    except RuntimeError:
        pass

@app.route('/api/status')
def api_status():
    """Endpoint to get current API status"""
    return jsonify({
        'queue_size': request_queue.qsize(),
        'consecutive_failures': rate_limit_manager.consecutive_failures,
        'retry_delay': rate_limit_manager.retry_delay,
        'status': 'normal' if rate_limit_manager.consecutive_failures == 0 else 'delayed'
    })

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode, threaded=True)