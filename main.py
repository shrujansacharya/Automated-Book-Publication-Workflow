import asyncio
import os
import logging
import json
from playwright.async_api import async_playwright
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
import random
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ebooklib import epub
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configuration
URL = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
SCREENSHOT_DIR = "screenshots"
CONTENT_DIR = "content"
CHROMADB_DIR = "./chroma_db"
FRONTEND_DIR = "frontend"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(CHROMADB_DIR, exist_ok=True)
os.makedirs(FRONTEND_DIR, exist_ok=True)

# Initialize HuggingFace model
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    logger.error("HUGGINGFACEHUB_API_TOKEN not found in .env")
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env")
try:
    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-large",
        temperature=0.7,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token
    )
    logger.info("HuggingFace model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize HuggingFace model: {e}")
    raise

# Initialize ChromaDB
try:
    chroma_client = chromadb.Client(Settings(persist_directory=CHROMADB_DIR, anonymized_telemetry=False))
    collection = chroma_client.get_or_create_collection(name="book_versions")
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    raise

# Temporary storage for review states
review_states = {}

# --- Web Scraping ---
async def scrape_content_and_screenshot(url):
    logger.debug(f"Starting scrape for URL: {url}")
    try:
        async with async_playwright() as p:
            logger.debug("Launching browser")
            browser = await p.chromium.launch()
            page = await browser.new_page()
            logger.debug(f"Navigating to {url}")
            await page.goto(url)
            logger.debug("Extracting content")
            content = await page.locator("div.mw-parser-output").inner_text()
            logger.debug("Extracting images")
            images = await page.locator("img").all()
            alt_texts = [await img.get_attribute("alt") or "" for img in images]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(SCREENSHOT_DIR, f"chapter_1_{timestamp}.png")
            logger.debug(f"Saving screenshot to {screenshot_path}")
            await page.screenshot(path=screenshot_path)
            await browser.close()
            content_path = os.path.join(CONTENT_DIR, f"raw_chapter_1_{timestamp}.txt")
            logger.debug(f"Saving content to {content_path}")
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Scraped content saved to {content_path}, screenshot to {screenshot_path}")
            return content, screenshot_path, content_path, {"alt_texts": alt_texts}
    except Exception as e:
        logger.error(f"Error in scrape_content_and_screenshot: {str(e)}")
        raise

# --- Spin Content ---
def spin_content(content):
    logger.debug("Starting content spinning")
    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Rewrite the following text in a different style while preserving the meaning:\n{content}"
    )
    prompt = prompt_template.format(content=content[:500])
    try:
        result = llm.invoke(prompt)
        logger.info("Content spinning completed")
        return result
    except Exception as e:
        logger.error(f"Error in spin_content: {e}")
        return content

# --- Review Content ---
def review_content(content, alt_texts):
    logger.debug("Starting content review")
    prompt_template = PromptTemplate(
        input_variables=["content", "alt_texts"],
        template="Review and improve the following text for clarity, grammar, and coherence. For any missing alt texts, generate descriptive alt texts:\nContent: {content}\nMissing Alt Texts: {alt_texts}"
    )
    missing_alt_texts = [alt for alt in alt_texts if not alt]
    prompt = prompt_template.format(content=content[:500], alt_texts=str(missing_alt_texts))
    try:
        result = llm.invoke(prompt)
        logger.info("Content review completed")
        return result, missing_alt_texts
    except Exception as e:
        logger.error(f"Error in review_content: {e}")
        return content, missing_alt_texts

# --- Analyze Content ---
def analyze_content(content):
    logger.debug("Starting content analysis")
    readability = flesch_reading_ease(content)
    sentiment = SentimentIntensityAnalyzer().polarity_scores(content)
    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Summarize the following text in 100 words or less:\n{content}"
    )
    prompt = prompt_template.format(content=content[:1000])
    try:
        summary = llm.invoke(prompt)
        logger.info("Content analysis completed")
        return {"summary": summary, "readability": readability, "sentiment": sentiment}
    except Exception as e:
        logger.error(f"Error in analyze_content: {e}")
        return {"summary": "", "readability": readability, "sentiment": sentiment}

# --- RL Search ---
class RLSearch:
    def __init__(self, collection):
        self.collection = collection
        self.actions = ["query_by_id", "query_by_content", "query_by_metadata"]
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        logger.debug("RLSearch initialized")

    def get_state(self, query): return len(query)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        self.q_table.setdefault(state, {a: 0 for a in self.actions})
        return max(self.q_table[state], key=self.q_table[state].get)

    def query_collection(self, action, query, doc_id=None, metadata=None):
        logger.debug(f"Querying collection with action: {action}")
        try:
            if action == "query_by_id" and doc_id:
                return self.collection.get(ids=[doc_id])
            elif action == "query_by_content":
                return self.collection.query(query_texts=[query], n_results=1)
            elif action == "query_by_metadata" and metadata:
                return self.collection.get(where=metadata)
        except Exception as e:
            logger.error(f"Error in query_collection: {e}")
        return {"documents": [], "ids": [], "metadatas": []}

    def update_q_table(self, state, action, reward, next_state):
        self.q_table.setdefault(state, {a: 0 for a in self.actions})
        self.q_table.setdefault(next_state, {a: 0 for a in self.actions})
        current_q = self.q_table[state][action]
        future_q = max(self.q_table[next_state].values())
        self.q_table[state][action] += self.alpha * (reward + self.gamma * future_q - current_q)

    def search(self, query, doc_id=None, metadata=None):
        state = self.get_state(query)
        action = self.choose_action(state)
        results = self.query_collection(action, query, doc_id, metadata)
        reward = 1 if results["documents"] else -1
        self.update_q_table(state, action, reward, self.get_state(query))
        logger.info(f"Search completed with {len(results['documents'])} results")
        return results

# --- Save Version ---
def save_version(content, version_id, metadata):
    logger.debug(f"Saving version {version_id}")
    try:
        # Serialize accessibility_data to JSON string if present
        if 'accessibility_data' in metadata:
            metadata['accessibility_data'] = json.dumps(metadata['accessibility_data'])
        logger.debug(f"Serialized metadata: {metadata}")
        collection.add(documents=[content], metadatas=[metadata], ids=[version_id])
        logger.info(f"Version {version_id} saved")
    except Exception as e:
        logger.error(f"Error saving version: {e}")
        raise

# --- Flask API Endpoints ---
@app.route('/')
def serve_index():
    logger.debug("Serving index.html")
    index_path = os.path.join(FRONTEND_DIR, 'index.html')
    if not os.path.exists(index_path):
        logger.error(f"index.html not found at {index_path}")
        return jsonify({'status': 'error', 'message': f'index.html not found in {FRONTEND_DIR} folder'}), 500
    return send_file(index_path)

@app.route('/api/scrape', methods=['GET'])
async def scrape():
    logger.debug(f"Request: {request.url}")
    try:
        logger.debug("Calling scrape_content_and_screenshot")
        content, screenshot_path, content_path, accessibility_data = await scrape_content_and_screenshot(URL)
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        review_states[session_id] = {'raw_content': content, 'accessibility_data': accessibility_data}
        logger.info(f"Scrape successful, session_id: {session_id}")
        return jsonify({
            'status': 'success',
            'content': content[:500],
            'screenshot_filename': os.path.basename(screenshot_path),
            'content_path': content_path,
            'session_id': session_id,
            'accessibility_data': accessibility_data
        })
    except Exception as e:
        logger.error(f"Error in /api/scrape: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Scrape failed: {str(e)}"}), 500

@app.route('/api/status/<session_id>', methods=['GET'])
def get_status(session_id):
    logger.debug(f"Request: {request.url}")
    if session_id not in review_states:
        logger.error(f"Invalid session ID: {session_id}")
        return jsonify({'status': 'error', 'message': 'Invalid session ID'}), 400
    state = review_states[session_id]
    completed_steps = []
    if 'raw_content' in state:
        completed_steps.append('scrape')
    if 'spun_content' in state:
        completed_steps.append('spin')
    if 'reviewed_content' in state:
        completed_steps.append('review')
    if 'final_content' in state:
        completed_steps.append('edit')
    logger.info(f"Status for session {session_id}: {completed_steps}")
    return jsonify({
        'status': 'success',
        'session_id': session_id,
        'completed_steps': completed_steps
    })

@app.route('/api/process/<session_id>/<step>', methods=['POST'])
def process_content(session_id, step):
    logger.debug(f"Request: {request.url}")
    if session_id not in review_states:
        logger.error(f"Invalid session ID: {session_id}")
        return jsonify({'status': 'error', 'message': 'Invalid session ID'}), 400
    data = request.json
    content = data.get('content', '')
    approved = data.get('approved', False)
    state = review_states[session_id]
    try:
        if step == 'spin':
            spun = spin_content(state['raw_content'] if not content else content)
            state['spun_content'] = spun
            logger.info(f"Spin step completed for session {session_id}")
            return jsonify({'status': 'success', 'content': spun, 'role': 'writer'})
        elif step == 'review':
            if 'spun_content' not in state:
                logger.error("Spin step not completed")
                return jsonify({'status': 'error', 'message': 'Spin step not completed'}), 400
            reviewed, generated_alt_texts = review_content(
                state['spun_content'] if not content else content,
                state['accessibility_data']['alt_texts']
            )
            state['reviewed_content'] = reviewed
            state['accessibility_data']['generated_alt_texts'] = generated_alt_texts
            logger.info(f"Review step completed for session {session_id}")
            return jsonify({
                'status': 'success',
                'content': reviewed,
                'role': 'reviewer',
                'accessibility_data': state['accessibility_data']
            })
        elif step == 'edit':
            if 'reviewed_content' not in state:
                logger.error("Review step not completed")
                return jsonify({'status': 'error', 'message': 'Review step not completed'}), 400
            state['final_content'] = content if content else state['reviewed_content']
            version_id = f"chapter_1_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metadata = {
                "chapter": "1",
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "accessibility_data": state['accessibility_data'],
                "format": "raw"
            }
            save_version(state['final_content'], version_id, metadata)
            logger.info(f"Edit step completed for session {session_id}, version_id: {version_id}")
            return jsonify({
                'status': 'success',
                'content': state['final_content'],
                'version_id': version_id,
                'accessibility_data': state['accessibility_data']
            })
        else:
            logger.error(f"Invalid step: {step}")
            return jsonify({'status': 'error', 'message': 'Invalid step'}), 400
    except Exception as e:
        logger.error(f"Error in /api/process: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/export/<session_id>/<format>', methods=['GET'])
def export_content(session_id, format):
    logger.debug(f"Request: {request.url}")
    if session_id not in review_states:
        logger.error(f"Invalid session ID: {session_id}")
        return jsonify({'status': 'error', 'message': 'Invalid session ID'}), 400
    content = review_states[session_id].get('final_content', review_states[session_id]['raw_content'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if format == 'epub':
            book = epub.EpubBook()
            book.set_identifier(f'book_{timestamp}')
            book.set_title('The Gates of Morning')
            chapter = epub.EpubHtml(title='Chapter 1', file_name='chap_1.xhtml', lang='en')
            chapter.content = content.encode('utf-8')
            book.add_item(chapter)
            book.add_item(epub.EpubNcx())
            book.add_item(epub.EpubNav())
            epub_path = os.path.join(CONTENT_DIR, f'book_{timestamp}.epub')
            epub.write_book(book, epub_path)
            version_id = f"chapter_1_v1_{timestamp}_epub"
            metadata = {
                "chapter": "1",
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "format": "epub",
                "accessibility_data": review_states[session_id].get('accessibility_data', {})
            }
            save_version(content, version_id, metadata)
            return send_file(epub_path)
        elif format == 'markdown':
            md_path = os.path.join(CONTENT_DIR, f'book_{timestamp}.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# The Gates of Morning\n\n## Chapter 1\n\n{content}")
            version_id = f"chapter_1_v1_{timestamp}_markdown"
            metadata = {
                "chapter": "1",
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "format": "markdown",
                "accessibility_data": review_states[session_id].get('accessibility_data', {})
            }
            save_version(content, version_id, metadata)
            return send_file(md_path)
        else:
            logger.error(f"Unsupported format: {format}")
            return jsonify({'status': 'error', 'message': 'Unsupported format'}), 400
    except Exception as e:
        logger.error(f"Error in export_content: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/analyze/<session_id>', methods=['GET'])
def analyze(session_id):
    logger.debug(f"Request: {request.url}")
    if session_id not in review_states:
        logger.error(f"Invalid session ID: {session_id}")
        return jsonify({'status': 'error', 'message': 'Invalid session ID'}), 400
    content = review_states[session_id].get('final_content', review_states[session_id]['raw_content'])
    try:
        analysis = analyze_content(content)
        version_id = review_states[session_id].get('version_id', f"chapter_1_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        metadata = {
            "chapter": "1",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "analysis": json.dumps(analysis),  # Serialize analysis to string
            "accessibility_data": review_states[session_id].get('accessibility_data', {})
        }
        save_version(content, version_id, metadata)
        return jsonify({'status': 'success', 'analysis': analysis})
    except Exception as e:
        logger.error(f"Error in analyze: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    logger.debug(f"Request: {request.url}")
    try:
        data = request.json
        query = data.get('query', '')
        doc_id = data.get('doc_id')
        metadata = data.get('metadata', {"chapter": "1"})
        rl = RLSearch(collection)
        results = rl.search(query, doc_id, metadata)
        return jsonify({'status': 'success', 'results': results})
    except Exception as e:
        logger.error(f"Error in /api/search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/screenshots/<filename>')
def serve_screenshot(filename):
    logger.debug(f"Request: {request.url}")
    try:
        return send_file(os.path.join(SCREENSHOT_DIR, filename))
    except FileNotFoundError:
        logger.error(f"Screenshot not found: {filename}")
        return jsonify({'status': 'error', 'message': 'Screenshot not found'}), 404

# Fallback route for unknown endpoints
@app.route('/<path:path>')
def catch_all(path):
    logger.error(f"Unknown route: {path}")
    return jsonify({'status': 'error', 'message': f'Route /{path} not found'}), 404

# --- Main ---
if __name__ == "__main__":
    logger.info("Starting Flask server on port 5000")
    app.run(debug=True, port=5000)