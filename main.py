import asyncio
import os
import logging
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
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(CHROMADB_DIR, exist_ok=True)

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
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            content = await page.locator("div.mw-parser-output").inner_text()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(SCREENSHOT_DIR, f"chapter_1_{timestamp}.png")
            await page.screenshot(path=screenshot_path)
            await browser.close()
            content_path = os.path.join(CONTENT_DIR, f"raw_chapter_1_{timestamp}.txt")
            with open(content_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Scraped content saved to {content_path}, screenshot to {screenshot_path}")
            return content, screenshot_path, content_path
    except Exception as e:
        logger.error(f"Error in scrape_content_and_screenshot: {e}")
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
def review_content(content):
    logger.debug("Starting content review")
    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Review and improve the following text for clarity, grammar, and coherence:\n{content}"
    )
    prompt = prompt_template.format(content=content)
    try:
        result = llm.invoke(prompt)
        logger.info("Content review completed")
        return result
    except Exception as e:
        logger.error(f"Error in review_content: {e}")
        return content

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
        collection.add(documents=[content], metadatas=[metadata], ids=[version_id])
        logger.info(f"Version {version_id} saved")
    except Exception as e:
        logger.error(f"Error saving version: {e}")
        raise

# --- Flask API Endpoints ---
@app.route('/api/scrape', methods=['GET'])
async def scrape():
    logger.debug("Received /api/scrape request")
    try:
        content, screenshot_path, content_path = await scrape_content_and_screenshot(URL)
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        review_states[session_id] = {'raw_content': content}
        return jsonify({
            'status': 'success',
            'content': content[:500],
            'screenshot_filename': os.path.basename(screenshot_path),
            'content_path': content_path,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error in /api/scrape: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/process/<session_id>/<step>', methods=['POST'])
def process_content(session_id, step):
    logger.debug(f"Received /api/process/{session_id}/{step} request")
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
            return jsonify({'status': 'success', 'content': spun, 'role': 'writer'})
        elif step == 'review':
            if 'spun_content' not in state:
                logger.error("Spin step not completed")
                return jsonify({'status': 'error', 'message': 'Spin step not completed'}), 400
            reviewed = review_content(state['spun_content'] if not content else content)
            state['reviewed_content'] = reviewed
            return jsonify({'status': 'success', 'content': reviewed, 'role': 'reviewer'})
        elif step == 'edit':
            if 'reviewed_content' not in state:
                logger.error("Review step not completed")
                return jsonify({'status': 'error', 'message': 'Review step not completed'}), 400
            state['final_content'] = content if content else state['reviewed_content']
            version_id = f"chapter_1_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metadata = {"chapter": "1", "version": "1.0", "timestamp": datetime.now().isoformat()}
            save_version(state['final_content'], version_id, metadata)
            del review_states[session_id]
            return jsonify({'status': 'success', 'content': state['final_content'], 'version_id': version_id})
        else:
            logger.error(f"Invalid step: {step}")
            return jsonify({'status': 'error', 'message': 'Invalid step'}), 400
    except Exception as e:
        logger.error(f"Error in /api/process: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    logger.debug("Received /api/search request")
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
    logger.debug(f"Received /screenshots/{filename} request")
    try:
        return send_file(os.path.join(SCREENSHOT_DIR, filename))
    except FileNotFoundError:
        logger.error(f"Screenshot not found: {filename}")
        return jsonify({'status': 'error', 'message': 'Screenshot not found'}), 404

# --- Main ---
if __name__ == "__main__":
    logger.info("Starting Flask server")
    app.run(debug=True, port=5000)