import asyncio
import json
import os
from playwright.async_api import async_playwright
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
import numpy as np
import random
from datetime import datetime
from dotenv import load_dotenv  # ✅ for loading secrets from .env

# Load environment variables
load_dotenv()

# Configuration
URL = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
SCREENSHOT_DIR = "screenshots"
CONTENT_DIR = "content"
CHROMADB_DIR = "./chroma_db"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(CHROMADB_DIR, exist_ok=True)

# ✅ Use token securely
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=0.7,
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token
)

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(persist_directory=CHROMADB_DIR, anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection(name="book_versions")

# --- Web Scraping ---
async def scrape_content_and_screenshot(url):
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

        return content, screenshot_path, content_path

# --- Spin Content ---
def spin_content(content):
    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Rewrite the following text in a different style while preserving the meaning:\n{content}"
    )
    prompt = prompt_template.format(content=content[:500])
    try:
        return llm.invoke(prompt)
    except Exception as e:
        print(f"[ERROR in spin_content] {e}")
        return content

# --- Review Content ---
def review_content(content):
    prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Review and improve the following text for clarity, grammar, and coherence:\n{content}"
    )
    prompt = prompt_template.format(content=content)
    try:
        return llm.invoke(prompt)
    except Exception as e:
        print(f"[ERROR in review_content] {e}")
        return content

# --- Human Review ---
def human_review(content, role="writer"):
    print(f"\n--- {role.capitalize()} Review ---")
    print(content[:500] + "..." if len(content) > 500 else content)
    feedback = input(f"Enter feedback for {role} (or 'approve' to accept): ")
    return (content, True) if feedback.lower() == "approve" else (feedback, False)

# --- Agentic Workflow ---
def agentic_workflow(raw_content):
    spun = spin_content(raw_content)
    spun, ok = human_review(spun, "writer")
    if not ok: spun = spun

    reviewed = review_content(spun)
    reviewed, ok = human_review(reviewed, "reviewer")
    if not ok: reviewed = reviewed

    edited, ok = human_review(reviewed, "editor")
    if not ok: edited = edited

    return edited

# --- RL Search ---
class RLSearch:
    def __init__(self, collection):
        self.collection = collection
        self.actions = ["query_by_id", "query_by_content", "query_by_metadata"]
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def get_state(self, query): return len(query)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        self.q_table.setdefault(state, {a: 0 for a in self.actions})
        return max(self.q_table[state], key=self.q_table[state].get)

    def query_collection(self, action, query, doc_id=None, metadata=None):
        try:
            if action == "query_by_id" and doc_id:
                return self.collection.get(ids=[doc_id])
            elif action == "query_by_content":
                return self.collection.query(query_texts=[query], n_results=1)
            elif action == "query_by_metadata" and metadata:
                return self.collection.get(where=metadata)
        except Exception as e:
            print(f"[ERROR in query_collection] {e}")
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
        return results

# --- Save Version ---
def save_version(content, version_id, metadata):
    collection.add(documents=[content], metadatas=[metadata], ids=[version_id])

# --- Main ---
async def main():
    print("Scraping content...")
    raw, shot_path, text_path = await scrape_content_and_screenshot(URL)
    print(f"Content saved to {text_path}, screenshot saved to {shot_path}")

    print("Starting agentic workflow...")
    final = agentic_workflow(raw)

    version_id = f"chapter_1_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata = {"chapter": "1", "version": "1.0", "timestamp": datetime.now().isoformat()}
    save_version(final, version_id, metadata)
    print(f"Version {version_id} saved to ChromaDB")

    rl = RLSearch(collection)
    results = rl.search("chapter 1 content", doc_id=version_id, metadata={"chapter": "1"})

    if results["documents"]:
        print(f"Retrieved: {results['documents'][0][:500]}...")
    else:
        print("No results found.")

if __name__ == "__main__":
    asyncio.run(main())
