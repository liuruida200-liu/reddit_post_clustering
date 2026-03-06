import sys
import re
import time
import threading
import hashlib
from datetime import datetime
from io import BytesIO
import warnings

import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import mysql.connector

warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'sub_reddit'
}

SUBREDDITS = [
    'technology', 'tech', 'gadgets', 'programming', 'computerscience',
    'artificial', 'MachineLearning', 'cybersecurity', 'netsec', 'hardware',
    'software', 'linux', 'apple', 'android', 'gaming',
]

REQUEST_SIZE = 3000

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
    'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
    'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    'if', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'once', 'can', 'get', 'got', 'like', 'even',
    'new', 'want', 'use', 'used', 'using', 'make', 'made', 'know', 'take', 'see', 'come',
    'think', 'look', 'going', 'way', 'well', 'back', 'much', 'because', 'good', 'give',
    'dont', 'im', 'ive', 'youre', 'thats', 'theyre', 'weve', 'hes', 'shes', 'lets'
}


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.close()
        conn.close()
        self.conn = mysql.connector.connect(**DB_CONFIG)

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                post_id VARCHAR(255) UNIQUE,
                subreddit VARCHAR(100),
                title TEXT,
                content TEXT,
                author_hash VARCHAR(64),
                timestamp DATETIME,
                url TEXT,
                likes VARCHAR(20),
                comments INT,
                image_text TEXT,
                keywords TEXT,
                topics TEXT,
                embedding BLOB NULL,
                cluster_id INT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        try:
            cursor.execute('ALTER TABLE posts ADD COLUMN subreddit VARCHAR(100) AFTER post_id')
        except:
            pass
        self.conn.commit()

    def insert_post(self, post_data):
        cursor = self.conn.cursor()
        embedding_bytes = None
        cluster_id = None

        cursor.execute('''
            INSERT INTO posts (post_id, subreddit, title, content, author_hash, timestamp, url, likes, comments,
                             image_text, keywords, topics, embedding, cluster_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                subreddit=VALUES(subreddit), title=VALUES(title), content=VALUES(content), likes=VALUES(likes),
                comments=VALUES(comments), keywords=VALUES(keywords), topics=VALUES(topics)
        ''', (
            post_data['post_id'], post_data.get('subreddit', ''), post_data['title'], post_data['content'],
            post_data['author_hash'], post_data['timestamp'], post_data['url'],
            post_data['likes'], post_data['comments'], post_data['image_text'],
            post_data['keywords'], post_data['topics'], embedding_bytes, cluster_id
        ))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()


class RedditScraper:
    BATCH_SIZE = 100
    REQUEST_TIMEOUT = 30
    DELAY_BETWEEN_REQUESTS = 3
    DELAY_BETWEEN_SUBREDDITS = 5

    def __init__(self, subreddits=None):
        self.subreddits = subreddits or SUBREDDITS
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        })

    def scrape_posts(self, num_posts):
        print(f"Fetching {num_posts} posts from {len(self.subreddits)} subreddits...")
        posts_per_subreddit = max(1, num_posts // len(self.subreddits))
        all_posts = []

        for i, subreddit in enumerate(self.subreddits):
            if len(all_posts) >= num_posts:
                break

            print(f"\n  Scraping r/{subreddit}...")
            posts = self._scrape_subreddit_json(subreddit, posts_per_subreddit)

            for post in posts:
                post['subreddit'] = subreddit

            all_posts.extend(posts)
            print(f"  Got {len(posts)} posts from r/{subreddit} (total so far: {len(all_posts)})")
            time.sleep(self.DELAY_BETWEEN_SUBREDDITS)

        return all_posts[:num_posts]

    def _scrape_subreddit_json(self, subreddit, num_posts):
        posts = []
        url = f'https://old.reddit.com/r/{subreddit}.json'
        after = None

        while len(posts) < num_posts:
            params = {'limit': 100}
            if after:
                params['after'] = after

            try:
                resp = self.session.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    print(f"  HTTP {resp.status_code} for r/{subreddit}, stopping.")
                    break

                data = resp.json()
                children = data.get('data', {}).get('children', [])
                if not children:
                    break

                for child in children:
                    post_data = child.get('data', {})
                    if post_data.get('stickied'):
                        continue
                    posts.append({
                        'post_id': post_data.get('name', ''),
                        'title': post_data.get('title', ''),
                        'content': post_data.get('selftext', ''),
                        'author': post_data.get('author', '[deleted]'),
                        'timestamp': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat() if post_data.get('created_utc') else None,
                        'url': post_data.get('url', ''),
                        'likes': str(post_data.get('score', 0)),
                        'comments': post_data.get('num_comments', 0),
                        'image_url': post_data.get('thumbnail') if post_data.get('thumbnail', '').startswith('http') else None
                    })

                after = data.get('data', {}).get('after')
                if not after:
                    break  # No more pages available

                time.sleep(self.DELAY_BETWEEN_REQUESTS)

            except Exception as e:
                print(f"  Error scraping r/{subreddit}: {e}")
                break

        return posts[:num_posts]


class DataPreprocessor:
    def preprocess(self, post):
        processed = post.copy()
        processed['title'] = self._clean_text(post.get('title', ''))
        processed['content'] = self._clean_text(post.get('content', ''))
        processed['author_hash'] = self._hash_username(post.get('author', ''))
        processed['timestamp'] = self._parse_timestamp(post.get('timestamp'))
        processed['image_text'] = self._extract_image_text(post.get('image_url'))

        full_text = f"{processed['title']} {processed['content']} {processed['image_text']}"
        processed['keywords'] = ','.join(self._extract_keywords(full_text))
        processed['topics'] = ','.join(self._extract_topics(full_text))

        return processed

    def _clean_text(self, text):
        if not text: return ''
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _hash_username(self, username):
        if not username: return hashlib.sha256(b'anonymous').hexdigest()
        return hashlib.sha256(username.encode()).hexdigest()

    def _parse_timestamp(self, ts):
        if not ts: return datetime.now()
        if isinstance(ts, datetime): return ts
        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z']:
            try:
                return datetime.strptime(ts.replace('+00:00', '+0000'), fmt)
            except ValueError:
                continue
        return datetime.now()

    def _extract_image_text(self, image_url):
        if not image_url or not OCR_AVAILABLE: return ''
        try:
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content))
            return self._clean_text(pytesseract.image_to_string(img))
        except Exception:
            return ''

    def _extract_keywords(self, text, top_n=10):
        if not text: return []
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in STOP_WORDS]
        freq = {}
        for w in words: freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    def _extract_topics(self, text):
        tech_topics = {
            'ai': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'gpt'],
            'security': ['security', 'hack', 'breach', 'cyber', 'privacy', 'vulnerability'],
            'hardware': ['cpu', 'gpu', 'chip', 'processor', 'hardware', 'computer'],
            'software': ['software', 'program', 'code', 'developer', 'programming']
        }
        text_lower = text.lower()
        found = [topic for topic, kws in tech_topics.items() if any(kw in text_lower for kw in kws)]
        return found if found else ['general']


class RedditPipeline:
    def __init__(self):
        self.db = DatabaseManager()
        self.scraper = RedditScraper()
        self.preprocessor = DataPreprocessor()
        self.running = False

    def run_pipeline(self, num_posts=3000):
        print("=" * 50)
        print(f"[{datetime.now()}] Starting data collection...")

        raw_posts = self.scraper.scrape_posts(num_posts)
        if not raw_posts:
            print("No posts fetched. Exiting.")
            return

        print(f"\nPreprocessing and storing {len(raw_posts)} posts...")
        for i, raw_post in enumerate(raw_posts):
            processed = self.preprocessor.preprocess(raw_post)
            self.db.insert_post(processed)
            if (i + 1) % 100 == 0:
                print(f"  Stored {i + 1}/{len(raw_posts)} posts...")

        print(f"[{datetime.now()}] Successfully stored {len(raw_posts)} posts.")
        print("=" * 50)

    def start_automation(self, interval_minutes):
        self.running = True
        def update_loop():
            while self.running:
                self.run_pipeline(REQUEST_SIZE)
                for _ in range(interval_minutes * 60):
                    if not self.running: break
                    time.sleep(1)

        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
        print(f"Automation running. Press Ctrl+C to stop. Updates every {interval_minutes} mins.")

    def close(self):
        self.running = False
        self.db.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scraper.py <interval_minutes>")
        print("Example: python scraper.py 5")
        sys.exit(1)

    interval = int(sys.argv[1])
    pipeline = RedditPipeline()

    try:
        pipeline.start_automation(interval)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()