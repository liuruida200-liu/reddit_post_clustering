from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import mysql.connector
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import re

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '520529',
    'database': 'sub_reddit'
}

NUM_CLUSTERS = 5

def fetch_data():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title, content, subreddit FROM posts WHERE title != '' AND title IS NOT NULL")
    posts = cursor.fetchall()
    conn.close()
    return posts


def preprocess_doc2vec(posts):
    tagged_data = []
    for post in posts:
        text = post['title']
        if post['content']:
            text += " " + post['content']
        if not text:
            tokens = []
        tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
        if len(tokens) > 0:
            tagged_data.append(TaggedDocument(words=tokens, tags=[post['id']]))
    return tagged_data


def train_and_evaluate(tagged_data, posts, config_name, vector_size, min_count, epochs):
    print("=" * 60)
    print(f"Running {config_name}")
    print(f"Parameters: vector_size={vector_size}, min_count={min_count}, epochs={epochs}")
    print("-" * 60)
    model = Doc2Vec(vector_size=vector_size, window=5, min_count=min_count, workers=4, epochs=epochs)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    vectors = [model.dv[doc.tags[0]] for doc in tagged_data]
    vectors = np.array(vectors)
    normalized_vectors = normalize(vectors)
    
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_vectors)
    id_to_post = {post['id']: post for post in posts}
    for cluster_id in range(NUM_CLUSTERS):
        print(f"\n[ CLUSTER {cluster_id} ]")
        cluster_doc_ids = [tagged_data[i].tags[0] for i, label in enumerate(cluster_labels) if label == cluster_id]
        sample_ids = cluster_doc_ids[:5]
        for db_id in sample_ids:
            post = id_to_post[db_id]
            subreddit = post.get('subreddit', 'unknown')
            title = post.get('title', '')
            print(f"  -> [r/{subreddit}] {title}...")

    print("\n")
    return model, cluster_labels



if __name__ == '__main__':
    posts = fetch_data()  
    tagged_data = preprocess_doc2vec(posts)
    configs = [
        {"name": "Config 1", "size": 50, "min_count": 2, "epochs": 30},
        {"name": "Config 2", "size": 100, "min_count": 2, "epochs": 30},
        {"name": "Config 3", "size": 100, "min_count": 2, "epochs": 60}
    ]
    
    for cfg in configs:
        train_and_evaluate(
            tagged_data, 
            posts,
            config_name=cfg["name"], 
            vector_size=cfg["size"], 
            min_count=cfg["min_count"], 
            epochs=cfg["epochs"]
        )