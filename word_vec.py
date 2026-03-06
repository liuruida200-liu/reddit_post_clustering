import mysql.connector
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import re
from collections import Counter

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '520529',
    'database': 'sub_reddit'
}

NUM_DOC_CLUSTERS = 5
NUM_WORD_BINS = 100

def fetch_data():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title, content, subreddit FROM posts WHERE title != '' AND title IS NOT NULL")
    posts = cursor.fetchall()
    conn.close()
    return posts



def prepare_sentences(posts):
    sentences = []
    for post in posts:
        text = post['title'] + " " + (post['content'] or "")
        if not text:
            tokens = []
        tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
        if tokens:
            sentences.append(tokens)
    return sentences


def main():
    posts = fetch_data()
    sentences = prepare_sentences(posts)
    w2v_model = Word2Vec(sentences, vector_size=50, window=5, min_count=2, workers=4, epochs=30)
    words = w2v_model.wv.index_to_key
    word_vectors = np.array([w2v_model.wv[word] for word in words])
    word_kmeans = KMeans(n_clusters=NUM_WORD_BINS, random_state=42, n_init=10)
    word_bin_labels = word_kmeans.fit_predict(word_vectors)
    word_to_bin = {word: bin_id for word, bin_id in zip(words, word_bin_labels)}
    sample_bin = 5
    words_in_sample_bin = [w for w, b in word_to_bin.items() if b == sample_bin][:10]
    print(f"   -> Example words in Bin {sample_bin}: {words_in_sample_bin}")
    document_vectors = []
    valid_posts = []

    for i, post in enumerate(posts):
        tokens = sentences[i]
        doc_vec = np.zeros(NUM_WORD_BINS)
        valid_word_count = 0
        for word in tokens:
            if word in word_to_bin:
                bin_id = word_to_bin[word]
                doc_vec[bin_id] += 1
                valid_word_count += 1
        if valid_word_count > 0:
            doc_vec = doc_vec / valid_word_count
            document_vectors.append(doc_vec)
            valid_posts.append(post)
            
    document_vectors = np.array(document_vectors)
    print("\n4. Clustering")
    norm_doc_vectors = normalize(document_vectors)
    doc_kmeans = KMeans(n_clusters=NUM_DOC_CLUSTERS, random_state=42, n_init=10)
    doc_labels = doc_kmeans.fit_predict(norm_doc_vectors)

    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    
    for cluster_id in range(NUM_DOC_CLUSTERS):
        print(f"\n[ CLUSTER {cluster_id} ]")
        cluster_indices = [i for i, label in enumerate(doc_labels) if label == cluster_id]
        for idx in cluster_indices[:5]:
            post = valid_posts[idx]
            subreddit = post.get('subreddit', 'unknown')
            title = post.get('title', '')[:80]
            print(f"  -> [r/{subreddit}] {title}...")

if __name__ == '__main__':
    main()