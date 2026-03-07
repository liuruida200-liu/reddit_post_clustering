# README

This project implements document embeddings and clustering for Reddit posts using **Doc2Vec** and **Word2Vec + Bag-of-Words**.

## Files

scraper.py  
Scrapes Reddit posts from several subreddits and stores them in a MySQL database.

doc_vec.py  
Generates document embeddings using Doc2Vec with different configurations and clusters the posts.

word_vec.py  
Trains Word2Vec, clusters words into bins, converts documents into frequency vectors, and clusters the posts.

## Requirements

Install required packages:

pip install gensim scikit-learn numpy mysql-connector-python requests beautifulsoup4 pillow pytesseract

Make sure MySQL is running.

## Database

Database name: sub_reddit  
Table name: posts

Update the MySQL username and password in the scripts if needed.

## Run Instructions

1. Collect Reddit data:

python scraper.py 5

2. Run Doc2Vec embeddings and clustering:

python doc_vec.py

3. Run Word2Vec + Bag-of-Words embeddings and clustering:

python word_vec.py

## Notes

- The scripts load Reddit posts from the MySQL database.
- Make sure the database contains data before running the embedding scripts.
- OCR in scraper.py is optional and only works if pytesseract is installed.