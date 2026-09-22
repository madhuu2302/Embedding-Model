# Sentence Embedding Using Transformer

## Project Description

This project demonstrates how to convert multiple sentences into numerical vector representations called **embeddings** using a Transformer-based Sentence Embedding model.

The project uses the **`all-mpnet-base-v2`** model from Sentence Transformers. It generates a **768-dimensional embedding** for each sentence and uses **cosine similarity** to identify sentences with similar meanings.

## Technologies Used

* Python
* Sentence Transformers
* Scikit-learn
* Transformer Model
* VS Code

## Transformer Model

**Model:** `all-mpnet-base-v2`

**Embedding Dimension:** 768

The model is used for sentence similarity and semantic representation.

## Project Workflow

```text
Multiple Sentences
        ↓
Sentence Transformer
        ↓
Tokenization
        ↓
Contextual Representation
        ↓
Pooling
        ↓
768-Dimensional Embeddings
        ↓
Cosine Similarity
        ↓
Similar Sentences
```

## Example Sentences

The project uses multiple sentences such as:

* I enjoy coding in Python.
* I love programming in Python.
* Python is my favorite programming language.
* I am learning data science.
* Machine learning is very interesting.
* Artificial intelligence is changing the world.
* I like creating AI projects.
* The weather is very hot today.
* It is raining heavily outside.
* I want to become an AI engineer.

## Installation

Open the VS Code terminal and run:

```bash
pip install sentence-transformers scikit-learn
```

## How to Run

Clone or open the project in VS Code.

Run the Python file:

```bash
python embedding.py
```

If `python` does not work on Windows, use:

```bash
py embedding.py
```

## How It Works

The program loads the Transformer model:

```python
model = SentenceTransformer("all-mpnet-base-v2")
```

The sentences are converted into embeddings:

```python
embeddings = model.encode(sentences)
```

Cosine similarity is then used to compare the embeddings:

```python
similarity = cosine_similarity(embeddings)
```

Sentences with similar meanings generally have higher similarity values.

## Expected Output

```text
Total number of sentences: 15
Embedding dimension: 768

--- Embeddings ---

Sentence: I enjoy coding in Python.
Embedding: [ ... ]

Sentence: I love programming in Python.
Embedding: [ ... ]

--- Similarity between sentences ---

Similarity: 0.8...
```
<img width="691" height="872" alt="image" src="https://github.com/user-attachments/assets/bef50a2b-9c39-4c23-a67f-f88feea5bb31" />

The exact embedding values and similarity scores may vary depending on the model and execution environment.

## Applications

Sentence embeddings can be used for:

* Semantic Search
* Sentence Similarity
* Document Similarity
* Question Matching
* Recommendation Systems
* Clustering
* Retrieval-Augmented Generation (RAG)

## Conclusion

This project demonstrates how a Transformer-based Sentence Embedding model can convert multiple sentences into numerical vectors. These vectors can then be compared using cosine similarity to identify sentences that have similar meanings.

AUTHOR

Madhumitha 
