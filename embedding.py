from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load another Transformer model
model = SentenceTransformer("all-mpnet-base-v2")

# Many sentences
sentences = [
    "Artificial intelligence is changing the world.",
    "I like creating AI projects.",
    "The weather is very hot today.",
    "It is raining heavily outside.",
    "I enjoy learning new technologies.",
    "I want to become an AI engineer.",
    "Data science helps us understand data.",
    "Deep learning is a part of artificial intelligence."
]

# Convert sentences into embeddings
embeddings = model.encode(sentences)

# Display results
print("Total number of sentences:", len(sentences))
print("Embedding dimension:", len(embeddings[0]))

print("\n--- Embeddings ---")

for i, sentence in enumerate(sentences):
    print("\nSentence:", sentence)
    print("Embedding:", embeddings[i])

# Calculate cosine similarity
similarity = cosine_similarity(embeddings)

print("\n--- Similarity between sentences ---")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        if similarity[i][j] > 0.7:
            print(
                f"\n'{sentences[i]}'"
                f"\n'{sentences[j]}'"
                f"\nSimilarity: {similarity[i][j]:.4f}"
            )