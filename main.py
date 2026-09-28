import os
import math
import fitz
import pickle

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# 2. CHUNKING
# ============================================================

def chunk_text(text, chunk_size=1500):

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) <= chunk_size:

            current_chunk += paragraph + "\n\n"

        else:

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            current_chunk = paragraph + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# 3. CREATE EMBEDDING
# ============================================================

def get_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# ============================================================
# 4. COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(y * y for y in b)
    )

    return dot_product / (
        magnitude_a * magnitude_b
    )


# ============================================================
# 5. READ PDF
# ============================================================

pdf_path = "documents/preeclampsia.pdf"

document = fitz.open(pdf_path)

full_text = ""

for page in document:

    full_text += page.get_text() + "\n"


# ============================================================
# 6. CREATE CHUNKS
# ============================================================

chunks = chunk_text(full_text)

print("Number of chunks:", len(chunks))


# ============================================================
# 7. LOAD OR CREATE EMBEDDINGS
# ============================================================

if os.path.exists("embeddings.pkl"):

    print("Loading saved embeddings...")

    with open("embeddings.pkl", "rb") as file:

        data = pickle.load(file)

    chunks = data["chunks"]
    embeddings = data["embeddings"]

    print("Loaded embeddings:", len(embeddings))


    # --------------------------------------------------------
    # Continue if some embeddings are missing
    # --------------------------------------------------------

    if len(embeddings) < len(chunks):

        print("Some embeddings are missing.")

        print(
            "Continuing from chunk",
            len(embeddings) + 1
        )

        for i in range(
            len(embeddings),
            len(chunks)
        ):

            embedding = get_embedding(
                chunks[i]
            )

            embeddings.append(
                embedding
            )

            print(
                f"Embedded chunk "
                f"{i + 1}/{len(chunks)}"
            )


            # Save progress after every chunk

            with open(
                "embeddings.pkl",
                "wb"
            ) as file:

                pickle.dump(
                    {
                        "chunks": chunks,
                        "embeddings": embeddings
                    },
                    file
                )


        print(
            "All embeddings saved successfully."
        )


else:

    print(
        "No saved embeddings found."
    )

    print(
        "Generating embeddings..."
    )

    embeddings = []


    for i, chunk in enumerate(chunks):

        embedding = get_embedding(
            chunk
        )

        embeddings.append(
            embedding
        )

        print(
            f"Embedded chunk "
            f"{i + 1}/{len(chunks)}"
        )


        # Save progress after every chunk

        with open(
            "embeddings.pkl",
            "wb"
        ) as file:

            pickle.dump(
                {
                    "chunks": chunks,
                    "embeddings": embeddings
                },
                file
            )


    print(
        "All embeddings saved successfully."
    )


# ============================================================
# 8. USER QUESTION
# ============================================================

question = (
    "What biomarkers are used for "
    "first-trimester preeclampsia prediction?"
)


# ============================================================
# 9. EMBED THE QUESTION
# ============================================================

question_embedding = get_embedding(
    question
)


# ============================================================
# 10. COMPARE QUESTION WITH DOCUMENT CHUNKS
# ============================================================

results = []

for i, embedding in enumerate(
    embeddings
):

    similarity = cosine_similarity(
        question_embedding,
        embedding
    )

    results.append(
        (
            similarity,
            i
        )
    )


# ============================================================
# 11. SORT BY SIMILARITY
# ============================================================

results.sort(
    reverse=True
)


# ============================================================
# 12. SHOW TOP 3 RETRIEVED CHUNKS
# ============================================================

top_chunks = []

print()
print("==============================")
print("TOP RETRIEVED CHUNKS")
print("==============================")


for rank, (similarity, index) in enumerate(
    results[:3],
    start=1
):

    print()
    print(f"Rank: {rank}")
    print(f"Chunk: {index}")
    print(f"Similarity: {similarity:.4f}")
    print("------------------------------")
    print(chunks[index])

    top_chunks.append(
        chunks[index]
    )


# ============================================================
# 13. COMBINE RETRIEVED CONTEXT
# ============================================================

context = "\n\n---\n\n".join(
    top_chunks
)


# ============================================================
# 14. SEND RETRIEVED CONTEXT TO GEMINI
# ============================================================

prompt = f"""
You are a medical research assistant.

Answer the user's question using ONLY
the provided research paper excerpts.

If the excerpts do not contain enough
information to answer the question,
say that the provided excerpts do not
contain enough information.

Research paper excerpts:

{context}

User question:

{question}
"""


response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)


# ============================================================
# 15. DISPLAY ANSWER
# ============================================================

print()
print("==============================")
print("RAG ANSWER")
print("==============================")

print(response.text)