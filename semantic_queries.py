import pickle
from os import path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

DATA = "data.txt"
EMBEDDINGS = "embeddings.pkl"


def read_doc():
    with open(DATA, "r") as f:
        return [normalise(line) for line in f]


def normalise(txt):
    return " ".join(txt.strip().lower().split())


def load_corpus(model):
    sentences = read_doc()

    if path.isfile(EMBEDDINGS):
        embeddings = load_embeddings()
    else:
        embeddings = model.encode(sentences)
        save_embeddings(embeddings)

    return sentences, embeddings


def load_embeddings():
    with open(EMBEDDINGS, "rb") as f:
        embeddings = pickle.load(f)
        print("Loaded embeddings")
        return embeddings


def save_embeddings(embeddings):
    with open(EMBEDDINGS, "wb") as f:
        pickle.dump(embeddings, f)
        print("Saved embeddings")


def load_model():
    pretrained_model_name = "all-MiniLM-L6-v2"  # "msmarco-MiniLM-L6-cos-v5"
    return SentenceTransformer(pretrained_model_name)


def repl(model, sentences, corpus_embeddings):
    while True:
        query = normalise(input("Query ? "))
        if not query:
            continue
        query_em = model.encode(query, convert_to_tensor=True)
        hits = semantic_search(query_em, corpus_embeddings, top_k=3)[0]
        for h in hits:
            print(sentences[h["corpus_id"]])
        print("=" * 80)


def main():
    model = load_model()
    sentences, corpus_embeddings = load_corpus(model)
    repl(model, sentences, corpus_embeddings)


if __name__ == "__main__":
    main()
