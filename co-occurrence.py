# ==========================================
# Import Dependencies
# ==========================================
import html
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchtext
from numpy import array, diag, dot
from scipy.sparse.linalg import svds
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.metrics.pairwise import cosine_similarity
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator
from tqdm import tqdm

TRAIN = False

if TRAIN:
    # ==========================================
    # Read data
    # ==========================================
    df = pd.read_json("./datasubsets/subdata_svd.json", lines=True)
    corpus = df["reviewText"].to_numpy()

    # ==========================================
    # Data Preprocessing
    # ==========================================
    def clean_text(text):
        import re

        text = re.sub(r"([a-zA-Z]+)n[\'’]t", r"\1 not", text)
        text = re.sub(r"([iI])[\'’]m", r"\1 am", text)
        text = re.sub(r"([iI])[\'’]ll", r"\1 will", text)
        text = re.sub(r"[^a-zA-Z0-9\:\$\-\,\%\.\?\!]+", " ", text)
        text = html.unescape(text)
        # text = re.sub(r"([a-zA-Z]+)[\'’]s", r"\1 is", text)

        text = re.sub(r"_(.*?)_", r"\1", text)
        return text

    tokenizer = get_tokenizer("basic_english")

    tokens = []

    for sent in corpus:
        tokens.append(tokenizer(clean_text(sent)))

    # ==========================================
    # Build Vocabulary
    # ==========================================

    MIN_WORD_FREQUENCY = 10

    vocab = build_vocab_from_iterator(tokens, min_freq=MIN_WORD_FREQUENCY, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    print(f"Total sentences in text: {len(tokens)}")
    print(f"Unique words: {len(vocab)}")

    # ==========================================
    # Build Co-occurence Matrix
    # ==========================================

    WINDOW_LENGTH = 4
    co_occ_matrix = np.zeros((len(vocab), len(vocab)))

    def build_co_occ(sentence):
        sent = tokenizer(sentence)
        for idx, word in enumerate(sent):
            for context_id in range((max(0, idx - WINDOW_LENGTH)), (min(len(sent), idx + WINDOW_LENGTH + 1))):
                row = vocab[word]
                col = vocab[sent[context_id]]
                co_occ_matrix[row][col] += 1

    for sentence in corpus:
        build_co_occ(sentence)

    # ==========================================
    # Truncated SVDs
    # ==========================================

    df = pd.DataFrame(co_occ_matrix, index=vocab.get_stoi().keys(), columns=vocab.get_stoi().keys())
    co_occ_np = df.to_numpy()
    U, s, VT = svds(co_occ_np, k=300)

    # ==========================================
    # Incremental PCA
    # ==========================================
    # incr_pca = IncrementalPCA(n_components=300, batch_size=1000)
    # batches = np.array_split(np.nan_to_num(co_occ_matrix), int(len(co_occ_matrix) / 1000))

    # for batch in tqdm(batches):
    #     incr_pca.partial_fit(batch)

    # U = incr_pca.components_.T

    # ==========================================
    # Saving Matrix and Vocabulary
    # ==========================================
    np.save("ipca_matrix.npy", U)

    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

else:
    # ==========================================
    # Load Matrix and Vocabulary
    # ==========================================

    U = np.load("./svd_model/ipca_matrix.npy", allow_pickle=True)

    with open("./svd_model/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # ==========================================
    # Find Top 10 Similar Words
    # ==========================================
    def getTopSimilarWords(word):
        word_index = vocab[word]
        word_vector = U[word_index]

        similarities = []
        for i in range(U.shape[0]):
            simi = cosine_similarity(U[i].reshape(1, -1), [word_vector])
            similarities.extend(simi.flatten())

        k = 10
        top_k_similar_words = np.argsort(similarities)[-k:]

        index_to_words = vocab.get_itos()

        return list(reversed([index_to_words[idx] for idx in top_k_similar_words])), list(
            reversed([similarities[idx] for idx in top_k_similar_words])
        )

    # ==========================================
    # Predict words
    # ==========================================

    U_df = pd.DataFrame(U, index=vocab.get_stoi())
    model = {}
    c = 0
    for i in list(U_df.index):
        model[i] = list(U_df.loc[i])
        c = c + 1

    def display_pca_scatterplot(model, c, words=None, sample=0):
        word_vectors = np.array([model[w] for w in words])
        twodim = PCA().fit_transform(word_vectors)[:, :2]
        plt.figure(figsize=(5, 5))
        plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors="k", c=c)
        for word, (x, y) in zip(words, twodim):
            plt.text(x, y, word)
        plt.savefig(f"{similar_words[0]}.png")
        plt.show()

    # ==========================================
    # Predict words
    # ==========================================
    # while True:
    # word = input("Enter a word to get similar ones\n")
    word_list = ["titanic", "book", "definitely", "drink", "beautiful", "worth"]
    color_list = ["r", "b", "g", "black", "yellow", "pink"]
    for i, word in enumerate(word_list):
        similar_words, cosine_dist = getTopSimilarWords(word)
        display_pca_scatterplot(model, color_list[i], similar_words)
        print(f"{word}: {similar_words}")
