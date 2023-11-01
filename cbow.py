# ==========================================
# Import Dependencies
# ==========================================
import html
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtext
import tqdm
from sklearn.decomposition import PCA
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer, to_map_style_dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator

TRAIN = False


# ==========================================
# Create Model
# Here I am declaring simple model with just **2 embedding layers**
# I tried various model, first one being just CBOW, then I added on it to use negative sampling

# 1. Embedding Layer with 300 dimensions as mentioned in the paper that it worked well after trying a lot of dimensions.
# 2. Linear layer that will give back the output as the whole vocabulary.

# _Here we are only interested in the embedding layer as those are the **featurized representation** of the words._
# ==========================================


class CBOW(nn.Module):
    def __init__(self, vocab, embedding_dim, max_norm):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab, embedding_dim, max_norm)
        self.linear = nn.Linear(embedding_dim, vocab)

    def forward(self, input):
        x = self.embeddings(input)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class CBOW_1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device):
        super(CBOW_1, self).__init__()
        self.device = device
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.neg_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input, focus_word, weight_mask, labels):
        temp_src_embedding = [
            self.context_embeddings(torch.tensor(src_word, dtype=torch.long).to(self.device)).sum(dim=0)
            for src_word in input
        ]
        src_embeds = torch.stack(temp_src_embedding)
        target_embeds = self.neg_embeddings(torch.tensor(focus_word, dtype=torch.long).to(self.device))
        weight_mask = torch.tensor(weight_mask, dtype=torch.float).to(self.device)
        labels = torch.tensor(labels, dtype=torch.float).to(self.device)

        pred = torch.bmm(src_embeds.unsqueeze(1), target_embeds.permute(0, 2, 1)).squeeze(1)

        loss = nn.functional.binary_cross_entropy_with_logits(
            pred.float(), labels, reduction="none", weight=weight_mask
        )
        loss = (loss.mean(dim=1) * weight_mask.shape[1] / weight_mask.sum(dim=1)).mean()

        return loss


if TRAIN:
    # ==========================================
    # Read data
    # ==========================================
    df = pd.read_json("./datasubsets/subdata_cbow.json", lines=True)
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
    ## Preparing Dataset
    # - Here I am taking the _window = 4_ i.e. 4 words before and 4 words after to grab the context, as authors mention 3-5 word window works best for the large dataset.
    # - I am also truncating the sequence to maximum of 256 length and creating input-output tensors for them.
    # ==========================================
    CBOW_WINDOW = 4
    SEQ_LEN = 256
    NEG_SAMPLE_SIZE = 4

    def collate_cbow(batch, text_pipeline, vocab, word_freq):
        batch_src_words, batch_trg_words, wmasks, labels = [], [], [], []
        for text in batch:
            text_tokens_ids = text_pipeline(text)
            if len(text_tokens_ids) < CBOW_WINDOW * 2 + 1:
                continue

            if SEQ_LEN:
                text_tokens_ids = text_tokens_ids[:SEQ_LEN]

            for idx in range(len(text_tokens_ids) - CBOW_WINDOW * 2):
                token_id_sequence = text_tokens_ids[idx : (idx + CBOW_WINDOW * 2 + 1)]

                # Taking out the focused target word
                output = token_id_sequence.pop(CBOW_WINDOW)

                # Rest of the context
                input_ = token_id_sequence

                neg_samples = []
                for j in range(NEG_SAMPLE_SIZE):
                    rnd_word = random.randint(0, len(vocab) - 1)
                    while rnd_word in input_:
                        rnd_word = random.randint(0, len(vocab) - 1)
                    neg_samples.append(rnd_word)

                batch_src_words += [input_]
                batch_trg_words += [[output] + neg_samples]
                labels += [[1] + [0] * len(neg_samples)]
                wmasks += [[1] * (len(neg_samples) + 1)]

        batch_src_words = torch.tensor(batch_src_words, dtype=torch.long)
        batch_trg_words = torch.tensor(batch_trg_words, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        wmasks = torch.tensor(wmasks, dtype=torch.long)

        return batch_src_words, batch_trg_words, wmasks, labels

    # Creating word frequency map
    word_freq = torch.zeros(len(vocab))
    for word in vocab.get_itos():
        word_freq[vocab[word]] = word_freq[vocab[word]] + 1

    # Normalize word frequencies to create a probability distribution
    word_freq = word_freq / word_freq.sum()

    # Preparing Dataset
    matched_style_corpus = to_map_style_dataset(corpus)
    text_pipeline = lambda x: vocab(tokenizer(x))
    train_dataloader = DataLoader(
        matched_style_corpus,
        batch_size=64,
        shuffle=True,
        collate_fn=partial(collate_cbow, text_pipeline=text_pipeline, vocab=vocab, word_freq=word_freq),
    )

    # ==========================================
    ## Initialising Model
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EMBED_DIMENSION = 300

    model = CBOW_1(len(vocab), EMBED_DIMENSION, device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.025)

    for epoch in range(5):
        model.train()

        epoch_loss = 0
        for mini_batch in tqdm.tqdm(train_dataloader):
            batch_loss = model(*mini_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss
        print(f"Epoch {epoch+1}: Loss {epoch_loss}")

    torch.save(model, "./cbow_model/model_cbow.pt")
    torch.save(vocab, "./cbow_model/vocab_cbow.pt")
else:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load("./cbow_model/model_cbow.pt", map_location=device)
    vocab = torch.load(f"./cbow_model/vocab_cbow.pt")

    # embedding from first model layer
    embeddings = list(model.parameters())[0]
    embeddings = embeddings.cpu().detach().numpy()

    # normalization
    embed_norms = (embeddings**2).sum(axis=1) ** (1 / 2)
    embed_norms = np.reshape(embed_norms, (len(embed_norms), 1))
    embeddings = embeddings / embed_norms

    tokens = vocab.get_itos()

    def getTopSimilarWords(word):
        idx = vocab[word]

        word_vec = embeddings[idx]
        word_vec = np.reshape(word_vec, (len(word_vec), 1))
        dists = np.matmul(embeddings, word_vec).flatten()
        topN_ids = np.argsort(-dists)[1 : 10 + 1]

        topN_dict = {}
        for sim_word_id in topN_ids:
            sim_word = vocab.lookup_token(sim_word_id)
            dist = dists[sim_word_id]
            topN_dict[sim_word] = np.round(dist, 3)
        return topN_dict

    fig, ax = plt.subplots(figsize=(8, 4))

    def display_pca_scatterplot(modesl_map, c, words, word):
        word_vectors = np.array([model_map[w] for w in words])
        twodim = PCA().fit_transform(word_vectors)[:, :2]
        # plt.figure(figsize=(5, 5))
        ax.scatter(twodim[:, 0], twodim[:, 1], edgecolors="k", c=c, label=word)
        for word, (x, y) in zip(words, twodim):
            ax.text(x, y, word)
        # plt.savefig(f"{words[0]}_cbow.png")
        # plt.show()

    U_df = pd.DataFrame(embeddings, index=vocab.get_stoi())
    model_map = {}
    c = 0
    for i in list(U_df.index):
        model_map[i] = list(U_df.loc[i])
        c = c + 1

    similar_words = getTopSimilarWords("titanic")
    print(f"titanic: {list(similar_words.keys())}")
    display_pca_scatterplot(model_map, ["pink"], list(similar_words.keys()), "titanic")

    word_list = ["book", "definitely", "drink", "beautiful", "worth"]
    color_list = ["r", "b", "g", "black", "yellow"]

    for i, word in enumerate(word_list):
        similar_words = getTopSimilarWords(word)
        print(f"{word}: {list(similar_words.keys())}")
        display_pca_scatterplot(model_map, color_list[i], list(similar_words.keys()), word)
    plt.legend()
    plt.show()
    plt.savefig(f"all_cbow.png")
