# Word2Vec

## Directory Structure

```
2022201068_assignment3
├── cbow.py
├── cbow_model
│   ├── model_cbow.pt
│   └── vocab_cbow.pt
├── co-occurrence.py
├── datasubsets
│   ├── subdata_cbow.json
│   └── subdata_svd.json
├── readme.md
└── svd_model
    ├── ipca_matrix.npy
    └── vocab.pkl
```

> **_NOTE:_** As the submission format requires no command line arguments, I have hardcoded the file names and the formats. Please follow the directory stucture in order for the code to work.

- `cbow.py`: This is the main python file in which CBOW neural model code is prvovided. By default model will run in predict state. To enable train mode, you need to turn `TRAIN=True` inside the file. When the file runs, it will run for 5 words stored in the list, we can change them in order to predict more.

- `co-occurrence.py`: This is the main python file in which Co-occurrence model code is prvovided. By default model will run in predict state. To enable train mode, you need to turn `TRAIN=True` inside the file. When the file runs, it will run for 5 words stored in the list, we can change them in order to predict more.

> **_NOTE:_** Below folders and the files in them can be found at the `google-drive` link present at the end of the readme.

- `datasubsets`: This directory contains all the dataset files which are a subset of original dataset provided which had huge number of sentences. I have takes 100000 amazon reviews to work with.

- `svd_model` and `cbow_model`: Contains trained model as well as vocabulary stored in order for python scripts to predict

## Steps to execute

- Run the below commands

```sh
$ python cbow.py
$ python co-occurrence.py
```

- The output will be the **PCA plots** of the word list.
- Terminal will show top 10 nearest words based on the cosine similarity

[Link for the files](https://drive.google.com/drive/folders/1of9JnhbB_gbhyuoo3LROudhWc_5v6kgg?usp=share_link)

---

## Inspiration

Traditionally, the task of next word prediction was performed using statistical models such as `n-grams`. N-Grams worked on the ideas that given a certain history, probability of the next word is this. Now, a research in 2013 showed that how not only previous word but the words after the certain target is also useful in order to calculate the probability. Thus comes the Word2Vec, which focuses on the context around the words, both next and previous.

## Continuous Bag of Word

Instead of only looking two words before the target word, we can also look at two words after it.

![](attachments/Pasted%20image%2020231031203422.png)

So, the dataset we are virtually building and training the model against would look like:

![](attachments/Pasted%20image%2020231031203522.png)

This is called **Continuous Bag of Words** architecture.

## Skipgram

Instead of guessing a word based on its context, this architecture tries to guess neighboring words using the current word.

![](attachments/Pasted%20image%2020231031203709.png)

This pink boxes are in different shade because this sliding window actually creates four samples in our training set.

![](attachments/Pasted%20image%2020231031203750.png)

This method is called **skipgram** architecture.

## Model Training

For simplicity, let's just focus on the _skipgram_ architecture only for now.

![](attachments/Pasted%20image%2020231031203940.png)

1. Look up embeddings
2. Calculate prediction
3. Project to output vocabulary
4. Calculate prediction error
5. Back propagate

![](attachments/Pasted%20image%2020231031204058.png)

![](attachments/Pasted%20image%2020231031204108.png)

This whole picture concludes the first step of the training. We proceed to do the same process with the next sample in our dataset, and then the next, until we have covered all the samples. That concludes one epoch of training. We repeat this for several epochs.

**Then we would have our trained model and we can extract the embedding layer (matrix) from it and use it for any other application.**

## Negative Sampling

As seen in the neural model, **3rd step was most computationally intensive**, especially knowing that we will do it once for every training sample in our dataset (which even if we take the window size of 5 can shoot easily to millions).

> To make this approach efficient, we will shift from predicting next word and just focus on the embeddings.

![](attachments/Pasted%20image%2020231031205616.png)

We will switch to the model that takes the input and output word, and outputs a score indicating if they're neighbors or not. (0 for `not neighbors`, 1 for `neighbors`)

> This switch changes the model we need from a neural network, to a logistic regression model thus it becomes much simpler and much faster to calculate.

**New dataset would look like**

![](attachments/Pasted%20image%2020231031210020.png)

Now using logistic regression this can be computer really fast, but **one problem is if we have all 1's in our dataset then our model will learn just to output 1 for any input.** _In order words, model will absolutely nothing._

To address this, we need to introduce _negative samples_ to our dataset, which are sample of words that are not neighbors. Our model needs to return 0 for those samples. The output words are randomly sampled from the dataset which have 0 in front

![](attachments/Pasted%20image%2020231031210008.png)

## Skipgram with Negative Sampling

![](attachments/Pasted%20image%2020231031210342.png)

## Word2Vec Training Process

1. We pre-process the text we are training the model against. In this step, we determine the size of our vocabulary and which words belong to it.
2. We create two matrices, both of these matrices have an embedding for each word in our vocabulary (`vocab_size x embedding_dimension`)
   1. `Embedding Matrix`
   2. `Context Matrix`
3. Initialise these matrices with random values.
4. In each training step, we take one positive example and its associated negative examples.

![](attachments/Pasted%20image%2020231031210807.png)

5. Now we have say four words: the input word `not`, the true neighbor word `thou`, and two negative samples `aaron`, `taco`.
6. We look up the embedding for `input word from Embedding matrix` and `context words from Context Matrix`.

![](attachments/Pasted%20image%2020231031211012.png)

7. We take dot product of the input embedding with each of the context embeddings. In each case, the result would be a number, which indicates the similarity of the input and context embeddings.

![](attachments/Pasted%20image%2020231031211113.png)

8. Turn these scores in some sort of probabilities. So we use `sigmoid function` here.

![](attachments/Pasted%20image%2020231031211154.png)

9. Now we can calculate loss or error in the prediction.

![](attachments/Pasted%20image%2020231031211244.png)

10. Then the "learning" part comes, we can use this error to adjust the embeddings of `not`, `thou`, `aaron`, and `taco` so the next time we make this calculation, the results would be closer to the target scores.

![](attachments/Pasted%20image%2020231031211345.png)

11. The embeddings continue to be improved while we cycle through our entire dataset for a number of times. We can then stop the training process, discard the `Context` matrix, and use the `Embeddings` matrix as our pre-trained embeddings for the next task.

## Hyperparameters

1. Window Size
2. Number of Negative Samples

## References

https://jalammar.github.io/illustrated-word2vec/
