# 📚 NLP (Natural Language Processing) with Python

***
# 📌 Notebook Goals
> In this notebook we will discuss a higher level overview of the basics of Natural Language Processing, which basically consists of combining machine learning techniques with text, and using math and statistics to get that text in a format that the machine learning algorithms can understand!
-----

# 📝 Table of content

> 1. Representing text as numerical data
> 2. Reading a text-based dataset into pandas
> 3. Vectorizing our dataset
> 4. Building and evaluating a model
> 5. Comparing models
> 6. Examining a model for further insight
> 7. Practicing this workflow on another dataset
> 8. Tuning the vectorizer (discussion)

---
# 🔁 Representing text as numerical data
📌 From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):

> Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect **numerical feature vectors with a fixed size** rather than the **raw text documents with variable length**.

We will use [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to "convert text into a matrix of token counts":

📌 From the [scikit-learn documentation](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction):

> As most documents will typically use a very small subset of the words used in the corpus, the resulting matrix will have **many feature values that are zeros** (typically more than 99% of them).

> For instance, a collection of 10,000 short text documents (such as emails) will use a vocabulary with a size in the order of 100,000 unique words in total while each document will use 100 to 1000 unique words individually.

> In order to be able to **store such a matrix in memory** but also to **speed up operations**, implementations will typically use a **sparse representation** such as the implementations available in the `scipy.sparse` package.

## 📋 **Summary:**

> - `vect.fit(train)` **learns the vocabulary** of the training data
> - `vect.transform(train)` uses the **fitted vocabulary** to build a document-term matrix from the training data
> - `vect.transform(test)` uses the **fitted vocabulary** to build a document-term matrix from the testing data (and **ignores tokens** it hasn't seen before)

# 📑 Text Pre-processing

> Our main issue with our data is that it is all in text format (strings). The classification algorithms that we usally use need some sort of numerical feature vector in order to perform the classification task. There are actually many methods to convert a corpus to a vector format. The simplest is the `bag-of-words` approach, where each unique word in a text will be represented by one number.


> In this section we'll convert the raw messages (sequence of characters) into vectors (sequences of numbers).

> As a first step, let's write a function that will split a message into its individual words and return a list. We'll also remove very common words, ('the', 'a', etc..). To do this we will take advantage of the `NLTK` library. It's pretty much the standard library in Python for processing text and has a lot of useful features. We'll only use some of the basic ones here.

> Let's create a function that will process the string in the message column, then we can just use **apply()** in pandas do process all the text in the DataFrame.

>First removing punctuation. We can just take advantage of Python's built-in **string** library to get a quick list of all the possible punctuation:
>
