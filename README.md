# Kaggle-huggingface
Kaggle-huggingface



-------
-------

## huggingface / datasets
https://github.com/huggingface/datasets

Datasets is a lightweight library providing two main features:

- one-line dataloaders for many public datasets: one-liners to download and pre-process any of the number of datasets major public datasets (in 467 languages and dialects!) provided on the HuggingFace Datasets Hub. With a simple command like squad_dataset = load_dataset("squad"), get any of these datasets ready to use in a dataloader for training/evaluating a ML model (Numpy/Pandas/PyTorch/TensorFlow/JAX),

- efficient data pre-processing: simple, fast and reproducible data pre-processing for the above public datasets as well as your own local datasets in CSV/JSON/text. With simple commands like tokenized_dataset = dataset.map(tokenize_example), efficiently prepare the dataset for inspection and ML model evaluation and training.


Datasets also provides access to +15 evaluation metrics and is designed to let the community easily add and share new datasets and evaluation metrics.

Datasets has many additional interesting features:

- Thrive on large datasets: hugs Datasets naturally frees the user from RAM memory limitation, all datasets are memory-mapped using an efficient zero-serialization cost backend (Apache Arrow).

- Smart caching: never wait for your data to process several times.

- Lightweight and fast with a transparent and pythonic API (multi-processing/caching/memory-mapping).

- Built-in interoperability with NumPy, pandas, PyTorch, Tensorflow 2 and JAX.

Datasets originated from a fork of the awesome TensorFlow Datasets and the HuggingFace team want to deeply thank the TensorFlow Datasets team for building this amazing library. More details on the differences between hugs Datasets and tfds can be found in the section Main differences between hugs Datasets and tfds.


-------

# Paper 

## Datasets: A Community Library for Natural Language Processing
https://arxiv.org/pdf/2109.02846.pdf

7 Sep 2021


-------
-------

## [Code] PyTorch sentiment classifier from scratch with Huggingface NLP Library (Full Tutorial)
https://www.youtube.com/watch?v=G3pOvrKkFuk&list=PL1v8zpldgH3pQwRz1FORZdChMaNZaR3pu&index=16


- My Code: https://github.com/yk/huggingface-nlp-demo

- NLP Library: https://github.com/huggingface/nlp

- Tutorial Colab: https://colab.research.google.com/github/huggingface/nlp/blob/master/notebooks/Overview.ipynb

- Transformers Library: https://github.com/huggingface/transformers

- Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning






-------
-------

# github.com/huggingface
https://https://github.com/huggingface

-------

# github.com/huggingface/transformers
https://github.com/huggingface/transformers


-------
-------

# Hugging Face Course
https://huggingface.co/course/


asily share your fine-tuned models on the Hugging Face Hub using the push to hub API.

This video is part of the Hugging Face course: http://huggingface.co/course
Open in colab to run the code samples: 
https://colab.research.google.com/git...

You will need a Hugging Face account to manage a be able to push to the Model Hub.
Join now: http://huggingface.co/join

TensorFlow version: https://youtu.be/pUh5cGmNV8Y

Related videos:
- Navigate the Model Hub: https://youtu.be/XvSGPZFEjDY
- How to instantiate a Transformers model: https://youtu.be/AhChOFRegn4
- The Trainer API: https://youtu.be/nvBXf7s7vTI

Have a question? Checkout the forums: https://discuss.huggingface.co/c/cour...
Subscribe to our newsletter: https://huggingface.curated.co/

-------

## Hugging Face Course Chapter 1
https://www.youtube.com/playlist?list=PLo2EIpI_JMQtNtKNFFSMNIZwspj8H7-sQ

HuggingFace

### 1 4:33  Welcome to the Hugging Face course


### 2 4:36  The pipeline function


### 3 4:06  What is Transfer Learning?


### 4 5:24   The carbon footprint of Transformers


### 5 2:45  The Transformer architecture


### 6 4:46  Transformer models: Encoders


### 7 4:27  Transformer models: Decoders


### 8 6:47  Transformer models: Encoder-Decoders

-------

## Hugging Face Course Chapter 2
https://www.youtube.com/playlist?list=PLo2EIpI_JMQupmYlTIrUTWD_oV-kYA3Hx

HuggingFace



### 1 4:53  What happens inside the pipeline function? (PyTorch)


### 2 5:00  What happens inside the pipeline function? (TensorFlow)


### 3 3:20  Instantiate a Transformers model (PyTorch)


### 4 3:15  Instantiate a Transformers model (TensorFlow)


### 5 0:56  Tokenizers overview


### 6 2:53  Word-based tokenizers


### 7 3:01  Character-based tokenizers


### 8 3:29  Subword-based tokenizers


### 9 3:23  The tokenization pipeline


### 10 2:52  Batching inputs together (PyTorch)


### 11 2:51  Batching inputs together (TensorFlow)



-------

## Hugging Face Course Chapter 3: PyTorch version
https://www.youtube.com/playlist?list=PLo2EIpI_JMQvbh6diTDl2TwbAyu8QWdpx


HuggingFace


### 1 4:53  What happens inside the pipeline function? (PyTorch)


### 2 5:00  What happens inside the pipeline function? (TensorFlow)


### 3 3:20  Instantiate a Transformers model (PyTorch)


### 4 3:15  Instantiate a Transformers model (TensorFlow)


### 5 0:56  Tokenizers overview


### 6 2:53   Word-based tokenizers


### 7 3:01  Character-based tokenizers


### 8 3:29  Subword-based tokenizers


### 9 3:23  The tokenization pipeline


### 10 2:52  Batching inputs together (PyTorch)


### 11 2:5  Batching inputs together (TensorFlow)


-------

## Hugging Face Course Chapter 3: TensorFlow version
https://www.youtube.com/playlist?list=PLo2EIpI_JMQvXha8ltnkSGDfNCUE59YVm

HuggingFace

1

3:28
 
Hugging Face Datasets overview (Tensorflow)
 

2

3:09
 
Preprocessing sentence pairs (TensorFlow)
 

3

2:51
 
Keras introduction
 

4

5:05
 
Fine-tuning with TensorFlow
 

5

4:14
 
Learning rate scheduling with TensorFlow
 

6

4:11
 
TensorFlow Predictions and metrics
 

-------

## Hugging Face Course Chapter 4
https://www.youtube.com/playlist?list=PLo2EIpI_JMQvBf9VwoyXCkHldK6-4aGPQ

HuggingFace


1

3:55
 
Navigating the Model Hub
 

2

5:06
 
The Push to Hub API (PyTorch)
 

3

8:38
 
The Push to Hub API (TensorFlow)
 

4

7:55
  
Managing a repo on the Model Hub
 


-------

## Hugging Face Course Chapter 5
https://www.youtube.com/playlist?list=PLo2EIpI_JMQt7wOxx6EAnXIvJMK1hoSwm

HuggingFace


1

3:12
NOW PLAYING
Loading a custom dataset
HuggingFace

2

3:35
NOW PLAYING
Slice and dice a dataset 🔪
HuggingFace

3

2:33
NOW PLAYING
Datasets + DataFrames = ❤️
HuggingFace

4

3:27
NOW PLAYING
Saving and reloading a dataset
HuggingFace

5

3:28
NOW PLAYING
Memory mapping & streaming
HuggingFace

6

2:05
NOW PLAYING
Uploading a dataset to the Hub
HuggingFace

7

3:30
NOW PLAYING
Text embeddings & semantic search
HuggingFace


-------

## Hugging Face Course Chapter 6
https://www.youtube.com/playlist?list=PLo2EIpI_JMQshO8wBZ9Dp3E7Y4uiBmphN

HuggingFace


1

1:49
NOW PLAYING
Why are fast tokenizers called fast?
HuggingFace

2

3:11
NOW PLAYING
Fast tokenizer superpowers
HuggingFace

3

3:02
NOW PLAYING
Inside the Token classification pipeline (PyTorch)
HuggingFace

4

3:08
NOW PLAYING
Inside the Token classification pipeline (TensorFlow)
HuggingFace

5

3:23
NOW PLAYING
Inside the Question answering pipeline (PyTorch)
HuggingFace

6

3:28
NOW PLAYING
Inside the Question answering pipeline (TensorFlow)
HuggingFace

7

6:25
NOW PLAYING
Training a new tokenizer
HuggingFace

8

5:18
NOW PLAYING
What is normalization?
HuggingFace

9

2:50
NOW PLAYING
What is pre-tokenization?
HuggingFace

10

5:23
NOW PLAYING
Byte Pair Encoding Tokenization
HuggingFace

11

3:50
NOW PLAYING
WordPiece Tokenization
HuggingFace

12

8:20
NOW PLAYING
Unigram Tokenization
HuggingFace

13

5:18
NOW PLAYING
Building a new tokenizer
HuggingFace


-------

## Hugging Face Course Chapter 7
https://www.youtube.com/playlist?list=PLo2EIpI_JMQtYmOWSszkfIi4sgz2NsySi


HuggingFace

1

3:36
NOW PLAYING
Using a custom loss function
HuggingFace

2

1:51
NOW PLAYING
What is domain adaptation?
HuggingFace

3

3:22
NOW PLAYING
Data processing for Token Classification
HuggingFace

4

4:34
NOW PLAYING
Data processing for Causal Language Modeling
HuggingFace

5

2:30
NOW PLAYING
Data processing for Masked Language Modeling
HuggingFace

6

2:14
NOW PLAYING
What is perplexity?
HuggingFace

7

2:26
NOW PLAYING
Data processing for Translation
HuggingFace

8

4:42
NOW PLAYING
What is the BLEU metric?
HuggingFace

9

2:04
NOW PLAYING
Data processing for Summarization
HuggingFace

10

4:09
NOW PLAYING
What is the ROUGE metric?
HuggingFace

11

2:51
NOW PLAYING
Data processing for Question Answering
HuggingFace

12

3:16
NOW PLAYING
The Post processing step in Question Answering (PyTorch)
HuggingFace

13

3:09
NOW PLAYING
The Post processing step in Question Answering (TensorFlow)
HuggingFace

14

6:17

Data Collators: A Tour




-------

## Hugging Face Course Chapter 8
https://www.youtube.com/playlist?list=PLo2EIpI_JMQt52mV-fylktyVtznz4SU8i

HuggingFace


1

2:40

What to do when you get an error?


2

3:33

Using a debugger in a notebook


3

4:00

Using a debugger in a terminal


4

3:07

Asking for help on the forums


5

7:45

Debugging the Training Pipeline (TensorFlow)


6

4:16

Debugging the Training Pipeline (PyTorch)


7

3:11

Writing a good issue





-------
-------

