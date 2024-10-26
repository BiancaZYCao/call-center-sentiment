import os
import re
import ast
import json
import spacy
import torch
import joblib
import PyPDF2
import string
import fasttext
import numpy as np
import pandas as pd
import pickle
import gensim.downloader as api
from collections import defaultdict
from itertools import islice
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from gensim.models import TfidfModel

#import inflect
import TextPreprocessing as tp
#import SingletonMeta
from openai import OpenAI

import nltk
from nltk import pos_tag
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# from sentence_transformers import SentenceTransformer
from llama_index.core import StorageContext, load_index_from_storage
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from TextPreprocessing import text_preprocessing
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast
from transformers import BertTokenizer, BertModel

import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load the model and tokenizer, and move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for TopicModel
SIMILARITY_THRESHOLD = 0.75

NUM_OF_TOPICS = 2

NUM_OF_TOPIC_QUESTIONS = 2

# Load the pre-trained BERT model
#model = SentenceTransformer('all-MiniLM-L6-v2')

# Vector store path
vector_store_path = "./gpt_store3"

# Intent dictionary
intent_dictionary_file = "./intent_dictionary.json"

# Set the model to use:
# "entity" or "lda" or "bertopic"
model_to_use = "entity"

nlp = spacy.load('en_core_web_lg')
stop_words = set(stopwords.words('english'))

# Define a metaclass SingletonMeta
class SingletonMeta(type):
    # Dictionary to store instances of classes
    _instances = {}

    # Override the __call__ method of the metaclass
    def __call__(cls, *args, **kwargs):
        # Check if the class is not already instantiated
        if cls not in cls._instances:
            # If not, create a new instance and store it in _instances dictionary
            cls._instances[cls] = super().__call__(*args, **kwargs)
        # Return the existing instance if already instantiated
        return cls._instances[cls]

class TopicModel(metaclass=SingletonMeta):
    stopwords = []

    questions_answers = {}
    
    intents = {
            "apply credit card": {
                "unigrams": {"apply": 2, "credit": 2, "card": 2, "krisflyer": 2, "application": 2, "account": 2, "process": 1, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "apply card": 4, "apply krisflyer": 4},
                "trigrams": {"apply credit card": 5, "apply krisflyer card": 5, "credit card application": 5, "apply for card": 5, "apply for credit card": 5},
                "patterns": [r"(how|want) to .*? apply .*? credit card", r"process of .*? credit card application", r"apply (for)? (a|the)? credit card", 
                             r"how (can|do) (i|someone) apply (for)? (a|the)? credit card", r"i want to apply (for)? (a|the)? credit card"]
            },
            # "enquire credit card miles": {
            #     "unigrams": {"mile": 4, "krisflyer": 3, "credit": 2, "card": 2, "account": 2, "points": 2, "redeem":2, "compensation": 2, "transfer": 2, "convert": 2, "conversion": 2, "exchange": 2, "rate": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
            #     "bigrams": {"credit card": 3, "air mile": 4, "krysflyer mile": 4, "krysflyer point": 4, "card mile":4, "mile point": 3, "convert mile": 4, "conversion rate": 2, "exchange mile": 4, "redeem mile": 4, "transfer miles": 4, "foreign transaction": 2},
            #     "trigrams": {"redeem krisflyer mile": 5, "convert krisflyer mile": 5, "transfer krisflyer mile": 5, "credit card mile": 4, "convert to mile": 5, "mile exchange rate": 5},
            #     "patterns": [r"(how|want) to (redeem|convert|transfer)? mile", r"(how|want) to (redeem|convert|transfer)? krisflyer (mile|point)",
            #                  r"enquire (about|regarding)? (credit)? card mile", r"tell me about (credit)? card mile", r"(how|can I) (get|use|redeem) (credit)? card mile"]
            # },
            "waive credit card fee": {
                "unigrams": {"waive": 3, "waiver": 3, "credit": 2, "card": 2, "account": 2, "fee":2, "minimum": 2, "payment": 2, "annual": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "grace period": 3, "waive fee": 4, "late fee": 4, "late payment": 4, "interest rate": 3, "annual fee": 2, "annual payment": 2, "minimum payment": 3, "minimum fee": 4},
                "trigrams": {"waive card fee": 5, "waive card payment": 5, "annual interest rate": 3, "waive late fee": 4, "waive late payment": 4, "fee waiver": 4, "credit card fee": 2},
                "patterns": [r"(how|want) to waive card fee", r"(how|want) to waive credit card fee", r"waive (the)? (credit)? card fee", r"can you waive (my|the)? credit card fee",
                             r"(how can i|please) waive (the)? card fee"]
            },
            "enquire credit card rewards": {
                "unigrams": {"reward": 3, "credit": 2, "card": 2, "point": 2, "redeem":2, "account": 2, "transfer": 2, "convert": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "redeem rewards": 4, "rewards point": 3, "convert point": 4, "transfer point": 4, "foreign transaction": 2},
                "trigrams": {"redeem rewards point": 5, "convert rewards point": 5, "transfer rewards point": 5},
                "patterns": [r"(how|want) to (redeem|convert|transfer)? rewards", r"(how|want) to (redeem|convert|transfer)? rewards points", r"enquire (about|regarding)? (credit)? card rewards",
                             r"(how|where) can i see (credit)? card rewards", r"tell me about (credit)? card rewards"]
            },
            # "enquire credit card cashback": {
            #     "unigrams": {"cashback": 3, "credit": 2, "card": 2, "point": 2, "redeem":2, "amount": 2, "account": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
            #     "bigrams": {"credit card": 3, "redeem cashback": 5, "cashback amount": 4, "card cashback": 4, "credit cashback": 4, "about cashback": 5, "foreign transaction": 2},
            #     "trigrams": {"amount of cashback": 5, "credit card cashback": 5},
            #     "patterns": [r"(how|where) can i (redeem|get) (credit)? card cashback", r"how much is the cashback", r"tell me (more)? about (credit)? card cashback"]
            # },
            # "enquire credit card plans": {
            #     "unigrams": {"plan": 3, "credit": 2, "card": 2, "limit": 2, "interest": 2, "balance": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
            #     "bigrams": {"credit card": 3, "card plan": 4, "credit plan": 4, "card fee": 3, "card balance": 3, "annual fee": 2, "minimum payment": 2, "interest rate": 2, "grace period": 2, "late fee": 2, "foreign transaction": 2},
            #     "trigrams": {"credit card plan": 5, "credit card balance": 5, "amount of cashback": 5, "credit card limit": 5, "foreign transaction fee": 3},
            #     "patterns": [r"tell me (about|more)? (credit)? card plans", r"enquire (about|regarding)? (credit)? card plans", r"(which|what) (credit)? card plans are available"]
            # },
            # "compare credit cards": {
            #     "unigrams": {"compare": 3, "which":3, "credit": 2, "card": 2, "plans": 2, "limit": 2, "interest": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
            #     "bigrams": {"credit card": 3, "compare card": 4, "which card": 4},
            #     "trigrams": {"compare credit card plan": 5, "compare credit card": 5, "which credit card": 5, "compare cards": 5, "compare plans": 2},
            #     "patterns": [r".*compare .*?(credit)? cards", r"compare (the)? (available|best) (credit)? cards"]
            # },
            # "enquire property loan": {
            #     "unigrams": {"apply": 2, "property": 3, "loan": 2, "hdb": 3, "bank": 2, "limit": 2, "mortgage": 3, "interest": 2, "rate": 2, "amortization": 3, "equity": 2, "refinance": 3,
            #                  "valuation": 2, "private": 2, "account": 2, "instalment": 2, "float": 2, "fix": 2, "bto": 3, "sibor": 3, "cpf": 2, "rate": 2,
            #                  "duration": 2, "flat": 3, "resale": 3, "migration": 2, "buy": 2, "sell": 2, "transfer": 2, "purchase": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
            #     "bigrams": {"property loan": 3, "mortgage loan": 4, "hdb loan": 4, "bank loan": 3, "interest rate": 2, "loan period": 3, "down payment": 2, "loan tenure": 3,
            #                 "floating rate": 3, "float rate": 3, "fix rate": 3, "fixed rate": 3, "home loan": 4, "bank account": 2, "resale flat": 3, "loan application": 3,
            #                 "buy flat": 3, "sell flat": 3, "buy house": 3, "housing loan": 3, "bank account": 2, "purchase flat": 3, "private property": 3},
            #     "trigrams": {"enquire property loan": 5, "enquire home loan": 5, "enquire mortgage loan": 5, "float interest rate": 4, "fix interest rate": 4, "buy resale flat": 4,
            #                  "buy resale house": 4, "apply property loan": 5, "apply home loan": 5, "apply mortgage loan": 5,},
            #     "patterns": [r"enquire (about|regarding)? (a|the)? property loan", r"tell me (about|more about) (a|the)? property loan",
            #                  r"apply (for)? (a|the)? property loan", r"how (can|do) (i|someone) apply (for)? (a|the)? property loan",
            #                  r"i want to apply (for)? (a|the)? property loan"]
            # },
            # "enquire travel insurance": {
            #     "unigrams": {"enquire": 2, "travel": 3, "insurance": 2, "coverage": 3, "medical": 2, "limit": 2, "policy": 2, "trip": 2, "accident": 2, "compensation": 2, "emergency": 2, "terrorism": 3, "belonging": 2, "assistance": 3,
            #                  "premium": 2, "private": 2, "region": 2, "luggage": 2, "baggage": 2, "delay": 2, "flight": 3, "claim": 3, "process": 2, "sick": 2,
            #                  "duration": 2, "policy": 2, "cancel": 2, "buy": 2, "process": 2, "transfer": 2, "purchase": 2, "geographical": 2, "covid-19": 3, "group": 1, "personal": 1},
            #     "bigrams": {"travel insurance": 4, "insurance coverage": 4, "insurance premium": 4, "medical expense": 3, "policy number": 3, "coverage amount": 2, "regional coverage": 2, "flight delay": 2, "flight cancel": 3,
            #                 "flight postpone": 3, "luggage delay": 4, "baggage delay": 4, "luggage loss": 4, "lost luggage": 4, "baggage loss": 4, "lost baggage": 4, "lost belonging": 4, "personal belonging": 2, "trip cancel": 3, "personal accident": 3,
            #                 "policy number": 2, "car rental": 2, "claim process": 2, "geographical coverage": 3, "trip cancellation": 2, "flight cancellation": 3, "emergency evacuation": 3},
            #     "trigrams": {"apply travel insurance": 5, "how to claim": 3, "travel insurance enquiry": 5, "enquire travel insurance": 4, "buy travel insurance": 4, "cancel travel insurance": 4,
            #                  "purchase travel insurance": 3, "cannot find luggage": 4, "cannot find my luggage": 4, "cannot find baggage": 4, "cannot find my baggage": 4, "lost my luggage": 4, "lost my baggage": 4},
            #     "patterns": [r"enquire (about|regarding)? (a|the)? travel insurance", r"tell me (about|more about) (a|the)? travel insurance",
            #                  r"buy (a|the)? travel insurance", r"how (can|do) (i|someone) buy (a|the)? travel insurance", r"i want to buy (a|the)? travel insurance"]
            # },

        }

    def __init__(self, model=model_to_use):
        self.nlp = spacy.load('en_core_web_lg')
        self.nlp = self.add_stopwords(self.nlp)
        self.stop_words = self.nlp.Defaults.stop_words
        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.intents_dictionary = {}
        # Load the intent dictionary from a JSON file
        # with open('intent_dictionary.json', 'r') as f:
        #     self.intents_dictionary = json.load(f)
        
        self.initOpenAI()
        if (model == "lda"):
            self.load_lda()
        elif (model == "bertopic"):
            self.load_bertopic()
        self.topics = []
        self.topicsAndQuestions = {}

    def initOpenAI(self):
        os.environ['OPENAI_API_KEY'] = "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"
        self.model_id = "ft:gpt-4o-mini-2024-07-18:personal::A0l6mkLn"

        self.client = OpenAI(
            # This is the default and can be omitted
            api_key = os.environ.get("OPENAI_API_KEY"),
        )

    # Function to detect acronyms, including ones with periods like "U.O.B."
    def is_acronym(word):
        return bool(re.match(r'([A-Z]\.){2,}|[A-Z]{2,}', word))

    def getOpenAIResponses(self, prompt):
        #prompt = "What are the credit cards with KrisFlyer?"

        response = self.client.chat.completions.create(
            model = self.model_id,
            messages=[
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "You are a customer service representative for financial information across banks in Singapore. Summarise your response within 20 words."
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    }
                ]
                }
            ],
            temperature=1,
            max_tokens=20,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "text"
            }
        )

        return response.choices[0].message.content


    def generate_response(self, prompt):        
        # If not already done, initialize ‘index’ and ‘query_engine’
        if not hasattr(self, "index"):
            # rebuild storage context and load index
            storage_context = StorageContext.from_defaults(persist_dir=vector_store_path)
            self.index = load_index_from_storage(storage_context=storage_context, index_id="vector_index")
            self.query_engine = self.index.as_query_engine()

        # Submit query
        response = self.query_engine.query(prompt)

        return response.response

    # Function to get response from RAG based on the question or prompt.
    # Parameters:
    # input_text - the question or prompt
    # Return a text response.
    def getResponseForQuestions(self, input_text):
        # Get the retrieved context from the index
        context = self.generate_response(input_text)
       

        # Integrate the context with the input text for the generative model
        prompt = f"Context: {context}\n\nQuestion: {input_text}\n\nAnswer:"

        # Call the OpenAI API to generate a response using GPT-4o-mini
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",  # or the appropriate model name
            messages=[
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "You are a customer service representative for financial information across banks in Singapore. Summarise your response within 40 words."
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    }
                ]
                }
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.5
        )

        # Extract the text from the completion
        generated_text = completion.choices[0].message.content.strip()

        return generated_text
    
    # Function to extract text from a PDF file
    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
            return text

    # Function to preprocess the extracted text
    def preprocess_text_from_pdf(self, text):
        stop_words = set(stopwords.words('english'))
        # Lowercase and remove punctuation
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        # Tokenize the text
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        
        return tokens

    # Function to load all PDFs from a directory
    def load_pdfs_from_directory(self, directory):
        pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
        texts = []
        for pdf_file in pdf_files:
            raw_text = self.extract_text_from_pdf(pdf_file)
            processed_text = self.preprocess_text_from_pdf(raw_text)
            texts.append(processed_text)
        return texts

    def create_dictionary(self, pdf_directory):
        # Specify the directory containing PDF files
        #pdf_directory = "path_to_your_pdf_directory"  # Replace with your directory path
        
        # Extract and preprocess text from all PDF files in the directory
        texts = self.load_pdfs_from_directory(pdf_directory)
        
        # Create a Gensim dictionary from the preprocessed texts
        dictionary = corpora.Dictionary(texts)
        
        # Output the dictionary (you can save or print its contents)
        #print(dictionary.token2id)
        
        # Create a bag-of-words corpus if needed
        corpus = [dictionary.doc2bow(text) for text in texts]
        
        # Example: Save the dictionary to disk
        dictionary.save('./dictionary/pdf_dictionary.dict')
        print("Dictionary created!")

    # Function to load the index from storage and create a retriever
    def load_and_create_retriever(self, persist_dir):
        # Load the index from the persisted storage directory
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        
        # Convert the index into a retriever
        #retriever = index.as_retriever(similarity_top_k=1)

        retriever = index.as_query_engine(
                        response_mode="tree_summarize",
                        verbose=True,
                    )

        return retriever

    # Function to query the index and retrieve relevant information
    def retrieve_information(self, retriever, prompt):        
        # Use the retriever to fetch the relevant information for the prompt
        retrieved_docs = retriever.query(prompt)
        #retrieved_docs = retriever.retrieve(prompt)

        #print("RESPONSE = ", retrieved_docs.response)

        # You can process the retrieved documents if needed
        #return retrieved_docs[0].text
        return retrieved_docs

    # Function to generate a response using the model
    def get_response(self, prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move input to GPU
        attention_mask = inputs.get("attention_mask", None)
        outputs = model.generate(inputs["input_ids"], attention_mask=attention_mask, max_length=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response


    # Function to load the index from storage
    def load_gpt_index(self, storage_path):
        # Load the pre-built index from storage path
        index = load_index_from_storage(storage_path)

        return index

    # Function to create a retriever and get information based on a prompt
    def create_retrieval_function(self, index, llm):
        retriever = index.as_retriever()
    
        def retrieve(prompt):
            # Query the retriever with the prompt to get relevant documents
            retrieved_docs = retriever.query(prompt)
            
            # Combine the retrieved information and pass it to the LLM for a final response
            context = " ".join([doc.get_text() for doc in retrieved_docs])
            
            # Ask the LLM to generate a final response based on the context
            full_prompt = f"Based on the following information, answer the query:\n\n{context}\n\nQuery: {prompt}"
            response = llm(full_prompt)

            return response
        
        return retrieve

    def generate_response(self, prompt):        
        # If not already done, initialize ‘index’ and ‘query_engine’
        if not hasattr(self, "index"):
            # rebuild storage context and load index
            storage_context = StorageContext.from_defaults(persist_dir=vector_store_path)
            self.index = load_index_from_storage(storage_context=storage_context, index_id="vector_index")
            self.query_engine = self.index.as_query_engine()

        # Submit query
        response = self.query_engine.query(prompt)

        return response.response

    def remove_consecutive_duplicates(self, sentence):
        # This regular expression matches any word that appears consecutively
        pattern = r'\b(\w+)\s+\1\b'
        
        # Substitutes the consecutive duplicate with a single instance of the word
        result = re.sub(pattern, r'\1', sentence)
        
        return result
    
    def remove_duplicates(self, word_list):
        # Split the trigram into individual words
        ngram = word_list[0].split()

        # Remove 'mile' and 'card' (and any other words in the trigram) from the list if they exist
        result = [word for word in word_list if word not in ngram]
        
        return result

    def load_lda_model(self, path):
        """Load the trained LDA model."""
        model = joblib.load(path)
        #print(f"LDA model loaded from {path}")
        return model

    def load_lda_vectorizer(self, path):
        """Load the trained vectorizer."""
        vectorizer = joblib.load(path)
        #print(f"Vectorizer loaded from {path}")
        return vectorizer
    

    def getTopics(self, sentence, n_top_words=NUM_OF_TOPICS, model=model_to_use):
        if model == "bertopic":
            return self.getBERTopics(sentence, n_top_words=n_top_words)
        elif model == "lda":
            return self.getLDATopics(sentence, n_top_words=n_top_words)
        elif model == "entity":
            return self.getEntityTopic(sentence, n_top_words=n_top_words)
        
    def load_lda(self):
        # Path to saved LDA model and vectorizer
        lda_model_path = './models/lda/lda_model_5.pkl'
        vectorizer_path = './models/lda/vectorizer_5.pkl'

        # Set paths for loading the model and vectorizer
        self.model_path = lda_model_path
        self.vectorizer_path = vectorizer_path

        # Load the LDA model and vectorizer
        self.lda_model = self.load_lda_model(self.model_path)
        self.vectorizer = self.load_lda_vectorizer(self.vectorizer_path)
    
    def load_bertopic(self):
        # Initialize vectorizer with n-grams (1 to 3)
        vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words='english')

        # Initialize BERTopic with the n-gram vectorizer
        self.topic_model = BERTopic(vectorizer_model=vectorizer_model)

        # Load the saved model
        model_path = "./models/bertopic_model11"  # Path where the model was saved
        self.topic_model = BERTopic.load(model_path)
        #print("BERTopic model loaded successfully.")


    # Helper function to get n-grams
    def get_ngrams(self, input_tokens, n):
        return zip(*[input_tokens[i:] for i in range(n)])
    
    # Function to lemmatize words
    def lemmatize(self, text):
        doc = nlp(text)
        return [token.lemma_ for token in doc]
    
    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)
    
    def remove_stopwords(self, tokens):
        return [word for word in tokens if word not in stop_words]
    
    def normalize_whitespace(self, text):
        return ' '.join(text.split())
    
    # Define the recognize_intent function
    def recognize_intent(self, user_input):
        #print("\n==========\nUSER INPUT = ", user_input)                    
        # Preprocess user input
        user_input = self.remove_punctuation(user_input.lower())
        user_input = self.normalize_whitespace(user_input)
        lemmatized_input = self.lemmatize(user_input)
        tokens = lemmatized_input  #self.remove_stopwords(lemmatized_input)
        
        # Initialize score and keyword dictionary for each intent
        intent_scores = defaultdict(int)
        
        # Separate sets for trigrams, bigrams, and unigrams to avoid duplicates
        trigram_keywords = set()
        bigram_keywords = set()
        unigram_keywords = set()
    
        # Generate unigrams, bigrams, and trigrams from lemmatized user input
        unigrams = tokens
        bigrams = [' '.join(bigram) for bigram in self.get_ngrams(tokens, 2)]
        trigrams = [' '.join(trigram) for trigram in self.get_ngrams(tokens, 3)]
    
        # Match unigrams, bigrams, and trigrams with weighted keywords
        for intent, data in self.intents.items():
            #print(f"\nProcessing intent: {intent}")
            
            for trigram in trigrams:
                if trigram in data["trigrams"]:
                    intent_scores[intent] += data["trigrams"][trigram]
                    trigram_keywords.add(trigram)
                    #print(f"Matched trigram '{trigram}' for intent '{intent}' - Score: {intent_scores[intent]}")
            
            for bigram in bigrams:
                if bigram in data["bigrams"]:
                    intent_scores[intent] += data["bigrams"][bigram]
                    bigram_keywords.add(bigram)
                    #print(f"Matched bigram '{bigram}' for intent '{intent}' - Score: {intent_scores[intent]}")
    
            for unigram in unigrams:
                if unigram in data["unigrams"]:
                    intent_scores[intent] += data["unigrams"][unigram]
                    unigram_keywords.add(unigram)
                    #print(f"Matched unigram '{unigram}' for intent '{intent}' - Score: {intent_scores[intent]}")
    
            # Check patterns using regular expressions
            for pattern in data["patterns"]:
                if re.search(pattern, user_input):
                    intent_scores[intent] += 5  # Add pattern match boost
                    trigram_keywords.add(f"Pattern match: {pattern}")  # Add pattern as trigram for sequence priority
                    #print(f"Matched pattern '{pattern}' for intent '{intent}' - Score: {intent_scores[intent]}")
    
        # Identify the intent with the highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            #print(f"\nBest intent: {best_intent} with score {intent_scores[best_intent]}\n")
            
            # Combine keywords in the order: trigrams, bigrams, unigrams
            combined_keywords = list(trigram_keywords) + list(bigram_keywords) + list(unigram_keywords)
            
            # Remove unigrams and bigrams that are part of a trigram
            for unigram in list(unigram_keywords):  # Iterate over the unigrams
                for phrase in bigram_keywords.union(trigram_keywords):  # Check both bigrams and trigrams
                    if unigram in phrase.split():  # If unigram is part of a bigram or trigram
                        if unigram in combined_keywords:  # Check if the unigram is in the list before removing
                            combined_keywords.remove(unigram)  # Remove the unigram
    
            return best_intent, combined_keywords
        return "unknown_intent", []


    def getEntityTopic(self, text, n_top_words=NUM_OF_TOPICS):
        prompt = "Rephrase the following text into formal and concise language: " + text        
        text = self.getOpenAIResponses(prompt)

        # Preprocess the text
        text = self.preprocess_text_2(text)

        intent, keywords = self.recognize_intent(text)
        print("INTENT = ", intent)
        print("KEYWORDS = ", keywords)
        self.topics = keywords
        if intent != "unknown_intent":
            self.topics.insert(0, intent)

        self.topics = self.remove_duplicates(self.topics)

        return self.topics[:n_top_words + 1]

    def predict_intent(self, user_input):
        # Call recognize_intent for each user input
        best_intent, _ = self.recognize_intent(user_input)
        
        return best_intent

    def predict_topics(self, user_input):
        #print("user_input = ", user_input)
        # Call recognize_intent for each user input
        _, predicted_topics = self.recognize_intent(user_input)
        #print("predicted_topics = ", predicted_topics)
        
        return predicted_topics

    def evaluate_intent_recognition(self, test_dataset_filepath):
        # Load the test dataset
        df = pd.read_csv(test_dataset_filepath)

        # Generate predicted intents using the model
        df['Predicted Intent'] = df['User Input'].apply(self.predict_intent)

        # Add the "Result" column (1 if correct, 0 if incorrect)
        df['Result'] = (df['Predicted Intent'] == df['Correct Intent']).astype(int)

        # Save the updated dataframe back to the file with "Predicted Intent" and "Result" columns
        df.to_csv(test_dataset_filepath, index=False)

        # Calculate accuracy, precision, recall, F1-score
        accuracy = accuracy_score(df['Correct Intent'], df['Predicted Intent'])
        precision = precision_score(df['Correct Intent'], df['Predicted Intent'], average='weighted')
        recall = recall_score(df['Correct Intent'], df['Predicted Intent'], average='weighted')
        f1 = f1_score(df['Correct Intent'], df['Predicted Intent'], average='weighted')

        print("=========================")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print("=========================")

        # Ensure the unique labels are correctly sorted
        unique_labels = sorted(df['Correct Intent'].unique())  # Sort the labels for consistent order

        # Generate confusion matrix
        conf_matrix = confusion_matrix(df['Correct Intent'], df['Predicted Intent'])

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))        
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=unique_labels, yticklabels=unique_labels)  # Use the sorted labels
        plt.xlabel('Predicted Intent')
        plt.ylabel('True Intent')
        plt.show()

    # Function to preprocess text and generate unigrams, bigrams, and trigrams
    def preprocess_text_with_ngrams(self, text):
        stop_words = set(stopwords.words('english'))
        # Lowercase and remove punctuation
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = self.lemmatize(text)
        # Tokenize the text
        #tokens = word_tokenize(text)       
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        tagged_tokens = nltk.pos_tag(tokens)
        tokens = [word for word, pos in tagged_tokens if pos != 'PRP']  # 'PRP' is the tag for pronouns
    
        # Generate unigrams (original tokens), bigrams, and trigrams
        unigrams = tokens
        bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
        trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]
    
        # Combine unigrams, bigrams, and trigrams
        all_ngrams = unigrams + bigrams + trigrams
        return all_ngrams

    def generate_ngrams(self, text):
        # Tokenize the text into words
        tokens = word_tokenize(text.lower())
        
        # Generate unigrams, bigrams, and trigrams
        unigrams = tokens
        bigrams = [' '.join(gram) for gram in ngrams(tokens, 2)]
        trigrams = [' '.join(gram) for gram in ngrams(tokens, 3)]
        
        # Combine all n-grams and remove duplicates by converting to a set, then back to list
        all_ngrams = list(set(unigrams + bigrams + trigrams))
        
        return all_ngrams

    def replace_spaces_in_tokens(self, token_lists):
        new_token_lists = []
        
        for tokens in token_lists:
            new_tokens = []
            for token in tokens:
                # Replace any spaces in the token with an underscore
                new_token = token.replace(" ", "_")
                new_tokens.append(new_token)
            
            new_token_lists.append(new_tokens)
        
        return new_token_lists

    def remove_unigrams_in_ngrams(self, tokens):
        unigrams = set(tokens)
        bigrams = set()
        trigrams = set()
    
        # Create bigrams and trigrams from the list
        for i in range(len(tokens) - 1):
            bigrams.add(tokens[i] + " " + tokens[i + 1])
            if i < len(tokens) - 2:
                trigrams.add(tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2])
    
        # Remove unigrams if they are part of any bigram or trigram
        filtered_tokens = [token for token in tokens if token not in unigrams or all(
            token not in bigram.split() and token not in trigram.split() for bigram in bigrams for trigram in trigrams)]
    
        return filtered_tokens

    def get_ngram_embedding(self, ngram, embedding_model):
        words = ngram.split()  # Split n-gram into individual words
        embeddings = []
        
        for word in words:
            if word in embedding_model:
                embeddings.append(embedding_model[word])
    
        if embeddings:
            # Calculate the average embedding for the n-gram
            return np.mean(embeddings, axis=0)
        else:
            return None  # Return None if no embeddings were found for this n-gram

    def calculate_lists_similarity(self, list1, list2, fasttext_model):
        # Get embeddings for each n-gram in list1 and list2
        embeddings1 = [self.get_ngram_embedding(ngram, fasttext_model) for ngram in list1]
        embeddings2 = [self.get_ngram_embedding(ngram, fasttext_model) for ngram in list2]
    
        # Filter out None values (cases where no embedding was found)
        embeddings1 = [embedding for embedding in embeddings1 if embedding is not None]
        embeddings2 = [embedding for embedding in embeddings2 if embedding is not None]
    
        # Check if there are valid embeddings in both lists
        if embeddings1:
            avg_embedding1 = np.mean(embeddings1, axis=0).reshape(1, -1)
        else:
            return 0  # Return 0 similarity if no valid embeddings found in list1
    
        if embeddings2:
            avg_embedding2 = np.mean(embeddings2, axis=0).reshape(1, -1)
        else:
            return 0  # Return 0 similarity if no valid embeddings found in list2
    
        # Calculate cosine similarity between the two average embeddings
        similarity = cosine_similarity(avg_embedding1, avg_embedding2)[0][0]
        print("Similarity =", similarity)
        return similarity


    def calculate_similarity_score(self, user_input_path, dictionary_path, model_path, vectorizer_path, model_type="LDA"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"        
        
        # Load user input from CSV file under the column "User Input"
        csv_file_path = user_input_path
        df = pd.read_csv(csv_file_path)
        
        # Preprocess the text in the "User Input" column        
        processed_texts = df['User Input'].apply(self.preprocess_text).tolist()
        processed_ngram_texts = df['User Input'].apply(self.preprocess_text_with_ngrams).tolist()

        predicted_topics = []

        # Load the specified topic model
        if model_type == "BERTopic":
            # Load BERTopic model
            topic_model = BERTopic.load(model_path)

            # Load the saved dictionary
            dictionary = corpora.Dictionary.load(dictionary_path)
            
            # Generate predicted topics using BERTopic
            for processed_text in processed_texts:
                topics, _ = topic_model.transform(processed_text)                
            
                # Extract unique topics and top words per topic                     
                for topic in set(topics):
                    if topic != -1:  # Ignore outliers in BERTopic
                        words = [word for word, _ in topic_model.get_topic(topic)]
                        predicted_topics.append(words)

            fasttext_model = KeyedVectors.load_word2vec_format("./models/wiki-news-300d-1M-subword.vec")

            temp_score = 0
            total = len(predicted_topics)
            count = 0
            for processed_ngram_text, predicted_topic in zip(processed_ngram_texts, predicted_topics):
                print("processed_ngram_text = ", processed_ngram_text)
                print("predicted_topic = ", predicted_topic)                
                similarity = self.calculate_lists_similarity(processed_ngram_text, predicted_topic, fasttext_model)
                temp_score += similarity
                if similarity >= 0:
                    count += 1
                
            average_cosine_similarity_score = temp_score / count
        
        elif model_type == "LDA":
            # Load LDA model
            topic_model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)

            # Load the saved dictionary
            dictionary = corpora.Dictionary.load(dictionary_path)
            
            # Transform the new documents to match the vectorizer
            doc_term_matrix = vectorizer.transform(processed_texts)
    
            # Get topic distribution for each new document
            topic_distributions = topic_model.transform(doc_term_matrix)
        
            # Get feature names from the vectorizer's vocabulary
            feature_names = vectorizer.get_feature_names_out()
            n_top_words = 10

            for i, dist in enumerate(topic_distributions):
                # Get the most likely topic for this document
                most_likely_topic = dist.argmax()
                # Get the top words for each topic
                top_words = [feature_names[j] for j in topic_model.components_[most_likely_topic].argsort()[:-n_top_words - 1:-1]]
                predicted_topics.append(top_words)

            fasttext_model = KeyedVectors.load_word2vec_format("./models/wiki-news-300d-1M-subword.vec")

            temp_score = 0
            total = len(predicted_topics)
            count = 0
            for processed_ngram_text, predicted_topic in zip(processed_ngram_texts, predicted_topics):
                print("processed_ngram_text = ", processed_ngram_text)
                print("predicted_topic = ", predicted_topic)                
                similarity = self.calculate_lists_similarity(processed_ngram_text, predicted_topic, fasttext_model)
                temp_score += similarity
                if similarity >= 0:
                    count += 1
                
            average_cosine_similarity_score = temp_score / count
            
        elif model_type == "IRS":  # If model_type is "Intent Recognition System"            
            for processed_text in processed_texts:
                predicted_topics.append(self.predict_topics(processed_text))

            fasttext_model = KeyedVectors.load_word2vec_format("./models/wiki-news-300d-1M-subword.vec")

            temp_score = 0
            total = len(predicted_topics)
            count = 0
            for processed_ngram_text, predicted_topic in zip(processed_ngram_texts, predicted_topics):
                print("processed_ngram_text = ", processed_ngram_text)
                print("predicted_topic = ", predicted_topic)                
                similarity = self.calculate_lists_similarity(processed_ngram_text, predicted_topic, fasttext_model)
                temp_score += similarity
                if similarity >= 0:
                    count += 1
                
            average_cosine_similarity_score = temp_score / count
        else:
            raise ValueError("Unsupported model type. Please use 'BERTopic' or 'LDA' or 'IRS'.")           
        
        print(f'Average Similarity Score for {count} records: {average_cosine_similarity_score}')
          

    def getLDATopics(self, sentence, n_top_words=NUM_OF_TOPICS):
        # Step 1: Preprocess the entire document as one unit
        preprocessed_doc = self.preprocess_text(sentence)

        # Step 2: Transform the document to match the vectorizer (which should have been trained with n-grams)
        doc_term_matrix = self.vectorizer.transform([preprocessed_doc])  # Treat as a single document

        # Step 3: Get topic distribution for the document
        topic_distribution = self.lda_model.transform(doc_term_matrix)[0]  # Only one document, so we take the first element

        # Step 4: Get the feature names (words or n-grams) from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Step 5: Print the topic distribution and the top n-grams for the most likely topic
        #print(f"\nDocument topic distribution: {topic_distribution}")
        
        # Get the most likely topic for this document
        most_likely_topic = topic_distribution.argmax()
        #print(f"Most likely topic: Topic {most_likely_topic}")
        
        # Display the top n-grams for the most likely topic
        top_words = [feature_names[j] for j in self.lda_model.components_[most_likely_topic].argsort()[:-n_top_words - 1:-1]]
        if len(top_words) == 0:
            return []
        self.topics = top_words[:n_top_words]
        #print(f"Top words/n-grams for Topic {most_likely_topic}: {top_words}")

        return self.topics


    # Function to get the best topics from a list of topics.
    # Parameters:
    # sentence - the inut sentence, can consist of multiple sentences.
    # num_of_topics - the number of topics to retrieve. Maximum is 10. Default is 7.
    # Return a list of topic words.
    def getBERTopics(self, sentence, n_top_words=NUM_OF_TOPICS):
        topic_threshold = 0.3
        # Preprocess the texts (same as during training)
        #preprocessed_texts = self.preprocess_text(sentence)
        preprocessed_texts = tp.text_preprocessing(sentence)
        preprocessed_texts = self.remove_duplicates(preprocessed_texts)
        n = len(preprocessed_texts)
        if n > 10:
            # Calculate the index for the first percentage of topics to be used
            # Default here is first 30% of topic words
            percent_index = int(n * 0.3)
            # Get the first 10% of the list using slicing
            preprocessed_texts = preprocessed_texts[:percent_index]
        #print("preprocessed_texts = ", preprocessed_texts)

        if n == 0:
            return []

        # Infer topics for the new documents
        topics, probabilities = self.topic_model.transform(preprocessed_texts)
        preprocessed_texts_joined = [' '.join(preprocessed_texts)]

        # Generate embeddings for the input sentence
        embedding_model = self.topic_model.embedding_model.embed
        sentence_embedding = embedding_model(preprocessed_texts_joined)

        # Check if topics exist
        if not topics:
            #print("No topics found in the model.")
            return None, None

        topic_words = []
        similarities_result = {}

        for topic in topics:
            topic_words = self.topic_model.get_topic(topic)
            # Extract the words from the list of tuples
            words_list = [word for word, score in topic_words]
            sent = [' '.join(words_list)]
            ##print(f"Words for topic {topic}: {topic_words}")
            topic_embedding = embedding_model(sent)
            ##print(f"topic sentence {topic} = ", sent)
            similarities = cosine_similarity(sentence_embedding, topic_embedding)
            similarities_result[topic] = similarities


        # Sort the dictionary by the float values in descending order
        sorted_data = sorted(similarities_result.items(), key=lambda x: x[1][0][0], reverse=True)

        # Print the sorted list of tuples
        # for k, v in sorted_data:
        #     print(f"Key: {k}, Value: {v[0][0]}")

        best_key = sorted_data[0][0]
        best_value = sorted_data[0][1][0][0] 
        #print("best_key = ", best_key)
        #print("best_value = ", best_value)

        # if the best topics is less than the threshold, return an empty list
        if best_value < topic_threshold:
            return []
        
        best_topic = self.topic_model.get_topic(best_key)
        #print("best_topic = ", best_topic)
        best_topic = [word for word, score in best_topic]
        self.topics = best_topic[:n_top_words]

        return self.topics


    def generateQuestionsFromTopic(self, topic, category, num_of_questions=NUM_OF_TOPIC_QUESTIONS):        
        prompt = "List " + str(num_of_questions) + " questions that can be generated from the topic \'" + topic + "\' as a Python list that can be assigned to a variable."
        prompt += "The generated questions should be in the context of \'" + category + "\' and targeted to its representative. Limit each question to 10 words."
        #prompt += "Output the questions as a Python list. "
        ##print("Prompt: ", prompt)

        response = self.getResponseForQuestions(prompt)
        #print("RESPONSE 2 = ", response)
        questions = self.extractListFromResponse(response)
        
        for question in questions:
            self.questions_answers[question] = self.getResponseForQuestions(question)

        ##print(questions)
        return questions
    
    def getAnswerFromQuestion(self, question):
        print("IN getAnswerFromQuestion!!!!")
        return self.questions_answers[question]

    def getQuestionAnswerList(self):
        return self.questions_answers

    # Function to get generated questions for each topic.
    # Return a dictionary in the format: 
    # {topic: [questions]}
    def getTopicsAndQuestions(self):
        topicsAndQuestions = {}
        intent = ""
        if len(self.topics) > 0:
            intent = self.topics[0]
            ##print("topics = ", self.topics)
            #self.topics.pop()
            for topic in self.topics:
                if topic not in topicsAndQuestions:
                    topicsAndQuestions[topic] = []  # Initialize an empty list if the key doesn't exist
                topicsAndQuestions[topic] = self.generateQuestionsFromTopic(topic, intent)
            self.topicsAndQuestions = topicsAndQuestions

        return topicsAndQuestions

    def extractListFromResponse(self, text):
        # Extract the list part from the text
        start_index = text.find('[')
        end_index = text.rfind(']') + 1

        # Convert the list string to an actual Python list
        list_string = text[start_index:end_index]
        items = ast.literal_eval(list_string)

        # Now `credit_card_questions` is a Python list
        #for item in items:
        #    #print(item)
        
        return items

    def build_top_model(self, text, delimiter=" "):
        text = text_preprocessing(text)
        sparse_vectorizer = CountVectorizer(strip_accents='unicode')
        sparse_vectors = sparse_vectorizer.fit_transform(text)
        # #print(sparse_vectors.shape)
        # To define number of topics
        n_topics = 1

        # Run LDA to generate topics/clusters
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=1000,
                                        learning_method='online',
                                        random_state=0)

        lda.fit(sparse_vectors)

        # Show the first n_top_words key words
        n_top_words = 10
        feature_names = sparse_vectorizer.get_feature_names()

        t = None
        for i, topic in enumerate(lda.components_):
            t = delimiter.join([feature_names[i]
                               for i in topic.argsort()[:-n_top_words - 1:-1]])

        return t

    # Print the top-n key words
    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
                  for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    def jaccard_similarity(self, x, y):
        """ returns the jaccard similarity between two lists """
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)

    def convert_to_tokens(self, desc: str, delimiter=" "):
        tokens = self.build_top_model(desc, delimiter)

        return tokens

    def lower_casing(self, sentence):
        new_sentence = ''.join([(chr(ord(char) + 32) if ord(char) >
                               64 and ord(char) < 91 else chr(ord(char))) for char in sentence])
        return new_sentence
    
    def add_stopwords(self, nlp):
        stopwords = set()
        with open('./stopwords.txt') as file:  # Ensure the correct file path is used
            stopwords.update([line.strip() for line in file])

        for stopword in stopwords:
            nlp.Defaults.stop_words.add(stopword)

        self.stop_words = nlp.Defaults.stop_words  # Store in self.stop_words for use later
        
        return nlp

    def preprocess_text_2(self, text):
        """Preprocess the input text by removing stopwords, lemmatization, and cleaning."""
        # Lowercase the text
        text = text.lower()
        
        # Remove special characters, numbers, and punctuation
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize and remove stopwords, lemmatize
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.lemma_ not in self.stop_words and len(token.text) > 2]
        
        # Join tokens back into a single string
        return ' '.join(tokens)


    #def preprocess_texts(self, texts):
        """Preprocess a list of texts."""
    #    return [self.preprocess_text(text) for text in texts]
    
    def text_preprocessing(self, raw_sentence, nlp_tool):
        token_sentence = nlp_tool(self.lower_casing(raw_sentence))
        preprocessed_sentence = None

        preprocessed_sentence = [token.lemma_ for token in token_sentence if token.text not in self.stop_words and not token.pos_ ==
                                 'X' and not token.is_punct and not token.is_digit and not token.is_quote]
        #preprocessed_sentence = spell_correction(preprocessed_sentence)

        preprocessed_sentence = " ".join(preprocessed_sentence)
        ##print("processed user sentence: ", preprocessed_sentence)
        return preprocessed_sentence

    def lemmatize_token(self, token):
        """Lemmatize a single token."""
        return self.lemmatizer.lemmatize(token)

    def preprocess_text(self, text):
        """Preprocess the input text by removing stopwords, lemmatization, and cleaning, and generate n-grams."""
        # Lowercase the text
        text = text.lower()

        # Remove special characters, numbers, and punctuation
        text = re.sub(r'[^a-z\s]', '', text)

        # Tokenize and remove stopwords
        tokens = [self.lemmatize_token(word) for word in word_tokenize(text) if word not in self.stop_words and len(word) > 2]

        # Generate 1 to 2 n-grams
        all_ngrams = []
        for n in range(1, 3):
            all_ngrams.extend([' '.join(ngram) for ngram in ngrams(tokens, n)])

        # Remove all repeated n-grams
        unique_ngrams = list(dict.fromkeys(all_ngrams))

        # Join n-grams back into a single string
        return ' '.join(unique_ngrams)

    def preprocess_texts(self, texts):
        """Preprocess a list of texts or a single string of sentences."""
        if isinstance(texts, str):  # If a single string is provided
            texts = [texts]  # Treat the input as a single document
        
        return [self.preprocess_text(text) for text in texts]


    # Function for text preprocessing (including plural to singular conversion)
    """
    def preprocess_text_for_entity(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers, keeping only alphanumeric and spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Convert plural words to singular
        words = text.split()
        singular_words = [p.singular_noun(word) if p.singular_noun(word) else word for word in words]
        
        # Join the singular words back into a string
        text = ' '.join(singular_words)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    """