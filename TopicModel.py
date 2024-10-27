import os
import re
import ast
import spacy
import torch
import joblib
import numpy as np
from collections import defaultdict
from itertools import islice
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
from TextPreprocessing import text_preprocessing
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast
#from huggingface_hub import login

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize the inflect engine
#p = inflect.engine()

# Load the model and tokenizer, and move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hf_token = "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"
#login(token=hf_token)

# Example: Using GPT-2 or any other model from Hugging Face
#model_name = "meta-llama/Llama-3.2-1B"  # Replace with the desired model
#local_model_path = "./models/Llama3.2-1B"
#model = LlamaForCausalLM.from_pretrained(local_model_path).to(device)
#tokenizer = LlamaTokenizerFast.from_pretrained(local_model_path, legacy=False)

# Hyperparameters for TopicModel
SIMILARITY_THRESHOLD = 0.75

NUM_OF_TOPICS = 3

NUM_OF_TOPIC_QUESTIONS = 2

# Load the pre-trained BERT model - not using
# model = SentenceTransformer('all-MiniLM-L6-v2')

# Vector store path
vector_store_path = "./gpt_store3"

# Set the model to use:
# "entity" or "lda" or "bertopic"
model_to_use = "entity"

#nlp = spacy.load('en_core_web_lg')
#stop_words = set(stopwords.words('english'))

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
                "unigrams": {"apply": 2, "credit": 2, "card": 2, "application": 2, "account": 2, "process": 1, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "apply card": 4},
                "trigrams": {"apply credit card": 5, "credit card application": 5, "apply for card": 5, "apply for credit card": 5},
                "patterns": [r"(how|want) to apply credit card", r"process of credit card application", r"apply (for)? (a|the)? credit card",
                             r"how (can|do) (i|someone) apply (for)? (a|the)? credit card", r"i want to apply (for)? (a|the)? credit card"]
            },
            "enquire credit card miles": {
                "unigrams": {"mile": 4, "krisflyer": 4, "credit": 2, "card": 2, "account": 2, "points": 2, "redeem":2, "compensation": 2, "transfer": 2, "convert": 2, "conversion": 2, "exchange": 2, "rate": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "air mile": 4, "krysflyer mile": 4, "krysflyer point": 4, "card mile":4, "mile point": 3, "convert mile": 4, "conversion rate": 2, "exchange mile": 4, "redeem mile": 4, "transfer miles": 4, "foreign transaction": 2},
                "trigrams": {"redeem krisflyer mile": 5, "convert krisflyer mile": 5, "transfer krisflyer mile": 5, "credit card mile": 4, "convert to mile": 5, "mile exchange rate": 5},
                "patterns": [r"(how|want) to (redeem|convert|transfer)? mile", r"(how|want) to (redeem|convert|transfer)? krisflyer (mile|point)",
                             r"enquire (about|regarding)? (credit)? card mile", r"tell me about (credit)? card mile", r"(how|can I) (get|use|redeem) (credit)? card mile"]
            },
            "waive fee": {
                "unigrams": {"waive": 3, "waiver": 3, "credit": 2, "card": 2, "account": 2, "fee":2, "minimum": 2, "payment": 2, "annual": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "grace period": 3, "waive fee": 4, "late fee": 4, "late payment": 4, "interest rate": 3, "annual fee": 2, "annual payment": 2, "minimum payment": 3, "minimum fee": 4},
                "trigrams": {"waive card fee": 5, "waive card payment": 5, "annual interest rate": 3, "waive late fee": 4, "waive late payment": 4, "fee waiver": 4, "credit card fee": 2},
                "patterns": [r"(how|want) to waive card fee", r"(how|want) to waive credit card fee", r"waive (the)? (credit)? card fee", r"can you waive (my|the)? credit card fee",
                             r"(how can i|please) waive (the)? card fee"]
            },
            "credit card rewards": {
                "unigrams": {"rewards": 3, "credit": 2, "card": 2, "points": 2, "redeem":2, "account": 2, "transfer": 2, "convert": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "redeem rewards": 4, "rewards point": 3, "convert point": 4, "transfer point": 4, "foreign transaction": 2},
                "trigrams": {"redeem rewards point": 5, "convert rewards point": 5, "transfer rewards point": 5},
                "patterns": [r"(how|want) to (redeem|convert|transfer)? rewards", r"(how|want) to (redeem|convert|transfer)? rewards points", r"enquire (about|regarding)? (credit)? card rewards",
                             r"(how|where) can i see (credit)? card rewards", r"tell me about (credit)? card rewards"]
            },
            "enquire credit card cashback": {
                "unigrams": {"cashback": 3, "credit": 2, "card": 2, "points": 2, "redeem":2, "amount": 2, "account": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "redeem cashback": 5, "cashback amount": 4, "card cashback": 4, "credit cashback": 4, "about cashback": 5, "foreign transaction": 2},
                "trigrams": {"amount of cashback": 5, "credit card cashback": 5},
                "patterns": [r"(how|where) can i (redeem|get) (credit)? card cashback", r"how much is the cashback", r"tell me (more)? about (credit)? card cashback"]
            },
            "enquire credit card plans": {
                "unigrams": {"plans": 3, "credit": 2, "card": 2, "limit": 2, "interest": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "card plans": 4, "credit plans": 4, "annual fee": 2, "minimum payment": 2, "interest rate": 2, "grace period": 2, "late fee": 2, "foreign transaction": 2},
                "trigrams": {"credit card plans": 5, "amount of cashback": 5, "credit card limit": 5, "foreign transaction fee": 3},
                "patterns": [r"tell me (about|more)? (credit)? card plans", r"enquire (about|regarding)? (credit)? card plans", r"(which|what) (credit)? card plans are available"]
            },
            "compare credit cards": {
                "unigrams": {"compare": 3, "credit": 2, "card": 2, "plans": 2, "limit": 2, "interest": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"credit card": 3, "compare cards": 4},
                "trigrams": {"compare credit card plans": 5, "compare credit cards": 5, "compare cards": 5, "compare plans": 2},
                "patterns": [r"compare (credit)? cards", r"compare (the)? (available|best) (credit)? cards"]
            },
            "enquire property loan": {
                "unigrams": {"apply": 2, "property": 3, "loan": 2, "hdb": 3, "bank": 2, "limit": 2, "mortgage": 3, "interest": 2, "rate": 2, "amortization": 3, "equity": 2, "refinance": 3,
                             "valuation": 2, "private": 2, "account": 2, "instalment": 2, "float": 2, "fix": 2, "bto": 3, "sibor": 3, "cpf": 2, "rate": 2,
                             "duration": 2, "flat": 3, "resale": 3, "migration": 2, "buy": 2, "sell": 2, "transfer": 2, "purchase": 2, "ocbc": 1, "dbs": 1, "uob": 1, "hsbc": 1},
                "bigrams": {"property loan": 3, "mortgage loan": 4, "hdb loan": 4, "bank loan": 3, "interest rate": 2, "loan period": 3, "down payment": 2, "loan tenure": 3,
                            "floating rate": 3, "float rate": 3, "fix rate": 3, "fixed rate": 3, "home loan": 4, "bank account": 2, "resale flat": 3, "loan application": 3,
                            "buy flat": 3, "sell flat": 3, "buy house": 3, "housing loan": 3, "bank account": 2, "purchase flat": 3, "private property": 3},
                "trigrams": {"enquire property loan": 5, "enquire home loan": 5, "enquire mortgage loan": 5, "float interest rate": 4, "fix interest rate": 4, "buy resale flat": 4,
                             "buy resale house": 4, "apply property loan": 5, "apply home loan": 5, "apply mortgage loan": 5,},
                "patterns": [r"enquire (about|regarding)? (a|the)? property loan", r"tell me (about|more about) (a|the)? property loan",
                             r"apply (for)? (a|the)? property loan", r"how (can|do) (i|someone) apply (for)? (a|the)? property loan",
                             r"i want to apply (for)? (a|the)? property loan"]
            },
            "claim travel insurance": {
                "unigrams": {"enquire": 2, "travel": 3, "insurance": 2, "coverage": 3, "medical": 2, "limit": 2, "policy": 2, "trip": 2, "accident": 2, "compensation": 2, "emergency": 2, "terrorism": 3, "belonging": 2, "assistance": 3,
                             "premium": 2, "private": 2, "region": 2, "luggage": 2, "baggage": 2, "delay": 2, "flight": 3, "claim": 3, "process": 2, "sick": 2,
                             "duration": 2, "policy": 2, "cancel": 2, "buy": 2, "process": 2, "transfer": 2, "purchase": 2, "geographical": 2, "covid-19": 3, "group": 1, "personal": 1},
                "bigrams": {"travel insurance": 4, "insurance coverage": 4, "insurance premium": 4, "medical expense": 3, "policy number": 3, "coverage amount": 2, "regional coverage": 2, "flight delay": 2, "flight cancel": 3,
                            "flight postpone": 3, "luggage delay": 4, "baggage delay": 4, "luggage loss": 4, "lost luggage": 4, "baggage loss": 4, "lost baggage": 4, "lost belonging": 4, "personal belonging": 2, "trip cancel": 3, "personal accident": 3,
                            "policy number": 2, "car rental": 2, "claim process": 2, "geographical coverage": 3, "trip cancellation": 2, "flight cancellation": 3, "emergency evacuation": 3},
                "trigrams": {"apply travel insurance": 5, "how to claim": 3, "travel insurance enquiry": 5, "enquire travel insurance": 4, "buy travel insurance": 4, "cancel travel insurance": 4,
                             "purchase travel insurance": 3, "cannot find luggage": 4, "cannot find my luggage": 4, "cannot find baggage": 4, "cannot find my baggage": 4, "lost my luggage": 4, "lost my baggage": 4},
                "patterns": [r"enquire (about|regarding)? (a|the)? travel insurance", r"tell me (about|more about) (a|the)? travel insurance",
                             r"buy (a|the)? travel insurance", r"how (can|do) (i|someone) buy (a|the)? travel insurance", r"i want to buy (a|the)? travel insurance"]
            }
        }

    def __init__(self, model=model_to_use):
        self.nlp = spacy.load('en_core_web_lg')
        self.nlp = self.add_stopwords(self.nlp)
        self.stop_words = self.nlp.Defaults.stop_words
        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.init_open_ai()
        if (model == "lda"):
            self.load_lda()
        elif (model == "bertopic"):
            self.load_bertopic()
        self.topics = []
        self.topicsAndQuestions = {}
        self.text_history = ""

    def init_open_ai(self):
        os.environ['OPENAI_API_KEY'] = "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"
        self.model_id = "ft:gpt-4o-mini-2024-07-18:personal::A0l6mkLn"

        self.client = OpenAI(
            # This is the default and can be omitted
            api_key = os.environ.get("OPENAI_API_KEY"),
        )

    # Function to detect acronyms, including ones with periods like "U.O.B."
    def is_acronym(word):
        return bool(re.match(r'([A-Z]\.){2,}|[A-Z]{2,}', word))

    def get_openAI_responses(self, prompt):
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


    def gen_response_for_questions_w_RAG(self, input_text):
        """
        Function to get response from RAG based on the question or prompt.
        Parameters:
        input_text - the question or prompt
        Return a text response.
        """
        # Get the retrieved context from the index
        context = self.retrieve_RAG_context(input_text)
        # print("CONTEXT = ", context)

        # Integrate the context with the input text for the generative model
        prompt = f"Context: {context}\n\nQuestion: {input_text}\n\nAnswer:"

        # Call the OpenAI API to generate a response using GPT-3.5
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",  # or the appropriate model name
            messages=[
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": "You are a customer service representative for financial information across banks and "
                            "insurance companies in Singapore. Summarise your response within 50 words."
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
            max_tokens=80,
            n=1,
            stop=None,
            temperature=0.7  # configurable
        )

        # Extract the text from the completion
        generated_text = completion.choices[0].message.content.strip()

        return generated_text
    
    # Function to load the index from storage and create a retriever
    def load_and_create_retriever(self, persist_dir):
        # Load the index from the persisted storage directory
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        
        # Convert the index into a retriever
        # retriever = index.as_retriever(similarity_top_k=1)

        retriever = index.as_query_engine(
                        response_mode="tree_summarize",
                        verbose=True,
                    )

        return retriever

    # Function to load the index from storage
    def load_gpt_index(self, storage_path):
        # Load the pre-built index from storage path
        index = load_index_from_storage(storage_path)

        return index


    def retrieve_RAG_context(self, prompt):
        # If not already done, initialize ‘index’ and ‘query_engine’
        if not hasattr(self, "index"):
            # rebuild storage context and load index
            storage_context = StorageContext.from_defaults(persist_dir=vector_store_path)
            self.index = load_index_from_storage(storage_context=storage_context, index_id="vector_index")
            self.query_engine = self.index.as_query_engine()

        # Submit query
        response = self.query_engine.query(prompt)

        return response.response

    
    def remove_duplicates(self, word_list):
        seen = set()
        result = []
        
        for word in word_list:
            if word not in seen:
                seen.add(word)
                result.append(word)
        
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
    

    def find_topics(self, sentence, n_top_words=NUM_OF_TOPICS, model=model_to_use):
        if model == "bertopic":
            return self.get_bert_topic(sentence, n_top_words=n_top_words)
        elif model == "lda":
            return self.get_lda_topic(sentence, n_top_words=n_top_words)
        elif model == "entity":
            return self.get_entity_topic(sentence, n_top_words=n_top_words)
        
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

    # Function to generate n-grams from input
    def get_ngrams(self, user_input, n):
        tokens = user_input.split()
        return zip(*[islice(tokens, i, None) for i in range(n)])

    def recognize_intent(self, user_input):
        user_input = user_input.lower()
        
        # Initialize score and keyword dictionary for each intent
        intent_scores = defaultdict(int)
        
        # Separate sets for trigrams, bigrams, and unigrams to avoid duplicates
        trigram_keywords = set()
        bigram_keywords = set()
        unigram_keywords = set()

        # Generate unigrams, bigrams, and trigrams from user input
        unigrams = user_input.split()
        bigrams = [' '.join(bigram) for bigram in self.get_ngrams(user_input, 2)]
        trigrams = [' '.join(trigram) for trigram in self.get_ngrams(user_input, 3)]

        # Match unigrams, bigrams, and trigrams with weighted keywords
        for intent, data in self.intents.items():
            for trigram in trigrams:
                if trigram in data["trigrams"]:
                    intent_scores[intent] += data["trigrams"][trigram]
                    trigram_keywords.add(trigram)
            
            for bigram in bigrams:
                if bigram in data["bigrams"]:
                    intent_scores[intent] += data["bigrams"][bigram]
                    bigram_keywords.add(bigram)

            for unigram in unigrams:
                if unigram in data["unigrams"]:
                    intent_scores[intent] += data["unigrams"][unigram]
                    unigram_keywords.add(unigram)

            # Check patterns using regular expressions
            for pattern in data["patterns"]:
                if re.search(pattern, user_input):
                    intent_scores[intent] += 5  # Add pattern match boost
                    trigram_keywords.add(f"Pattern match: {pattern}")  # Add pattern as trigram for sequence priority

        # Identify the intent with the highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            
            # Combine keywords in the order: trigrams, bigrams, unigrams
            combined_keywords = list(trigram_keywords) + list(bigram_keywords) + list(unigram_keywords)

            # Remove unigrams that are part of a bigram or trigram
            for unigram in list(unigram_keywords):  # Iterate over the unigrams
                for phrase in bigram_keywords.union(trigram_keywords):  # Check both bigrams and trigrams
                    if unigram in phrase.split():  # If unigram is part of a bigram or trigram
                        combined_keywords.remove(unigram)  # Remove the unigram

            return best_intent, combined_keywords
        
        return "unknown_intent", []


    def get_entity_topic(self, text, n_top_words=NUM_OF_TOPICS):
        prompt = "Rephrase the following text into formal and concise language: " + text        
        text = self.get_openAI_responses(prompt)

        # add into text_history and pop out previous one
        self.text_history += text
        current_words = self.text_history.split()
        if len(current_words) > 100:
            current_words = current_words[-100:]
            self.text_history = " ".join(current_words)

        # Preprocess the text
        text = self.preprocess_text(self.text_history)

        intent, words = self.recognize_intent(text)
        print("INTENT = ", intent)
        print("WORDS = ", words)
        # record all Topics in history
        self.topics = ([word for word in words if word not in intent] +
                       [topic for topic in self.topics if topic not in words])
        if intent != "unknown_intent":
            self.topics.insert(0, intent)

        return self.topics[:n_top_words]
    

    def get_lda_topic(self, sentence, n_top_words=NUM_OF_TOPICS):
        """
        LDA method of Topic, not used in real-time but used in summary
        """
        # Step 1: Preprocess the entire document as one unit
        preprocessed_doc = self.preprocess_text_LDA(sentence)
        #preprocessed_texts = tp.text_preprocessing(sentence)
        #preprocessed_texts = self.remove_duplicates(preprocessed_texts)
        #preprocessed_texts_joined = [' '.join(preprocessed_texts)]
        #print("Preprocessed text = ", preprocessed_doc)

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


    def get_bert_topic(self, sentence, n_top_words=NUM_OF_TOPICS):
        """
        # Function to get the best topics from a list of topics.
        # Parameters:
        # sentence - the input sentence, can consist of multiple sentences.
        # num_of_topics - the number of topics to retrieve. Maximum is 10. Default is 7.
        # Return a list of topic words.
        """
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
        prompt = ("We are on the call with client where he understand credit card choice to apply for, specific on \'" + topic + "\'. Client saying that \'" +
                  self.text_history + "\' ")
        prompt += ("List " + str(num_of_questions) +
                   " follow-up response that can be helpful: one is a question to clarify the information with client" +
                   " and the other is description on key information agent need to refer to reply " +
                    "customer requirement as a python string list, "
                    "e.g. ['coverage of the insurance policy','May I know your policy number please?']. ")
        prompt += ("The generated questions should be in the context of \'" + category +
                   "\' and targeted to its representative. Limit each one to 10 words.")
        print(f"Prompt input: {topic} - {category} - {self.text_history}")

        response = self.gen_response_for_questions_w_RAG(prompt)
        questions = self.extractListFromResponse(response)
        
        # for question in questions:
        #     self.questions_answers[question] = self.getResponseForQuestions(question)

        ##print(questions)
        return questions
    
    def getAnswerFromQuestion(self, question):
        print("IN getAnswerFromQuestion!!!!")
        return self.questions_answers[question]

    def getQuestionAnswerList(self):
        return self.questions_answers


    def getTopicsAndQuestions(self):
        """
        Function to get generated questions for each topic.
        Return a dictionary in the format:
        {topic: [questions]}
        """
        topicsAndQuestions = {}
        category = ""
        if len(self.topics) > 0:
            category = self.topics[0]
            ##print("topics = ", self.topics)
            for topic in self.topics[:NUM_OF_TOPICS]:
                if topic not in self.topicsAndQuestions:
                    # topicsAndQuestions[topic] = []  # Initialize an empty list if the key doesn't exist
                    topicsAndQuestions[topic] = self.generateQuestionsFromTopic(topic, category)
                    self.topicsAndQuestions[topic] = topicsAndQuestions[topic]
                else:
                    topicsAndQuestions[topic] = self.topicsAndQuestions[topic]
            # self.topicsAndQuestions = topicsAndQuestions
        return topicsAndQuestions

    def extractListFromResponse(self, text):
        # Extract the list part from the text
        start_index = text.find('[')
        end_index = text.rfind(']') + 1

        # Convert the list string to an actual Python list
        list_string = text[start_index:end_index]
        try:
            items = ast.literal_eval(list_string)
            # Ensure the result is actually a list
            if not isinstance(items, list):
                raise ValueError("Extracted content is not a list")
        except (ValueError, SyntaxError) as e:
            print(f"Error converting to list: {e}")
            return []
        
        return items

    
    def add_stopwords(self, nlp):
        stopwords = set()
        with open('./stopwords.txt') as file:  # Ensure the correct file path is used
            stopwords.update([line.strip() for line in file])

        for stopword in stopwords:
            nlp.Defaults.stop_words.add(stopword)

        self.stop_words = nlp.Defaults.stop_words  # Store in self.stop_words for use later
        
        return nlp

    def preprocess_text(self, text):
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

    def lemmatize_token(self, token):
        """Lemmatize a single token."""
        return self.lemmatizer.lemmatize(token)

    def preprocess_text_LDA(self, text):
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

