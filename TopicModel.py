import os
import re
import ast
import spacy
import joblib
import numpy as np
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

# Download NLTK stopwords
nltk.download('stopwords')

# Initialize the inflect engine
#p = inflect.engine()

# Hyperparameters for TopicModel
similarityThreshold = 0.75

# Load the pre-trained BERT model
#model = SentenceTransformer('all-MiniLM-L6-v2')

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

    def __init__(self, model=model_to_use):
        self.nlp = spacy.load('en_core_web_lg')
        self.nlp = self.add_stopwords(self.nlp)
        self.stop_words = self.nlp.Defaults.stop_words
        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.initOpenAI()
        if (model == "lda"):
            self.load_lda()
        elif (model == "bertopic"):
            self.load_bertopic()
        self.topics = []
        self.topicsAndQuestions = {}

    def initOpenAI(self):
        #os.environ['OPENAI_API_KEY'] = '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'
        #os.environ['OPENAI_API_KEY'] = 'sk-proj-NNn2ogDOcAahK91dPqzAT3BlbkFJoUPvh8YGkkCkzlOtx4sL'
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
                    "text": "You are a customer service representative for financial analysis across banks in Singapore. Summarise your response within 256 words."
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
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "text"
            }
        )

        return response.choices[0].message.content
    
    # Function to get response from RAG based on the question or prompt.
    # Parameters:
    # input_text - the question or prompt
    # Return a text response.
    def getResponseForQuestions(self, input_text):
        # Get the retrieved context from the index
        context = self.generate_response(input_text)

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
                    "text": "You are a customer service representative for financial analysis across banks in Singapore. Summarise your response within 256 words."
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
            max_tokens=256,
            n=1,
            stop=None,
            temperature=1
        )

        # Extract the text from the completion
        #generated_text = completion.choices[0].message.content.strip()
        generated_text = completion.choices[0].message.content.strip()
        #generated_text = completion.choices

        return generated_text
    
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
    

    def getTopics(self, sentence, n_top_words=7, model=model_to_use):
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


    def getEntityTopic(self, text, n_top_words=10):
        custom_keywords = { "credit card": ["credit card", "card", "reward", "waive", "credit limit", "payment", "annual fee", "interest rate", "cashback",
                                            "reward point", "minimum payment", "late fee", "grace period", "foreign transaction fee", "transaction fee", "penalty",
                                            "late payment", "minimum fee", "limit", "debt", "owe", "uob", "ocbc", "dbs", "air miles", "air mile", "krisflyer", "redemption"],
                            "property loan": ["mortgage", "principal", "interest rate", "loan tenure", "down payment", "amortization", "equity", "fixed rate",
                                              "floating rate", "refinance", "stamp duty", "valuation", "loan agreement", "agreement", "legal fee", "loan", "tenure",
                                              "property loan", "hdb", "private property", "private", "home loan", "migrate", "cpf", "bank account", "bank loan",
                                              "value loss", "loss", "housing lone", "buy", "sell", "loan period", "instalment", "prepayment penalty", "foreclosure",
                                              "loan tenure", "resale flat", "bto", "loan application", "bank", "fix rate", "float rate", "transfer", "purchase",
                                              "application", "migration", "duration", "floating interest rate", "fixed interest rate", "sibor", "installment",
                                              "float interest rate", "fix interest rate"],
                            "travel insurance": ["travel insurance", "coverage", "medical expense", "trip cancellation", "trip cancel", "baggage loss", "baggage delay",
                                                 "personal accident", "accident", "pre-existing condition", "emergency", "emergency evacuation", "travel delay", "delay",
                                                 "loss of personal belonging", "loss", "belonging", "trip postpone", "assistance", "policy excess", "rental car excess", 
                                                 "terrorism", "covid-19", "repatriation", "premium", "insurance premium", "geographical coverage", "claim process", 
                                                 "regional coverage", "luggage loss", "luggage delay", "luggage", "baggage", "sick", "medical", "claim", "policy number"] }
        
        prompt = "Rephrase the following text into formal and concise language: " + text        
        text = self.getOpenAIResponses(prompt)

        # Preprocess the text
        text = self.preprocess_text_2(text)
        #text = self.text_preprocessing(text, self.nlp)
        #print("Preprocessed text = ", text)

        # Create a list to store the found keywords
        found_keywords = []
        self.topics = []
        found_category = None

        # Iterate over each category and its associated keywords
        for category, keywords in custom_keywords.items():
            for keyword in keywords:
                # Use regex to find whole words or phrases in the preprocessed text
                # re.escape is used to handle special characters in keywords
                if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                    found_category = category
                    #print("category = ", category)
                    if keyword not in self.topics:
                        found_keywords.append(keyword)
                        self.topics.append(keyword)
        
        # Sort found_keywords and self.topics based on the number of words (n-grams) in descending order
        found_keywords = sorted(found_keywords, key=lambda k: len(k.split()), reverse=True)
        self.topics = sorted(self.topics, key=lambda k: len(k.split()), reverse=True)

        if found_category != None:
            found_keywords.insert(0, found_category)
            self.topics.insert(0, found_category)

        #print("found_keywords = ", found_keywords[:n_top_words])
        
        #self.topics = found_category
        #print("[Topic] topics = ", self.topics)
        return self.topics[:n_top_words]
    

    def getLDATopics(self, sentence, n_top_words=7):
        # Step 1: Preprocess the entire document as one unit
        preprocessed_doc = self.preprocess_text(sentence)
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


    # Function to get the best topics from a list of topics.
    # Parameters:
    # sentence - the inut sentence, can consist of multiple sentences.
    # num_of_topics - the number of topics to retrieve. Maximum is 10. Default is 7.
    # Return a list of topic words.
    def getBERTopics(self, sentence, n_top_words=7):
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


    def generateQuestionsFromTopic(self, topic, category, num_of_questions=5):        
        prompt = "List " + str(num_of_questions) + " questions that can be generated from the topic \'" + topic + "\' as a Python list that can be assigned to a variable."
        prompt += "The generated questions should be in the context of \'" + category + "\' and targeted to its representative."
        #prompt += "Output the questions as a Python list. "
        ##print("Prompt: ", prompt)

        response = self.getResponseForQuestions(prompt)
        questions = self.extractListFromResponse(response)

        ##print(questions)
        return questions

    # Function to get generated questions for each topic.
    # Return a dictionary in the format: 
    # {topic: [questions]}
    def getTopicsAndQuestions(self):
        topicsAndQuestions = {}
        category = ""
        if len(self.topics) > 0:
            category = self.topics[0]
            ##print("topics = ", self.topics)
            for topic in self.topics:
                if topic not in topicsAndQuestions:
                    topicsAndQuestions[topic] = []  # Initialize an empty list if the key doesn't exist
                topicsAndQuestions[topic] = self.generateQuestionsFromTopic(topic, category)
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