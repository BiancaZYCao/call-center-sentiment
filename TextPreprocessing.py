import nltk
import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from itertools import chain

# Download the stopword list if not already downloaded
nltk.download('stopwords')

# Get the default NLTK stopwords for English
stop_words = set(stopwords.words('english'))

# 1. Lower casing
def lower_casing(sentence):
    new_sentence = sentence.lower()
    return new_sentence

# 2. Punctuation removal
def punctuation_removal(sentence):
    new_sentence = re.sub(r',|!|\?|\"|<|>|\(|\)|\[|\]|\{|\}|@|#|\+|\=|\-|\_|~|\&|\*|\^|%|\||\$|/|`|\.', '', sentence, count=0, flags=0)
    return new_sentence

# 3. Expand the abbreviation
def expand_abbriviation(sentence):
    replacement_patterns = [
        (r'won\'t', 'will not'),
        (r'can\'t', 'cannot'),
        (r'i\'m', 'i am'),
        (r'ain\'t', 'is not'),
        (r'(\w+)\'ll', '\g<1> will'),
        (r'(\w+)n\'t', '\g<1> not'),
        (r'(\w+)\'ve', '\g<1> have'),
        (r'(\w+)\'s', '\g<1> is'),
        (r'(\w+)\'re', '\g<1> are'),
        (r'(\w+)\'d', '\g<1> would')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in replacement_patterns]

    new_sentence = sentence
    for (pattern, repl) in patterns:
        new_sentence, _ = re.subn(pattern, repl, new_sentence)
    return new_sentence

# 4. Tokenize the sentence
def tokenization(sentence):
    return nltk.word_tokenize(sentence)

# 5. Remove the stopwords
def stopword_removal(sentence):    
    stopwords = add_stopwords()  # Get the updated stopword list
    new_sentence = [word for word in sentence if word.lower() not in stopwords]  # Compare in lowercase
    return new_sentence

# 6. Lemmatization
def get_wordnet_pos(word):
    pack = nltk.pos_tag([word])
    tag = pack[0][1]
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatization(sentence):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    new_sentence = [lemmatizer.lemmatize(word, get_wordnet_pos(word) or wordnet.NOUN) for word in sentence]
    return new_sentence

# Generate n-grams
def generate_ngrams(tokens, n=3):
    """Generate n-grams from a list of tokens."""
    n_grams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in n_grams]

# Combine unigrams, bigrams, and trigrams
def combine_ngrams(tokens):
    unigrams = tokens
    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)
    return list(chain(unigrams, bigrams, trigrams))

# Add custom stopwords from a file
def add_stopwords():
    stopwords_ = set()
    with open('./stopwords.txt') as file:  # Adjust path as necessary
        stopwords_.update([stopword.strip().lower() for stopword in file.readlines()])
    stop_words.update(stopwords_)  # Add custom stopwords to the existing set
    return stop_words

# Final preprocessing function that combines everything
def text_preprocessing(raw_sentence):
    sentence = lower_casing(raw_sentence)
    sentence = punctuation_removal(sentence)
    sentence = expand_abbriviation(sentence)
    sentence = tokenization(sentence)
    sentence = stopword_removal(sentence)
    sentence = lemmatization(sentence)
    sentence = combine_ngrams(sentence)  # Generate n-grams
    return sentence
