# new version

from flask import Flask, request, jsonify
from flask_cors import CORS
from azure.storage.blob import BlobServiceClient
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import uuid
import nltk
import re
from nltk.corpus import stopwords
import numpy as np
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
import spacy
# try:
#     importlib.import_module("en_core_web_sm")
# except ImportError:
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
import ssl
import certifi
from werkzeug.middleware.proxy_fix import ProxyFix

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab') # Ensure punkt_tab is downloaded
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# import subprocess
# import importlib



load_dotenv()

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB_NAME")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION_NAME")

# Setting up clients
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
mongo_client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=5000
    )
db = mongo_client[MONGO_DB]
collection = db[MONGO_COLLECTION]

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173","https://agreeable-meadow-00be2840f.2.azurestaticapps.net"], "methods": ["GET", "POST"], "allow_headers": ["Content-Type"]}})
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


# API routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "azure": bool(AZURE_CONNECTION_STRING),
        "mongo": bool(MONGO_URI)
    }), 200

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

custom_stopwords = {
    # Resume sections & structure
    'objective', 'summary', 'experience', 'skills', 'projects', 'education',
    'certification', 'declaration', 'references', 'details', 'background',

    # Contact info & generic nouns
    'email', 'mobile', 'phone', 'contact', 'address', 'name', 'dob', 'date', 'place',

    # Filler adjectives/adverbs
    'hardworking', 'motivated', 'dedicated', 'passionate', 'responsible',
    'effective', 'proactive', 'positive', 'efficient', 'dynamic', 'self', 'highly',

    # Generic verbs
    'worked', 'working', 'handled', 'managed', 'involved', 'participated',
    'performed', 'developed', 'gained', 'helped',

    # Generic nouns
    'team', 'organization', 'company', 'role', 'project', 'environment',
    'tasks', 'opportunity', 'career', 'knowledge', 'ability', 'goal', 'objective',

    # Others
    'curriculum', 'vitae', 'resume', 'respectively', 'currently', 'present',
    'etc', 'year', 'month', 'years', 'months', 'location', 'period',

    # Overused buzzwords
    'professional', 'profile', 'academic', 'strong', 'excellent', 'communication',
    'interpersonal', 'leadership', 'skills', 'expertise', 'domain', 'field'
}
stop_words.update(custom_stopwords)

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    doc = nlp(text)

    # Get unwanted tokens based on entity labels
    unwanted_labels = {'PERSON', 'GPE', 'ORG', 'DATE', 'TIME'}
    unwanted_tokens = set()
    for ent in doc.ents:
        if ent.label_ in unwanted_labels:
            text = text.replace(ent.text, '')

    text = text.lower()

    # Remove emails and phone numbers using regex
    text = re.sub(r'\b[\w\.-]+?@\w+?\.\w+?\b', '', text)
    text = re.sub(r'\+?\d[\d\-\.\s()]{8,}\d', '', text)

    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Tokenise
    tokens = word_tokenize(text)

    # Remove stopwords and unwanted NER tokens
    tokens = [word for word in tokens if word not in stop_words and word not in unwanted_tokens]

    #chunks = get_chunks(text)


    # POS tagging and lemmatisation
    pos_tags = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]

    # Join back to string
    clean_text = ' '.join(tokens).strip()

    return clean_text

# Job Description Upload taking place here
@app.route("/upload_job_description", methods=['POST'])
def upload_job_description():
    file = request.files.get('pdf')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    if file and file.filename.endswith('.pdf'):
        unique_name = f"jd_{uuid.uuid4()}_{file.filename}"
        blob_client = container_client.get_blob_client(unique_name)
        blob_client.upload_blob(file, overwrite=True)
        blob_url = blob_client.url

        file.seek(0)
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])

        collection.insert_one({
            "type": "job_description",
            "filename": file.filename,
            "blob_url": blob_url,
            "text_excerpt": text
        })

        return jsonify({"message": "Job description uploaded", "url": blob_url}), 200

    return jsonify({"error": "Invalid file type"}), 400


# candidate Resumes Upload taking place here
@app.route("/upload_candidate_resumes", methods=["POST"])
def upload_to_blob_and_mongo():
    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    uploaded_files = []

    for file in files:
        if file.filename.endswith(".pdf"):
            unique_name = f"{uuid.uuid4()}_{file.filename}"
            blob_client = container_client.get_blob_client(unique_name)
            blob_client.upload_blob(file, overwrite=True)
            blob_url = blob_client.url

            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = "".join([page.get_text() for page in doc])

            collection.insert_one({
                "type": "resume",
                "filename": file.filename,
                "blob_url": blob_url,
                "text_excerpt": text
            })

            uploaded_files.append({"filename": file.filename, "url": blob_url})

    return jsonify({"message": f"{len(uploaded_files)} files uploaded", "files": uploaded_files}), 200


#! Compute TF
def compute_tf(doc_tokens, vocab_index):
    tf = np.zeros(len(vocab_index))
    for word in doc_tokens:
        if word in vocab_index:
            tf[vocab_index[word]] += 1
    return tf / len(doc_tokens)

#! Compute IDF
def compute_idf(docs, vocab_index):
    N = len(docs)
    df = np.zeros(len(vocab_index))

    for doc in docs:
        unique_words = set(doc)
        for word in unique_words:
            if word in vocab_index:
                df[vocab_index[word]] += 1

    idf = np.log((N + 1) / (df + 1)) + 1
    return idf

#! Compute cosine Similarity
def cosine_sim(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0

# Here we calculate the ranks of resumes based on the job description
@app.route("/calculate_ranks", methods=["GET"])
def calculate_resume_rank():
    try:
        jd_doc = collection.find_one({"type": "job_description"}, sort=[("_id", -1)])
        if not jd_doc:
            return jsonify({"error": "No job description found in DB"}), 404

        resumes = list(collection.find({"type": "resume"}))
        if not resumes:
            return jsonify({"error": "No resumes found in DB"}), 404

        cleaned_job_desc = clean_text(jd_doc['text_excerpt'])
        cleaned_resumes = [(res["filename"], clean_text(res["text_excerpt"])) for res in resumes]

        documents = [cleaned_job_desc] + [txt for _, txt in cleaned_resumes]


        # Tokenize the documents
        tokenized_docs = [doc.split() for doc in documents]

        # Create sorted vocabulary which will be used in tf-idf processing
        vocab = sorted(set(word for doc in tokenized_docs for word in doc))

        #filtering the vocab to improve the working
        N = len(tokenized_docs)
        ### Step 1: Compute document frequency (DF) of all words
        df_counter = defaultdict(int)
        for doc in tokenized_docs:
            unique_words = set(doc)
            for word in unique_words:
                df_counter[word] += 1

        ### Step 2: Set min_df and max_df thresholds
        min_df = 5             # appears in at least 5 documents
        max_df_ratio = 0.85    # appears in at most 85% of documents
        max_df = max_df_ratio * N

        ### Step 3: Filter the vocabulary based on DF thresholds
        filtered_vocab = sorted([
            word for word, df in df_counter.items()
            if min_df <= df <= max_df
        ])

        ### Step 4: Create vocab index
        vocab_index = {word: idx for idx, word in enumerate(filtered_vocab)}
        #vocab_index = {word: idx for idx, word in enumerate(vocab)}

        tf_matrix = np.array([compute_tf(doc, vocab_index) for doc in tokenized_docs])
        idf = compute_idf(tokenized_docs, vocab_index)

        tfidf_matrix = tf_matrix * idf

        
        #fine-tuning the parameters and using the optimal value
        lsa = TruncatedSVD(n_components=25, n_iter=100, random_state=42)
        lsa.fit(tfidf_matrix)

        print("Components shape:", lsa.components_.shape)
        print(f"Total variance preserved: {lsa.explained_variance_ratio_.sum():.2%}")

        lsa_matrix = lsa.transform(tfidf_matrix)

        jd_vector = lsa_matrix[0]
        resume_vectors = lsa_matrix[1:]

        # Calculate cosine similarities
        scores = [cosine_sim(jd_vector, vec) for vec in resume_vectors]

        results = [
            {"filename": fname, "score": float(score*100)} # Convert to percentage
            for (fname, _), score in zip(cleaned_resumes, scores)
        ]

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    

# one endpoint to delete all documents from both.
@app.route("/clear_all_data", methods=["POST"])
def clear_all_data():
    try:
        print("Starting data cleanup...")
        
        # Clear Azure Blob Storage
        print("Clearing Azure Blob Storage...")
        blobs_deleted = 0
        try:
            blobs_list = container_client.list_blobs()
            for blob in blobs_list:
                try:
                    container_client.delete_blob(blob.name)
                    blobs_deleted += 1
                except Exception as blob_error:
                    print(f"Error deleting blob {blob.name}: {str(blob_error)}")
                    continue
        except Exception as storage_error:
            print(f"Error accessing blob storage: {str(storage_error)}")

        # Clear MongoDB Collection
        print("Clearing MongoDB Collection...")
        mongo_result = collection.delete_many({})
        docs_deleted = mongo_result.deleted_count

        return jsonify({
            "message": "Cleanup completed",
            "details": {
                "blobs_deleted": blobs_deleted,
                "documents_deleted": docs_deleted
            }
        }), 200

    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        return jsonify({
            "error": "Failed to clear data",
            "details": str(e)
        }), 500
 

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, ssl_context=None)





































