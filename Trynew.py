#%%
# Import necessary libraries
import pandas as pd
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
#%%
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

#%%
# Load the dataset
with ZipFile('/home/ubuntu/NLP_Data/train.csv.zip', 'r') as zip_file:
    with zip_file.open('train.csv') as file:
        df = pd.read_csv(file)

# Display dataset information
print(df.info())
print(df.head())

# Drop rows with null values
df = df.dropna()

#%%
# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['question1'] = df['question1'].apply(clean_text)
df['question2'] = df['question2'].apply(clean_text)

print("Cleaned Text:\n", df[['question1', 'question2']].head())

#%%
# Feature Engineering
df['q1_len'] = df['question1'].apply(len)
df['q2_len'] = df['question2'].apply(len)
df['len_diff'] = abs(df['q1_len'] - df['q2_len'])

def word_overlap(q1, q2):
    q1_words = set(q1.split())
    q2_words = set(q2.split())
    return len(q1_words.intersection(q2_words))

def common_word_ratio(q1, q2):
    q1_words = set(q1.split())
    q2_words = set(q2.split())
    union_len = len(q1_words.union(q2_words))
    return len(q1_words.intersection(q2_words)) / union_len if union_len > 0 else 0

df['word_overlap'] = df.apply(lambda row: word_overlap(row['question1'], row['question2']), axis=1)
df['common_word_ratio'] = df.apply(lambda row: common_word_ratio(row['question1'], row['question2']), axis=1)

# Additional Features: Named Entity Overlap
import spacy
nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

def named_entity_overlap(q1, q2):
    q1_entities = {ent.text for ent in nlp(q1).ents}
    q2_entities = {ent.text for ent in nlp(q2).ents}
    return len(q1_entities.intersection(q2_entities))

df['named_entity_overlap'] = df.apply(lambda row: named_entity_overlap(row['question1'], row['question2']), axis=1)
#%%
df.head()
#%%
# Additional Features: N-gram Overlap
from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm

def ngram_overlap_batch(questions1, questions2, n=2):
    """
    Calculate n-gram overlap for a batch of questions.
    """
    # Combine all questions into a single list
    combined_questions = questions1 + questions2

    # Fit CountVectorizer on the combined batch
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words="english")
    try:
        vectorizer.fit(combined_questions)
    except ValueError:
        # Handle empty vocabulary
        return [0] * len(questions1)

    # Transform each question into n-grams
    q1_ngrams = [set(vectorizer.transform([q]).nonzero()[1]) for q in questions1]
    q2_ngrams = [set(vectorizer.transform([q]).nonzero()[1]) for q in questions2]

    # Compute overlaps
    overlaps = []
    for ngrams1, ngrams2 in zip(q1_ngrams, q2_ngrams):
        if not ngrams1 or not ngrams2:
            overlaps.append(0)
        else:
            overlaps.append(len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2)))
    return overlaps

def process_batches_in_parallel(df, batch_size=1000, n=2, num_threads=8):
    """
    Process DataFrame batches in parallel for n-gram overlap.
    """
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size > 0 else 0)
    batches = [df.iloc[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    def process_batch(batch):
        return ngram_overlap_batch(batch['question1'].tolist(), batch['question2'].tolist(), n=n)

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for result in tqdm(executor.map(process_batch, batches), total=len(batches), desc=f"Processing {n}-grams"):
            results.extend(result)
    return results
#%%
# Process bigrams
df['bigram_overlap'] = process_batches_in_parallel(df, batch_size=5000, n=2)
df['trigram_overlap'] = process_batches_in_parallel(df, batch_size=5000, n=3)

print(df[['bigram_overlap', 'trigram_overlap']].head())


#%%
# Visualization: Lengths, Overlaps, and Similarity
plt.figure(figsize=(10, 6))
plt.hist(df['q1_len'], bins=50, alpha=0.5, label='Question 1 Length')
plt.hist(df['q2_len'], bins=50, alpha=0.5, label='Question 2 Length')
plt.title('Distribution of Question Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['is_duplicate'], y=df['len_diff'])
plt.title('Length Difference vs Is Duplicate')
plt.xlabel('Is Duplicate')
plt.ylabel('Length Difference')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['is_duplicate'], y=df['word_overlap'])
plt.title('Word Overlap vs Is Duplicate')
plt.xlabel('Is Duplicate')
plt.ylabel('Word Overlap')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['is_duplicate'], y=df['named_entity_overlap'])
plt.title('Named Entity Overlap vs Is Duplicate')
plt.xlabel('Is Duplicate')
plt.ylabel('Named Entity Overlap')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['is_duplicate'], y=df['bigram_overlap'])
plt.title('Bigram Overlap vs Is Duplicate')
plt.xlabel('Is Duplicate')
plt.ylabel('Bigram Overlap')
plt.show()

#%%
# Transformer-based embeddings
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def batch_embeddings(texts, tokenizer, model, device, batch_size=128):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.vstack(embeddings)

q1_embeddings = batch_embeddings(df['question1'].tolist(), tokenizer, model, device)
q2_embeddings = batch_embeddings(df['question2'].tolist(), tokenizer, model, device)

from sklearn.metrics.pairwise import cosine_similarity
df['embedding_similarity'] = [cosine_similarity([q1], [q2])[0][0] for q1, q2 in zip(q1_embeddings, q2_embeddings)]

plt.figure(figsize=(10, 6))
sns.histplot(df['embedding_similarity'], kde=True, bins=50)
plt.title('Distribution of Embedding Similarities')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.show()

#%%
# Prepare Features and Target
features = [
    'embedding_similarity', 'len_diff', 'word_overlap', 'common_word_ratio',
    'named_entity_overlap', 'bigram_overlap', 'trigram_overlap'
]
X = df[features]
y = df['is_duplicate']

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# Train and evaluate traditional ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, preds)}")
    print(f"{name} Classification Report:\n{classification_report(y_test, preds)}")

#%%
# Ensemble using Logistic Regression
from sklearn.linear_model import LogisticRegression
train_probs = np.column_stack([model.predict_proba(X_train)[:, 1] for model in models.values()])
test_probs = np.column_stack([model.predict_proba(X_test)[:, 1] for model in models.values()])

meta_model = LogisticRegression().fit(train_probs, y_train)
ensemble_preds = meta_model.predict(test_probs)

print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_preds))
print("Ensemble Classification Report:\n", classification_report(y_test, ensemble_preds))

#%%
import tensorflow.keras.backend as K
import gc
from tensorflow.compat.v1 import reset_default_graph

# Clear Keras session
K.clear_session()

# Reset TensorFlow default graph (for graph mode)
reset_default_graph()

# Trigger garbage collection
gc.collect()

print("GPU memory cleared!")

#%%
# LSTM Siamese Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
q1_embeddings = q1_embeddings.astype('float32')
q2_embeddings = q2_embeddings.astype('float32')

embedding_dim = q1_embeddings.shape[1]

# Shared model
def create_shared_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    dense_layer = Dense(128, activation='relu')(input_layer)
    dropout_layer = Dropout(0.2)(dense_layer)
    dense_output = Dense(64, activation='relu')(dropout_layer)
    return Model(input_layer, dense_output)

# Siamese model
input_q1 = Input(shape=(embedding_dim,))
input_q2 = Input(shape=(embedding_dim,))
shared_model = create_shared_model(embedding_dim)

processed_q1 = shared_model(input_q1)
processed_q2 = shared_model(input_q2)

distance = Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))([processed_q1, processed_q2])
output_layer = Dense(1, activation='sigmoid')(distance)

siamese_model = Model(inputs=[input_q1, input_q2], outputs=output_layer)
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Siamese model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

siamese_model.fit(
    [q1_embeddings, q2_embeddings], y,
    batch_size=8, epochs=20,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler]
)

#%%
# Evaluate the Siamese model
y_pred_siamese = (siamese_model.predict([q1_embeddings[len(y_train):], q2_embeddings[len(y_train):]]) >= 0.5).astype(int)
print("Siamese Model Accuracy:", accuracy_score(y_test, y_pred_siamese))
print("Siamese Model Classification Report:\n", classification_report(y_test, y_pred_siamese))

#%%
# LSTM Siamese

from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Ensure embeddings are float32
q1_embeddings = q1_embeddings.astype('float32')
q2_embeddings = q2_embeddings.astype('float32')

# Reshape embeddings for LSTM (if they are 2D, make them 3D for sequential processing)
q1_embeddings = q1_embeddings[:, None, :]  # Shape: (num_samples, 1, embedding_dim)
q2_embeddings = q2_embeddings[:, None, :]  # Shape: (num_samples, 1, embedding_dim)

embedding_dim = q1_embeddings.shape[2]

# Shared LSTM model
def create_shared_lstm_model(input_dim):
    input_layer = Input(shape=(None, input_dim))  # Accept variable-length sequences
    lstm_layer = Bidirectional(LSTM(128, return_sequences=False))(input_layer)
    dropout_layer = Dropout(0.2)(lstm_layer)
    dense_output = Dense(64, activation='relu')(dropout_layer)
    return Model(input_layer, dense_output)

# Siamese model
input_q1 = Input(shape=(None, embedding_dim))
input_q2 = Input(shape=(None, embedding_dim))
shared_lstm_model = create_shared_lstm_model(embedding_dim)

processed_q1 = shared_lstm_model(input_q1)
processed_q2 = shared_lstm_model(input_q2)

# Compute absolute difference between processed outputs
distance = Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))([processed_q1, processed_q2])
output_layer = Dense(1, activation='sigmoid')(distance)

siamese_lstm_model = Model(inputs=[input_q1, input_q2], outputs=output_layer)
siamese_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the LSTM-based Siamese model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

siamese_lstm_model.fit(
    [q1_embeddings, q2_embeddings], y,
    batch_size=8,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stopping, lr_scheduler]
)

#%%
y_pred_siamese = (siamese_lstm_model.predict([q1_embeddings[len(y_train):], q2_embeddings[len(y_train):]]) >= 0.5).astype(int)
print("LSTM Siamese Model Accuracy:", accuracy_score(y_test, y_pred_siamese))
print("LSTM Siamese Model Classification Report:\n", classification_report(y_test, y_pred_siamese))
#%%
# Save Models
import joblib

for name, model in models.items():
    joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')

joblib.dump(meta_model, 'meta_model.pkl')
siamese_model.save('siamese_model.h5')

print("All models saved successfully.")

# %%
