# Import packages
###############################################################################################
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import wordnet

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
##################################################################################################
## Define a functon to automatically read and preprocess text into training and testing datasets##
##################################################################################################
def read_and_preprocess_text(read_type=None, test_size=0.1, random_state=2):
    '''
    Read and preprocess text into training and testing data.

    Parameters:
    ----------------
    read_type: {'l', 's', None} (default=None)
        None: no lemmatization nor stemming; 'l': lemmatization; 's': stemming
    test_size: float (default=0.1)
        Portion of data split for testing (0.0 to 1.0)
    '''
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)  # default type NOUN

    # Initialization
    positive_reviews = []
    negative_reviews = []
    with open(r"/content/drive/My Drive/Machine Learning/McGill COMP 550 NLP/Assignment 1/rt-polarity.pos",
              encoding='latin-1') as pos_file:
        if read_type == None:
            for row in pos_file:
                positive_reviews.append(row)
        elif read_type == 'l':
            # Lemmatization
            for row in pos_file:
                tokenized_row = word_tokenize(row)
                lemmatized_row = []
                for word in tokenized_row:
                    lemmatized_row.append(lemmatizer.lemmatize(word, pos=get_wordnet_pos(word)))
                positive_reviews.append(' '.join(lemmatized_row))
        elif read_type == 's':
            # Stemming
            for row in pos_file:
                tokenized_row = word_tokenize(row)
                stemmed_row = []
                for word in tokenized_row:
                    stemmed_row.append(stemmer.stem(word))
                positive_reviews.append(' '.join(stemmed_row))
        else:
            raise ValueError("Invalid value for read_type. read_type can only be None, 's', or 'l'."
                             "You entered {}".format(read_type))
    with open(r"/content/drive/My Drive/Machine Learning/McGill COMP 550 NLP/Assignment 1/rt-polarity.neg",
              encoding='latin-1') as neg_file:
        if read_type == None:
            for row in neg_file:
                negative_reviews.append(row)
        elif read_type == 'l':
            # Lemmatization
            for row in neg_file:
                tokenized_row = word_tokenize(row)
                lemmatized_row = []
                for word in tokenized_row:
                    lemmatized_row.append(lemmatizer.lemmatize(word, pos=get_wordnet_pos(word)))
                negative_reviews.append(' '.join(lemmatized_row))
        elif read_type == 's':
            # Stemming
            for row in neg_file:
                tokenized_row = word_tokenize(row)
                stemmed_row = []
                for word in tokenized_row:
                    stemmed_row.append(stemmer.stem(word))
                negative_reviews.append(' '.join(stemmed_row))
        else:
            raise ValueError("Invalid value for read_type. read_type can only be None, 's', or 'l'."
                             "You entered {}".format(read_type))
    positive_reviews = np.array(positive_reviews)  # shape = (5331,)
    negative_reviews = np.array(negative_reviews)  # shape = (5331,)

    # split traning, validation, and test sets
    X = np.concatenate((positive_reviews, negative_reviews), axis=0)
    y = np.concatenate((np.array([1 for i in range(positive_reviews.shape[0])]),
                        np.array([0 for i in range(negative_reviews.shape[0])]))
                       , axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

#################################################################################################
######################### Define a function to calculate baseline ###############################
#################################################################################################
def baseline_model(X):
    '''
    Baseline prediction model through randomly assign the class.
    '''
    y_pred = np.random.uniform(size=X.shape[0])
    y_pred = np.where(y_pred>=0.5, 1, 0)
    return y_pred

#################################################################################################
##################### Model Evaluation Without Lemmatization and Stemming #######################
#################################################################################################

# Define models
lr_clf = Pipeline([('vect', TfidfVectorizer()), ('lr', LogisticRegression(solver='lbfgs'))])
svm_clf = Pipeline([('vect', TfidfVectorizer()), ('svm', SVC(kernel='linear'))])
nb_clf = Pipeline([('vect', TfidfVectorizer()), ('nb', MultinomialNB())])

# Read Data
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type=None, test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)
# GridSearch for best parameters
stop_words = set(stopwords.words('english'))
lr_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0], 'vect__min_df': [1, 5, 10],
                 'vect__max_features': [5000, 7000, 10000, None],'lr__C': [0.1, 1, 10, 20],
                 'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
svm_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0], 'vect__min_df': [1, 5, 10],
                 'vect__max_features': [5000, 7000, 10000, None], 'svm__C': [0.1, 1, 10, 20]}
nb_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0], 'vect__min_df': [1, 5, 10],
                 'vect__max_features': [5000, 7000, 10000, None], 'nb__alpha': [0.05, 0.1, 0.5, 1]}
cnb_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0], 'vect__min_df': [1, 5, 10],
                 'vect__max_features': [5000, 7000, 10000, None], 'nb__alpha': [0.05, 0.1, 0.5, 1], 'nb__norm': [True, False]}

# Logistic Regression
gs_lr = GridSearchCV(lr_clf, lr_param_grid, n_jobs=-1, cv=5)
gs_lr.fit(X_train, y_train)
print(gs_lr.best_params_)
print(gs_lr.best_score_)
gs_lr_best = gs_lr.best_estimator_

gs_lr_cv_result = pd.DataFrame(gs_lr.cv_results_)
print("Average accuracy of logistic regression: ")
print("No stopwords: {}".format(gs_lr_cv_result.loc[gs_lr_cv_result["param_vect__stop_words"].isnull(), "mean_test_score"].mean()))
print("Has stopwords: {}".format(gs_lr_cv_result.loc[gs_lr_cv_result["param_vect__stop_words"].notnull(), "mean_test_score"].mean()))
print("max features 5000: {}".format(gs_lr_cv_result.loc[gs_lr_cv_result["param_vect__max_features"]==5000, "mean_test_score"].mean()))
print("max features 7000: {}".format(gs_lr_cv_result.loc[gs_lr_cv_result["param_vect__max_features"]==5000, "mean_test_score"].mean()))
print("max features 10000: {}".format(gs_lr_cv_result.loc[gs_lr_cv_result["param_vect__max_features"]==10000, "mean_test_score"].mean()))
print("max features None: {}".format(gs_lr_cv_result.loc[gs_lr_cv_result["param_vect__max_features"].isnull(), "mean_test_score"].mean()))
print("Validation accuracy of logistic regression: {}".format(gs_lr_best.score(X_valid, y_valid)))

# Support Vector Machine
gs_svm = GridSearchCV(svm_clf, svm_param_grid, n_jobs=-1, cv=5)
gs_svm.fit(X_train, y_train)
print(gs_svm.best_params_)
print(gs_svm.best_score_)
gs_svm_best = gs_svm.best_estimator_

gs_svm_cv_result = pd.DataFrame(gs_svm.cv_results_)
print("Average accuracy of logistic regression: ")
print("No stopwords: {}".format(gs_svm_cv_result.loc[gs_svm_cv_result["param_vect__stop_words"].isnull(), "mean_test_score"].mean()))
print("Has stopwords: {}".format(gs_svm_cv_result.loc[gs_svm_cv_result["param_vect__stop_words"].notnull(), "mean_test_score"].mean()))
print("max features 5000: {}".format(gs_svm_cv_result.loc[gs_svm_cv_result["param_vect__max_features"]==5000, "mean_test_score"].mean()))
print("max features 7000: {}".format(gs_svm_cv_result.loc[gs_svm_cv_result["param_vect__max_features"]==5000, "mean_test_score"].mean()))
print("max features 10000: {}".format(gs_svm_cv_result.loc[gs_svm_cv_result["param_vect__max_features"]==10000, "mean_test_score"].mean()))
print("max features None: {}".format(gs_svm_cv_result.loc[gs_svm_cv_result["param_vect__max_features"].isnull(), "mean_test_score"].mean()))
print("Validation accuracy of Support Vector Machine: {}".format(gs_svm_best.score(X_valid, y_valid)))

# multinomialNB
gs_nb = GridSearchCV(nb_clf, nb_param_grid, n_jobs=-1, cv=5)
gs_nb.fit(X_train, y_train)
print(gs_nb.best_params_)
print(gs_nb.best_score_)
gs_nb_best = gs_nb.best_estimator_

gs_nb_cv_result = pd.DataFrame(gs_nb.cv_results_)
print("Average accuracy of logistic regression: ")
print("No stopwords: {}".format(gs_nb_cv_result.loc[gs_nb_cv_result["param_vect__stop_words"].isnull(), "mean_test_score"].mean()))
print("Has stopwords: {}".format(gs_nb_cv_result.loc[gs_nb_cv_result["param_vect__stop_words"].notnull(), "mean_test_score"].mean()))
print("max features 5000: {}".format(gs_nb_cv_result.loc[gs_nb_cv_result["param_vect__max_features"]==5000, "mean_test_score"].mean()))
print("max features 7000: {}".format(gs_nb_cv_result.loc[gs_nb_cv_result["param_vect__max_features"]==5000, "mean_test_score"].mean()))
print("max features 10000: {}".format(gs_nb_cv_result.loc[gs_nb_cv_result["param_vect__max_features"]==10000, "mean_test_score"].mean()))
print("max features None: {}".format(gs_nb_cv_result.loc[gs_nb_cv_result["param_vect__max_features"].isnull(), "mean_test_score"].mean()))
print("Validation accuracy of MultinomialNB: {}".format(gs_nb_best.score(X_valid, y_valid)))

#################################################################################################
############################# Model Evaluation With Lemmatization ###############################
#################################################################################################

# Read data
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type='l', test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)
stop_words = set(stopwords.words('english'))
# GridSearch for best parameters
lr_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0],
                 'vect__max_features': [5000, 10000, None],'lr__C': [0.1, 1, 100],
                 'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
svm_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0],
                 'vect__max_features': [5000, 10000, None], 'svm__C': [0.1, 1, 100]}
nb_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0],
                 'vect__max_features': [5000, 10000, None], 'nb__alpha': [0, 1, 2]}

# Define models
lr_clf_lemma = Pipeline([('vect', TfidfVectorizer()), ('lr', LogisticRegression(solver='lbfgs'))])
svm_clf_lemma = Pipeline([('vect', TfidfVectorizer()), ('svm', SVC(kernel='linear'))])
nb_clf_lemma = Pipeline([('vect', TfidfVectorizer()), ('nb', MultinomialNB())])

# Logistic Regression
gs_lr_lemma = GridSearchCV(lr_clf_lemma, lr_param_grid, n_jobs=-1, cv=5)
gs_lr_lemma.fit(X_train, y_train)
print(gs_lr_lemma.best_params_)
print(gs_lr_lemma.best_score_)
gs_lr_lemma_best = gs_lr_lemma.best_estimator_
print("Validation accuracy of logistic regression: {}".format(gs_lr_lemma_best.score(X_valid, y_valid)))

# Support Vector Machine
gs_svm_lemma = GridSearchCV(svm_clf_lemma, svm_param_grid, n_jobs=-1, cv=5)
gs_svm_lemma.fit(X_train, y_train)
print(gs_svm_lemma.best_params_)
print(gs_svm_lemma.best_score_)
gs_svm_lemma_best = gs_svm_lemma.best_estimator_
print("Validation accuracy of Support Vector Machine: {}".format(gs_svm_lemma_best.score(X_valid, y_valid)))

# multinomialNB
gs_nb_lemma = GridSearchCV(nb_clf_lemma, nb_param_grid, n_jobs=-1, cv=5)
gs_nb_lemma.fit(X_train, y_train)
print(gs_nb_lemma.best_params_)
print(gs_nb_lemma.best_score_)
gs_nb_lemma_best = gs_nb_lemma.best_estimator_
print("Validation accuracy of MultinomialNB: {}".format(gs_nb_lemma_best.score(X_valid, y_valid)))

#################################################################################################
################################ Model Evaluation With Stemming #################################
#################################################################################################
# Define models
lr_clf_stem = Pipeline([('vect', TfidfVectorizer()), ('lr', LogisticRegression(solver='lbfgs'))])
svm_clf_stem = Pipeline([('vect', TfidfVectorizer()), ('svm', SVC(kernel='linear'))])
nb_clf_stem = Pipeline([('vect', TfidfVectorizer()), ('nb', MultinomialNB())])

# Read data
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type='s', test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)
stop_words = set(stopwords.words('english'))
# GridSearch for best parameters
lr_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0],
                 'vect__max_features': [5000, 10000, None],'lr__C': [0.1, 1, 10],
                 'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
svm_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0],
                 'vect__max_features': [5000, 10000, None], 'svm__C': [0.1, 1, 10]}
nb_param_grid = {'vect__stop_words': [None, stop_words], 'vect__max_df': [0.6, 0.8, 1.0],
                 'vect__max_features': [5000, 10000, None], 'nb__alpha': [0.1, 1, 10]}

# Logistic Regression
gs_lr_stem = GridSearchCV(lr_clf, lr_param_grid, n_jobs=-1, cv=5)
gs_lr_stem.fit(X_train, y_train)
print(gs_lr_stem.best_params_)
print(gs_lr_stem.best_score_)
gs_lr_stem_best = gs_lr_stem.best_estimator_
print("Validation accuracy of logistic regression: {}".format(gs_lr_stem_best.score(X_valid, y_valid)))

# Support Vector Machine
gs_svm_stem = GridSearchCV(svm_clf_stem, svm_param_grid, n_jobs=-1, cv=5)
gs_svm_stem.fit(X_train, y_train)
print(gs_svm_stem.best_params_)
print(gs_svm_stem.best_score_)
gs_svm_stem_best = gs_svm_stem.best_estimator_
print("Validation accuracy of Support Vector Machine: {}".format(gs_svm_stem_best.score(X_valid, y_valid)))

# multinomialNB
gs_nb_stem = GridSearchCV(nb_clf_stem, nb_param_grid, n_jobs=-1, cv=5)
gs_nb_stem.fit(X_train, y_train)
print(gs_nb_stem.best_params_)
print(gs_nb_stem.best_score_)
gs_nb_stem_best = gs_nb_stem.best_estimator_
print("Validation accuracy of MultinomialNB: {}".format(gs_nb_stem_best.score(X_valid, y_valid)))

#################################################################################################
############ Artificial Neural Network Without Lemmatization and Stemming #######################
#################################################################################################

# Read data
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type=None, test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)

# Parameters
num_words = 10000
oov_token = '<OOV>'
maxlen= 60
embedding_dim = 32
# Preprocessing
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
padded_train_sequences = pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
valid_sequences = tokenizer.texts_to_sequences(X_valid)
padded_valid_sequences = pad_sequences(valid_sequences, maxlen=maxlen, padding='post', truncating='post')

# Build Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12, recurrent_dropout=0.5, dropout=0.5)))
model.add(tf.keras.layers.Dense(12))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=5)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
# Train
history = model.fit(padded_train_sequences, y_train, batch_size=32, epochs=100, validation_data=(padded_valid_sequences, y_valid), callbacks=[early_stop, lr_schedule])

#################################################################################################
####################### Artificial Neural Network WithLemmatization #############################
#################################################################################################

# Read data
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type='l', test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)

# Parameters
num_words = 10000
oov_token = '<OOV>'
maxlen= 60
embedding_dim = 32
# Preprocessing
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
padded_train_sequences = pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
valid_sequences = tokenizer.texts_to_sequences(X_valid)
padded_valid_sequences = pad_sequences(valid_sequences, maxlen=maxlen, padding='post', truncating='post')

# Build Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12, recurrent_dropout=0.5, dropout=0.5)))
model.add(tf.keras.layers.Dense(12))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=5)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
# Train
history = model.fit(padded_train_sequences, y_train, batch_size=32, epochs=100, validation_data=(padded_valid_sequences, y_valid), callbacks=[early_stop, lr_schedule])

#################################################################################################
######################### Artificial Neural Network With Stemming ###############################
#################################################################################################

# Read data
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type='s', test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)

# Parameters
num_words = 10000
oov_token = '<OOV>'
maxlen= 60
embedding_dim = 32
# Preprocessing
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
train_sequences = tokenizer.texts_to_sequences(X_train)
padded_train_sequences = pad_sequences(train_sequences, maxlen=maxlen, padding='post', truncating='post')
valid_sequences = tokenizer.texts_to_sequences(X_valid)
padded_valid_sequences = pad_sequences(valid_sequences, maxlen=maxlen, padding='post', truncating='post')

# Build Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12, recurrent_dropout=0.5, dropout=0.5)))
model.add(tf.keras.layers.Dense(12))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=5)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(patience=3)
# Train
history = model.fit(padded_train_sequences, y_train, batch_size=32, epochs=100, validation_data=(padded_valid_sequences, y_valid), callbacks=[early_stop, lr_schedule])


#################################################################################################
###################################### Baseline Model ###########################################
#################################################################################################

# No lemmatization nor stemming
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type=None, test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)
print("Baseline accuracy: {}".format((baseline_model(X_valid)==y_valid).sum()/y_valid.shape[0]))

# Lemmatization
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type='l', test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)
print("Baseline accuracy: {}".format((baseline_model(X_valid)==y_valid).sum()/y_valid.shape[0]))

# Stemming
X_train, X_test, y_train, y_test = read_and_preprocess_text(read_type='s', test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 2, test_size=0.2)
print("Baseline accuracy: {}".format((baseline_model(X_valid)==y_valid).sum()/y_valid.shape[0]))


#################################################################################################
############################## Confusion Matrix For multinomialNB ###############################
#################################################################################################

# Confusion matrix
y_pred = gs_nb_best.predict(X_test)
target_names = ['Negative', 'Positive']
print(classification_report(y_pred=y_pred, y_true=y_test, target_names=target_names))