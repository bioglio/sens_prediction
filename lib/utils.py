import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

##############################
#
#   Read dataset
#
##############################

# read raw data and keep only the tags having at least (or exactly) "lim" count
def get_data(path, lim=2, gte=True):
    data = pd.read_csv(path, sep=',', error_bad_lines=False)
    res = {"uri":[], "text":[], "class":[]}
    for tag in data.columns[2:]:
        if gte: # at least "lim" count
            tmp = data[data[tag] >= lim]
        else: # exactly "lim" count
            tmp = data[data[tag] == lim]
        res["uri"] += tmp["uri"].tolist()
        res["text"] += tmp["text"].tolist()
        res["class"] += [tag]*len(tmp)
    res = pd.DataFrame(res).sort_values(by="uri")
    return res

# keep only "Sensibile" and "Non sensibile" as "sens" and "ns"
def get_two_classes(df):
    res = df[df["class"].isin(["Sensibile", "Non sensibile"])].copy()
    res.loc[res["class"] == "Sensibile", "class"] = "sens"
    res.loc[res["class"] == "Non sensibile", "class"] = "ns"
    return res

##############################
#
#   Embedding
#
##############################

# get embedding matrix rom embedding index
def embedding_matrix(tokenizer, embeddings_index, glove_dim=100):
    hits = 0
    misses = 0
    embedding_matrix = np.random.random((len(tokenizer.word_index) + 1, glove_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Completed! Converted", hits, "words ("+str(misses)+" misses)")
    return embedding_matrix

# get word embedding
def word_embedding(pathname):
    embeddings_index = {}
    f = open(pathname, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()
    return embeddings_index

# get embedding matrix from retrained embedding file
def get_embedding_matrix(pathname, tokenizer, GLOVE_DIM):
  emb_index = word_embedding(pathname)
  print('Loaded %s word vectors.' % len(emb_index))

  return embedding_matrix(tokenizer, emb_index, GLOVE_DIM)

##############################
#
#   Training
#
##############################

# get train, validation and test sets
def my_train_val_test_split__(data, random_state):
    data_size = len(data)
    test_size = int(data_size*test_ratio)
    val_size = int(data_size*validation_ratio)
    # create test set
    X = data["text"].values
    Y = data['class'].values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=True, stratify=Y, random_state=random_state)
    # create train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, shuffle=True, stratify=y_train, random_state=random_state)
    return x_train, x_val, x_test, y_train, y_val, y_test

# get train, validation and test sets
def my_train_val_test_split(X, Y, validation_ratio, test_ratio, random_state):
    data_size = len(X)
    test_size = int(data_size*test_ratio)
    val_size = int(data_size*validation_ratio)
    # create test set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=True, stratify=Y, random_state=random_state)
    # create train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, shuffle=True, stratify=y_train, random_state=random_state)
    return x_train, x_val, x_test, y_train, y_val, y_test

# text preprocessing: return list of tensors for network
def texts_preprocessing(texts, tokenizer, max_len):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)

# fit with early stopping on Accuracy
def fit_es(model, x_train, y_train, x_val, y_val, tokenizer, max_len, max_epochs=30, patience=5):
    early_stopping_cb = EarlyStopping(monitor= 'val_accuracy', mode='max', 
                                                      verbose=1, patience=patience, restore_best_weights=True)
    x_train_pre = texts_preprocessing(x_train, tokenizer, max_len)
    x_val_pre = texts_preprocessing(x_val, tokenizer, max_len)
    history = model.fit(x_train_pre, y_train, epochs=max_epochs, batch_size=max_epochs, verbose=1, 
                        validation_data=(x_val_pre, y_val), callbacks=[early_stopping_cb])
    return history

##############################
#
#   Prediction
#
##############################

# use the trained model for predicting the class probabilities of a text
def predict_class(text, model, tokenizer, Y_feat_names, max_len=67, real_class=None):
    input_pred = texts_preprocessing([text], tokenizer, max_len)
    result_pred = model.predict(input_pred)
    result_feat = Y_feat_names[0]
    if result_pred[0][1] > result_pred[0][0]:
        result_feat = Y_feat_names[1]
    if real_class == None:
        print(text,"->",result_feat, "("+Y_feat_names[0]+": "+str(result_pred[0][0]), Y_feat_names[1]+": "+str(result_pred[0][1])+")")
    else:
        result_true = '(V)'
        if result_feat != real_class:
            result_true = '(X)'
        print(text,"->",result_feat,result_true, "("+Y_feat_names[0]+": "+str(result_pred[0][0]), Y_feat_names[1]+": "+str(result_pred[0][1])+")["+real_class+"]")
    return

##############################
#
#   Plots
#
##############################

# plot history of training
def plot_history(history, pathname=""):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
    if pathname != "":
        plt.savefig(pathname)
    plt.show()
    return

"""
# plot the confusion matrix
def plot_confusion_matrix__(c_mat, normalize=False, labels=[]):
    # Normalize confusion matrix
    cbar_label = "Number of samples"
    if normalize:
        c_mat = c_mat/c_mat.sum(axis=1)[:, np.newaxis]
        cbar_label = "Ratio of samples"
    # Plot the heatmap
    sns.heatmap(c_mat, annot=True, cmap="YlGnBu", cbar_kws={'label': cbar_label}, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=0) 
    plt.show()
    return
"""

# plot the confusion matrix
def plot_confusion_matrix(df, normalize=False, labels=[], pathname=""):
    # Calculate confusion matrix
    real_labels = df["true"]
    predicted_labels = df["pred"]
    c_mat = confusion_matrix(real_labels, predicted_labels)
    return plot_confusion_matrix_from_cmat(c_mat, normalize=normalize, labels=labels, pathname=pathname)

def plot_confusion_matrix_from_cmat(c_mat, normalize=False, labels=[], pathname=""):
    # Normalize confusion matrix
    cbar_label = "Number of samples"
    if normalize:
        c_mat = c_mat/c_mat.sum(axis=1)[:, np.newaxis]
        cbar_label = "Ratio of samples"
    # Plot the heatmap
    sns.heatmap(c_mat, annot=True, cmap="YlGnBu", cbar_kws={'label': cbar_label}, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
    plt.yticks(rotation=0) 
    if pathname != "":
        plt.savefig(pathname)
    plt.show()
    return

# get the prediction on a list of samples
def get_predictions_df(model, X, Y, batch_size):
    #Y_pred_m = model.predict_classes(X, batch_size=batch_size)
    Y_pred_m = np.argmax(model.predict(X, batch_size=batch_size), axis=-1)
    df_test = pd.DataFrame({'true': Y.tolist(), 'pred': Y_pred_m})
    df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))
    return df_test

# use the trained model for predicting the class probabilities of a text
def predict_class_BERT(text, predictor, Y_feat_names, real_class=None):
    result_pred = predictor.predict_proba(text)
    result_feat = Y_feat_names[0]
    if result_pred[1] > result_pred[0]:
        result_feat = Y_feat_names[1]
    if real_class == None:
      print(text,"->",result_feat, "("+Y_feat_names[0]+": "+str(result_pred[0]), Y_feat_names[1]+": "+str(result_pred[1])+")")
    else:
      result_true = '(V)'
      if result_feat != real_class:
        result_true = '(X)'
      print(text,"->",result_feat,result_true, "("+Y_feat_names[0]+": "+str(result_pred[0]), Y_feat_names[1]+": "+str(result_pred[1])+")["+real_class+"]")
    return

# plot confusion matrix and print model evaluaton on the test set    
def plot_model_evaluation(x_test, y_test, model, tokenizer, feat_labels, max_len, batch_size, output_dict=False, pathname=""):
    print("Model evaluation on test set")
    # preprocessing
    x_test_pre = texts_preprocessing(x_test, tokenizer, max_len)

    # print accuracy on test set
    scores = model.evaluate(x_test_pre, y_test, verbose=0)
    #res = model.metrics_names[1]+": "+"{:.2f}".format(scores[1] * 100)+"%\n\n"
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # plot confusion matrix on test set
    df_test = get_predictions_df(model, x_test_pre, y_test, batch_size)
    plot_confusion_matrix(df_test, normalize=True, labels=feat_labels, pathname=pathname)

    #print(classification_report(df_test.true, df_test.pred))
    #res = res + classification_report(df_test.true, df_test.pred) + "\n"
    res = classification_report(df_test.true, df_test.pred, output_dict=output_dict)
    return res


