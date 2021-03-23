######################################
#
#             EXPERIMENT 2
#
#  Train 4 Networks on the WH+TW datasets
#
######################################

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import time

from lib.utils import texts_preprocessing, get_embedding_matrix, fit_es, plot_history, plot_model_evaluation#plot_confusion_matrix#, predict_class
from lib.models import lstm_net, gru_net
from sklearn import metrics

def get_metrics(y_test, y_pred, Y_feat_names):
    res = {}
    res["accuracy"] = metrics.accuracy_score(y_test, y_pred)
    tmp = metrics.precision_score(y_test, y_pred, average=None)
    for index, cls in enumerate(Y_feat_names):
        res["precision_"+cls] = tmp[index]
    tmp = metrics.recall_score(y_test, y_pred, average=None)
    for index, cls in enumerate(Y_feat_names):
        res["recall_"+cls] = tmp[index]
    tmp = metrics.f1_score(y_test, y_pred, average=None)
    for index, cls in enumerate(Y_feat_names):
        res["f1_"+cls] = tmp[index]
    res["f1-micro"] = metrics.f1_score(y_test, y_pred, average="micro")
    res["f1-macro"] = metrics.f1_score(y_test, y_pred, average="macro")
    return res

def get_prediction_metrics(x_test, y_test, tokenizer, glove_dim):
    # preprocessing
    x_test_pre = texts_preprocessing(x_test, tokenizer, glove_dim)
    y_pred_loc = model.predict(x_test_pre)
    y_pred_loc = [Y_feat_names[0] if x[0]>=0.5 else Y_feat_names[1] for x in y_pred_loc]
    met_dict_loc = get_metrics(y_test, y_pred_loc, Y_feat_names)
    return met_dict_loc

#### Variables

# directories
main_path = "./"
dataset_filename = main_path+"data/annotation_results__ann"
dataset_wt_filename = main_path+"data/sample_ann2_"
glove_path = main_path+"glove/"
results_path = main_path+"experiments_02/"
models_path = results_path+"models/"
stats_path = results_path+"results/"
Y_feat_names = ["ns", "sens"]

# recurrent NN
VOCABULARY_SIZE = 10000
BATCH_SIZE = 50
DROPOUT = 0.5

# variables of NNs
ann_list = [2, 2, 3, 3]
glove_dim_list = [100, 100, 200, 200]
rnn_list = ["gru", "lstm", "gru", "lstm"]
bidirectional_list = [True, True, False, False]
num_layers_list = [2, 2, 1, 2]
num_nodes_list = [128, 256, 128, 128]
pyramid_list = [False, True, False, False]
glove = "twit"

experiment_time = time.perf_counter()

for index in range(10):
    # get data
    train_loc = pd.read_csv(dataset_wt_filename+(("0"+str(index+1))[-2:])+"_training.csv")
    val_loc = pd.read_csv(dataset_wt_filename+(("0"+str(index+1))[-2:])+"_validation.csv")
    test_loc = pd.read_csv(dataset_wt_filename+(("0"+str(index+1))[-2:])+"_test.csv")
    # transform data into sets for models
    x_train = train_loc["text"].tolist()
    y_train = pd.get_dummies(pd.DataFrame({"class": train_loc["class"].tolist()})["class"])[Y_feat_names].values
    x_val = val_loc["text"].tolist()
    y_val = pd.get_dummies(pd.DataFrame({"class": val_loc["class"].tolist()})["class"])[Y_feat_names].values
    x_test = test_loc["text"].tolist()
    y_test = pd.get_dummies(pd.DataFrame({"class": test_loc["class"].tolist()})["class"])[Y_feat_names].values
    # create tokenizer from training data
    tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, split=' ')
    tokenizer.fit_on_texts(x_train)
    length = len(tokenizer.word_index)
    # loop on models
    for ann, glove_dim, rnn, bidirectional, num_layers, num_nodes, pyramid in zip(ann_list, glove_dim_list, rnn_list, bidirectional_list, num_layers_list, num_nodes_list, pyramid_list):
        # other
        if bidirectional:
            bid = "_bid"
        pyr = ""
        div_nodes = 0
        if pyramid:
            pyr = "_pyr"
            div_nodes = 2
        # load embedding
        emb_path = glove_path+'/glove.twitter.27B.' + str(glove_dim) + 'd.txt'
        if emb_path != "":
            emb_matrix = get_embedding_matrix(emb_path, tokenizer, glove_dim)
        else:
            emb_matrix = []
        # create and compile the model
        if rnn == "lstm":
            model = lstm_net(length, glove_dim, emb_matrix, glove_dim, num_layers, num_nodes, num_layers, DROPOUT, div_nodes, bidirectional)
        elif rnn == "gru":
            model = gru_net(length, glove_dim, emb_matrix, glove_dim, num_layers, num_nodes, num_layers, DROPOUT, div_nodes, bidirectional)
        # fit the model
        training_time = time.perf_counter()
        history = fit_es(model, x_train, y_train, x_val, y_val, tokenizer, glove_dim)
        training_time = time.perf_counter() - training_time
        print("Training time:", time.strftime("%H:%M:%S",time.gmtime(training_time)))
        # save model
        model_name = "sample"+(("0"+str(index+1))[-2:])+"_ann"+str(ann)+"_"+glove+"_gdim"+str(glove_dim)+"_"+rnn+bid+"_nlayers"+str(num_layers)+"_nnodes"+str(num_nodes)+pyr
        model.save(models_path+model_name)
        # model evaluation on test set
        met_dict = get_prediction_metrics(x_test, [Y_feat_names[0] if x[0]>=0.5 else Y_feat_names[1] for x in y_test], tokenizer, glove_dim)
        pd.DataFrame([met_dict]).to_csv(stats_path+model_name+".csv")

experiment_time = time.perf_counter() - experiment_time
print("Experiment time in seconds:", int(experiment_time))
print("Experiment time:", time.strftime("/%d, %H:%M:%S",time.gmtime(experiment_time)))

