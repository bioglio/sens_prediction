######################################
#
#             EXPERIMENT 0
#
#  Find the best Networks for the current domain
#
######################################


import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import time

from lib.utils import get_data, get_two_classes, my_train_val_test_split, texts_preprocessing, get_embedding_matrix, fit_es, plot_history, plot_model_evaluation
from lib.models import lstm_net, gru_net

#### Variables

# directories
main_path = "./"
glove_path = main_path+"glove/"
results_path = main_path+"experiments_00/"
dataset_wt_filename = main_path+"data/annotation_results__ann"

# training
Y_feat_names = ["ns", "sens"]

# recurrent NN
VOCABULARY_SIZE = 10000
BATCH_SIZE = 50
DROPOUT = 0.5

os.mkdir(results_path)

#### Loop

count = 1
experiment_time = time.perf_counter()

for num_annotators in [2,3]:
    # get data
    # get data
    train_loc = pd.read_csv(dataset_wt_filename+str(num_annotators)+"_training.csv")
    val_loc = pd.read_csv(dataset_wt_filename+str(num_annotators)+"_validation.csv")
    test_loc = pd.read_csv(dataset_wt_filename+str(num_annotators)+"_test.csv")
    # transform data into sets for models
    x_train = train_loc["text"].tolist()
    y_train = pd.get_dummies(pd.DataFrame({"class": train_loc["class"].tolist()})["class"])[Y_feat_names].values
    x_val = val_loc["text"].tolist()
    y_val = pd.get_dummies(pd.DataFrame({"class": val_loc["class"].tolist()})["class"])[Y_feat_names].values
    x_test = test_loc["text"].tolist()
    y_test = pd.get_dummies(pd.DataFrame({"class": test_loc["class"].tolist()})["class"])[Y_feat_names].values
    print("\nNumber of samples in each class",Y_feat_names, "in: training set ->", y_train.sum(axis=0), "; validation set ->", y_val.sum(axis=0), "; test set ->", y_test.sum(axis=0))
    # create tokenizer from training data
    tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, split=' ')
    tokenizer.fit_on_texts(x_train)
    for glove in ["twit", "wiki", "no"]:
        for glove_dim in [100, 200]:
            emb_path = ""
            if glove == "wiki":
                emb_path = glove_path+'/glove.6B.' + str(glove_dim) + 'd.txt'
            elif glove == "twit":
                emb_path = glove_path+'/glove.twitter.27B.' + str(glove_dim) + 'd.txt'
            if emb_path != "":
                emb_matrix = get_embedding_matrix(emb_path, tokenizer, glove_dim)
                print("Shape of embedding matrix:", emb_matrix.shape)
            else:
                emb_matrix = []
            length = len(tokenizer.word_index)
            for num_layers in [1,2]:
                for num_nodes in [128,256]: #[64,128,256]:
                    for rnn in ["lstm", "gru"]:
                        for bidirectional in [True, False]:
                            bid = ""
                            if bidirectional:
                                    bid = "_bid"
                            for pyramid in [True, False]:
                                pyr = ""
                                div_nodes = 0
                                if pyramid:
                                    pyr = "_pyr"
                                    div_nodes = 2
                                # create directory
                                print('Model '+str(count)+':\n')
                                local_path = results_path+"ann"+str(num_annotators)+"_"+glove+"_gdim"+str(glove_dim)+"_"+rnn+bid+"_nlayers"+str(num_layers)+"_nnodes"+str(num_nodes)+pyr+"/"
                                print(local_path)
                                os.mkdir(local_path)
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
                                # plot history
                                plot_history(history, pathname=local_path+"history.png")
                                # model evaluation on test set
                                evalu = plot_model_evaluation(x_test, y_test, model, tokenizer, Y_feat_names, glove_dim, BATCH_SIZE, output_dict=True, pathname=local_path+"confusion_matrix.png")
                                evalu = pd.DataFrame.from_dict(evalu)
                                print(evalu)
                                evalu = evalu.append(pd.DataFrame([[int(training_time)]+[0]*(len(evalu.columns)-1)], columns=evalu.columns.to_list(), index=["training_time"]))
                                evalu.to_csv(local_path+"evaluation.csv", index=False)
                                # save model
                                model.save(local_path+"model")
                                count += 1


print("\n\nNumber of models:", count)
experiment_time = time.perf_counter() - experiment_time
print("Experiment time in seconds:", int(experiment_time))
print("Experiment time:", time.strftime("/%d, %H:%M:%S",time.gmtime(experiment_time)))

