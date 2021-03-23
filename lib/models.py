from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GRU, Bidirectional, Dropout

# LSTM Network
def lstm_net(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, div_nodes, bidirectional):
    return recurrent_net(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, LSTM, div_nodes, bidirectional)

# GRU Network
def gru_net(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, div_nodes, bidirectional):
    return recurrent_net(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, GRU, div_nodes, bidirectional)

# Recurrent network
def recurrent_net(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, rec_layer, div_nodes=0, bidirectional=False):
    model = Sequential()
    if embedding_matrix == []:
        model.add(Embedding(length + 1, embed_dim, input_length=input_length, trainable=True, mask_zero=True, name='embedding'))
    else:
        model.add(Embedding(length + 1,
                             embed_dim,
                             weights=[embedding_matrix],
                             input_length=input_length,
                             trainable=False,
                             mask_zero=True, 
                             name='glove'))
    for layer in range(rec_layers):
        return_sequences = (layer != (rec_layers-1))
        if bidirectional:
            model.add(Bidirectional(rec_layer(rec_nodes, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences)))
        else:
            model.add(rec_layer(rec_nodes, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences))
        if (div_nodes != 0) and return_sequences:
            rec_nodes = int(rec_nodes/div_nodes)
    for layer in range(dense_layers):
        model.add(Dense(rec_nodes, activation='relu'))
        model.add(Dropout(dropout))
        if (div_nodes != 0):
            rec_nodes = int(rec_nodes/div_nodes)
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

##########################
#
#        OLD
#
##########################

# LSTM with the same number of nodes in each layer
def lstm_1(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, bidirectional):
    return recurrent_1(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, LSTM, bidirectional)

# LSTM with a pyramidal number of nodes in each layer
def lstm_2(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, bidirectional):
    return recurrent_2(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, LSTM, 2, bidirectional)

# GRU with the same number of nodes in each layer
def gru_1(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, bidirectional):
    return recurrent_1(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, GRU, bidirectional)

# GRU with a pyramidal number of nodes in each layer
def gru_2(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, bidirectional):
    return recurrent_2(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, GRU, 2, bidirectional)

# Recurrent network with the same number of nodes in each layer
def recurrent_1(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, rec_layer, bidirectional=False):
    model = Sequential()
    model.add(Embedding(length + 1,
                             embed_dim,
                             weights=[embedding_matrix],
                             input_length=input_length,
                             trainable=False,
                             mask_zero=True, 
                             name='embeddings'))
    for layer in range(rec_layers):
        return_sequences = (layer != (rec_layers-1))
        if bidirectional:
            model.add(Bidirectional(rec_layer(rec_nodes, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences)))
        else:
            model.add(rec_layer(rec_nodes, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences))
    for layer in range(dense_layers):
        model.add(Dense(rec_nodes, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# Recurrent network with a pyramidal number of nodes in each layer
def recurrent_2(length, embed_dim, embedding_matrix, input_length, rec_layers, rec_nodes, dense_layers, dropout, rec_layer, div_nodes, bidirectional=False):
    model = Sequential()
    model.add(Embedding(length + 1,
                             embed_dim,
                             weights=[embedding_matrix],
                             input_length=input_length,
                             trainable=False,
                             mask_zero=True, 
                             name='embeddings'))
    for layer in range(rec_layers):
        return_sequences = (layer != (rec_layers-1))
        if bidirectional:
            model.add(Bidirectional(rec_layer(rec_nodes, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences)))
        else:
            model.add(rec_layer(rec_nodes, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_sequences))
        if return_sequences:
            rec_nodes = int(rec_nodes/div_nodes)
    for layer in range(dense_layers):
        model.add(Dense(rec_nodes, activation='relu'))
        model.add(Dropout(dropout))
        rec_nodes = int(rec_nodes/div_nodes)
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

