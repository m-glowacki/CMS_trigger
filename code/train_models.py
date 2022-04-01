from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tensorflow as tf  
from datetime import datetime
from tensorflow.keras.layers import *
import pickle
from tensorflow.keras import activations
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np
from qkeras import quantized_bits
from qkeras import QDense, QActivation, QConv2D
from qkeras import QBatchNormalization
import os


def override_tensorflow():
    from tensorflow.compat.v1.keras import backend as K
    if tf.__version__.startswith("2."):
        tf = tf.compat.v1
    tf.disable_eager_execution()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    from tensorflow.compat.v1.keras import backend as K
    K.set_learning_phase(0) 
    K.set_session(sess)

def create_keras_model(tensorflow):
    if tensorflow == 1:
        override_tensorflow()
    
    #1st layer
    print(f"training Keras model with tf {tensorflow}")
    model = tf.keras.Sequential()
    inputs = model.add(Input(shape=(20,12,1), name="input_1"))
    model.add(Conv2D(4, kernel_size=(3,3), input_shape=(20,12,1)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    #2nd layer
    model.add(Conv2D(8, kernel_size=(3,3)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    
    #output layer
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.1))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(BatchNormalization(axis=1))
    outputs = model.add(Activation('sigmoid', name="output/"))

    model.build(input_shape=(20,12,1))

    opt = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    return model


def create_model_q_model(tensorflow):
    if tensorflow==1:
        print("QKeras not compatible with TF < 2.6")
        exit()
    
    print("training QKeras model")
    #1st layer
    model = tf.keras.Sequential()
    inputs = model.add(Input(shape=(20,12,1), name="input_1"))
    model.add(QConv2D(4, kernel_size=(3,3), kernel_quantizer=quantized_bits(5),bias_quantizer=quantized_bits(5), input_shape=(20,12,1)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
    model.add(Dropout(0.1))
    model.add(QBatchNormalization(axis=1))
    model.add(QActivation('quantized_relu(5,0)'))
    
    #2nd layer
    model.add(QConv2D(8,kernel_quantizer=quantized_bits(5),bias_quantizer=quantized_bits(5), kernel_size=(3,3)))
    model.add(MaxPooling2D (pool_size=(2,2), padding='valid'))
    model.add(Dropout(0.1))
    model.add(QBatchNormalization(axis=1))
    model.add(QActivation('quantized_relu(5,0)'))

    #3rd layer
    model.add(Flatten())
    model.add(QDense(64,kernel_quantizer=quantized_bits(5),bias_quantizer=quantized_bits(5)))
    model.add(Dropout(0.1))
    model.add(QBatchNormalization(axis=1))
    model.add(QActivation('quantized_relu(5,0)'))

    model.add(QDense(1,kernel_quantizer=quantized_bits(5),bias_quantizer=quantized_bits(5)))
    model.add(QBatchNormalization(axis=1))
    outputs = model.add(Activation('sigmoid', name="output/"))

    model.build(input_shape=(20,12,1))

    opt = tf.keras.optimizers.SGD(learning_rate=0.01, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    return model

def prepare_dataset(data):
    f=open(data, 'rb')
    data = pickle.load(f)  
    X = np.array(data["events"].reshape(-1,20,12,1))
    y = np.array(data["labels"].reshape(-1,1))
    return X, y

def fitter(model, train_X, train_y, outdir, model_type, callback=True):
    if model_type == "qk":
        save_dir = "Qmodel"
    else:
        save_dir= "KerasModel"
    print(model.summary())
    if callback:
        history = model.fit(train_X, 
            train_y,
            epochs=50,
            shuffle=True,
            verbose=1,
            batch_size=128,
            validation_split=.25,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                patience=10,
                                                                restore_best_weights=True)])
    else:
         history = model.fit(train_X, 
            train_y,
            epochs=50,
            shuffle=True,
            verbose=1,
            batch_size=128,
            validation_split=.25)

    model.save(f"{outdir}/{save_dir}")
    model.save_weights(f"{outdir}/{save_dir}.h5")
    return model, history

def make_graph(model, outdir, tensorflow):
    if tensorflow == 1: 
        override_tensorflow()
        outputs = [x.op.name for x in model.outputs]
        inputs = [x.op.name for x in model.inputs]
        constant_graph = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), outputs)
        tf.train.write_graph(constant_graph, f"{outdir}/Qcnn_graph.pb", as_text=False)
    
    elif tensorflow==2:  
        import cmsml
        cmsml.tensorflow.save_graph(f"{outdir}/Qcnn_graph.pb", model, variables_to_constants=True)

def make_plots(history, outdir, model):
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    ax1.plot(history.history['binary_accuracy'])
    ax1.plot(history.history['val_binary_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='upper left')
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(f"{outdir}/training_curves{model}.png")


def main(data, outdir_, model_type, tensorflow):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H:%M")
    outdir = f"{outdir_}/{dt_string}"
    os.makedirs(outdir, exist_ok=True)
    if model_type == "qk":
        model = create_model_q_model(tensorflow)
    else:
        model = create_keras_model(tensorflow)

    X_train, y_train  = prepare_dataset(data)
    model, history = fitter(model, X_train, y_train, outdir, model_type)

    print("making constant graph..")
    make_graph(model, outdir, tensorflow)
    make_plots(history,outdir, model_type)
    print("done")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d","--data" , nargs="+", help="Path to pickled training data")
    parser.add_argument("-o","--outdir" , nargs="+", help="path to output dir of model and graph")
    parser.add_argument("-m","--model" , nargs="+", help="train with regular or QKeras. Options: [-m qk], [-m k]")
    parser.add_argument("-tf","--tensorflow" , nargs="+", help="which version of TF to use. Options: [-tf 1], [-tf 2]")
    args = parser.parse_args()
    main(args.data[0], args.outdir[0], args.model[0], args.tensorflow[0])
