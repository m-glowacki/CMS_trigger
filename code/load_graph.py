import numpy as np
import tensorflow as tf
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def main(training_data, graph):

    f=open(training_data, 'rb')
    data = pickle.load(f)  
    X = np.array(data["Events"].reshape(140000,20,12,1))
    y = np.array(data["Label"].reshape(-1,1))
    input_data = X[0][..., np.newaxis].reshape(1,20,12,1)

    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v2.io.gfile.GFile(graph, "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()

    with graph.as_default():
        net_inp, net_out = tf.import_graph_def(
            graph_def, return_elements=['input', 'output']
        )
        with tf.compat.v1.Session(graph=graph) as sess:
            print(np.array(net_out.outputs))
            out = sess.run(net_out.outputs, feed_dict={net_inp.outputs[0]: input_data})
            print((out))

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d","--data" , nargs="+", help="Path to pickled training data")
    parser.add_argument("-g","--graph" , nargs="+", help="saved constant graph")
    args = parser.parse_args()
    main(args.data[0], args.graph[0])