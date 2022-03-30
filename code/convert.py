# from architecture import EAST
# import torch
import onnx
# import onnxruntime
import numpy as np
from onnx_tf.backend import prepare
import tensorflow as tf
# from pathlib import Path
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util

import traceback


class TorchToOnnx(object):
    def __init__(self):
        self.onnx_file = '../models/ijlal_model.onnx'
        self.weights_file = '../models/model.pth'
        self.device = "cpu"
        self.model = EAST().to(self.device)

    def __call__(self):
        self.load_weights()
        output = self.export_onix()
        self.check_onnx_model()
        self.test_onnx_model()

    def load_weights(self):
        self.model.load_state_dict(torch.load(self.weights_file)) if self.device == "cuda" else \
            self.model.load_state_dict(torch.load(self.weights_file, map_location='cpu'))
        self.model.eval()

    def export_onix(self):
        x = torch.randn(1, 3, 256, 256, requires_grad=True, device=self.device)
        y = self.model(x)
        torch.onnx.export(self.model, x, self.onnx_file, export_params=True, opset_version=11,
                          do_constant_folding=False, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
        return y

    def check_onnx_model(self):
        model = onnx.load(self.onnx_file)
        onnx.checker.check_model(model)
        print("Check Successfull!")

    def test_onnx_model(self, y=None):
        ort_session = onnxruntime.InferenceSession(self.onnx_file)
        x = torch.randn(1, 3, 256, 256, requires_grad=True)

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
        if y: np.testing.assert_allclose(y[0], ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")


class OnnxToProtoBuf(object):
    def __init__(self):
        self.onnx_file = '../models/model.onnx'
        self.pb_file = '../models/saved_model.pb'
        self.cpkt_file = '../models/model.pth'
        self.pbtext_file = '../models/model.pbtxt'
        self.dir = '../models'
        self.tf_rep = None
        self.sess = tf.Session()

    def export_frozen_graph(self):
        # Freeze your model first.
        # tf.train.write_graph(graph_or_graph_def=self.tf_rep.graph,
        #                      logdir=self.dir,
        #                      name=self.pbtext_file,
        #                      as_text=True)
        freeze_graph.freeze_graph(input_graph=self.pb_file,
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint='',
                                  output_node_names='cnn/output',
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph=self.dir,
                                  clear_devices=True,
                                  initializer_nodes='',
                                )
        exit()
        # graph = tf.get_default_graph()
        # input_graph_def = graph.as_graph_def()
        # output_node_names = ['cnn/output']
        #
        # output_graph_def = graph_util.convert_variables_to_constants(
        #     self.sess, input_graph_def, output_node_names)
        #
        # with tf.gfile.GFile(self.pb_file, 'wb') as f:
        #     f.write(output_graph_def.SerializeToString())

    def export_graph(self):
        onnx_model = onnx.load(self.onnx_file)
        self.tf_rep = prepare(onnx_model, strict=False)
        self.tf_rep.export_graph(self.pb_file)
        self.test_pb_file()

    def load_pb(self):
        with tf.compat.v2.io.gfile.GFile(self.pb_file, 'rb') as f:
            graph_def = tf.compat.v2.GraphDef()
            try:
                graph_def.ParseFromString(f.read())
            except Exception as e:
                print(traceback.print_exc(e))
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph

    def test_pb_file(self):
        try:
            tf_graph = self.load_pb()
        except:
            tf_graph = self.tf_rep

        sess = tf.compat.v1.Session(graph=tf_graph)

        output_tensor = tf_graph.get_tensor_by_name('Sigmoid:0')
        input_tensor = tf_graph.get_tensor_by_name('input:0')

        dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True, device=self.device)
        output = sess.run(output_tensor, feed_dict={input_tensor: dummy_input})
        print(output)


if __name__ == "__main__":
    handler = OnnxToProtoBuf()
    handler.export_frozen_graph()


