import numpy as np
import mxnet as mx
from mxnet.gluon.model_zoo.vision import get_model

def end2end_benchmark(model, batch_size):
    num_classes = 1000
    image_shape = (3, 224, 224)
    data_shape = (batch_size,) + image_shape
    mx_data = mx.nd.normal(0, 1, shape=data_shape)

    block = get_model(model, pretrained=True)
    block.hybridize()
    block(mx_data)

    block.export("symbol/" + model)
    sym, arg_params, aux_params = mx.model.load_checkpoint("symbol/" + model, 0)

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', data_shape)])
    mod.set_params(arg_params, aux_params, allow_missing=True)

    input = mx.io.DataBatch([mx_data,])
    mod.forward(input, is_train=False)
    pred = mx.nd.softmax(mod.get_outputs()[0])
    print(mx.nd.argmax(pred, axis=1))

if __name__ == '__main__':
    end2end_benchmark('resnet50_v1', 1)