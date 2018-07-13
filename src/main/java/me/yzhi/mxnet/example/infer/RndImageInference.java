package me.yzhi.mxnet.example.infer;

import org.apache.mxnet.*;
import org.apache.mxnet.module.Module;
import scala.Tuple3;
import scala.collection.immutable.Map;

import java.io.File;
import java.util.Arrays;

public class RndImageInference {
  public static void main(String[] args) {
    final String modelName = "resnet50_v1";
    final Shape dataShape = Shape.create(1, 3, 224, 224);

    Tuple3<Symbol, Map<String, NDArray>, Map<String, NDArray>> model =
        Model.loadCheckpoint("symbol" + File.separator + modelName, 0);

    Symbol symbol = model._1();
    Map<String, NDArray> argParams = model._2();
    Map<String, NDArray> auxParams = model._3();

    Module module = new Module.Builder(symbol).setContext(Context.cpu(0)).build();
    module.bind(false, false, false,
        new DataDesc("data", dataShape, DType.Float32(), "NCHW"));
    module.setParams(argParams, auxParams, true, true, false);

    float[] argmax = NDArrayCollector.auto().withScope(new scala.runtime.AbstractFunction0<float[]>() {
      @Override
      public float[] apply() {
        NDArray data = NDArray.normal(ScalaConverter.convert(0, 1, dataShape)).get();
        DataBatch input = new DataBatch.Builder().setData(data).build();

        module.forward(input, false);
        NDArray pred = module.getOutputsMerged().apply(0);
        return NDArray.argmax(ScalaConverter.convert(pred, 1)).toArray();
      }
    });

    System.out.println(Arrays.toString(argmax));
  }
}
