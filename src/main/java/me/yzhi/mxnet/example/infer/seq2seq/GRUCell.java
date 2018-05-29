package me.yzhi.mxnet.example.infer.seq2seq;

import org.apache.mxnet.*;
import org.apache.mxnet.module.Module;
import scala.Tuple2;
import scala.Tuple3;
import scala.collection.immutable.Map;

public class GRUCell {
  private Module gru;
  private final String layout = "TNC";

  public GRUCell(String modelFile, int hiddenSize) {
    this(modelFile, hiddenSize, "data0", "data1", Context.cpu(0));
  }

  public GRUCell(String modelFile, int hiddenSize,
                 String inputName, String stateName, Context ctx) {
    Tuple3<Symbol, Map<String, NDArray>, Map<String, NDArray>> model = Model.loadCheckpoint(modelFile, 0);

    Symbol symbol = model._1();
    Map<String, NDArray> argParams = model._2();
    Map<String, NDArray> auxParams = model._3();

    gru = new Module.Builder(symbol).setContext(ctx).setDataNames(inputName, stateName).build();
    gru.bind(false, false, false,
        new DataDesc(inputName, Shape.create(1, hiddenSize), DType.Float32(), "NC"),
        new DataDesc(stateName, Shape.create(1, hiddenSize), DType.Float32(), "NC"));

    gru.setParams(argParams, auxParams, true, true, false);
  }

  public Tuple2<NDArray, NDArray> predict(NDArray inputs, NDArray state) {
    int axis = layout.indexOf('T');
    int length = inputs.shape().get(axis);

    Object[] outputArr = new Object[length+1];
    for (int i = 0; i < length; ++i) {
      NDArray output = gru.predict(new DataBatch.Builder().setData(inputs.at(i), state).build()).apply(0);
      outputArr[i] = output;
      state = output;
    }

    // stack outputs
//    HashMap<String, Object> kwargs = new HashMap<>();
//    kwargs.put("axis", axis);
//    NDArray mergedOutputs = NDArray.genericNDArrayFunctionInvoke("stack",
//        JavaConverters.asScalaIteratorConverter(outputs.iterator()).asScala().toSeq(),
//        new ScalaConverter().convert(kwargs)).get();
    outputArr[length] = 0;
    NDArray mergedOutputs = NDArray.stack(outputArr).get();

    return new Tuple2<>(mergedOutputs, state);
  }
}
