package me.yzhi.mxnet.example.infer.seq2seq;

import org.apache.mxnet.*;
import org.apache.mxnet.module.Module;
import scala.Tuple3;
import scala.collection.immutable.Map;

import java.io.File;

public class Encoder {
  private NDArray embeddingWeight;
  private Module gru;

  private final int numLayers;
  private final int numWords;
  private final int hiddenSize;

  private final String dtype = "float32";

  public Encoder(String modelDir, int numWords, int hiddenSize, int numLayers) {
    this.numLayers = numLayers;
    this.numWords = numWords;
    this.hiddenSize = hiddenSize;

    {
      Tuple3<Symbol, Map<String, NDArray>, Map<String, NDArray>> model =
          Model.loadCheckpoint(modelDir + File.separator + "encoder_embedding", 0);
      Map<String, NDArray> argParams = model._2();
      embeddingWeight = argParams.get("encoderrnn0_embedding0_weight").get();
    }
    {
      Tuple3<Symbol, Map<String, NDArray>, Map<String, NDArray>> model =
          Model.loadCheckpoint(modelDir + File.separator + "encoder_gru", 0);

      Symbol symbol = model._1();
      Map<String, NDArray> argParams = model._2();
      Map<String, NDArray> auxParams = model._3();


      Module module = new Module.Builder(symbol).setContext(Context.cpu(0)).setDataNames("data0", "data1").build();
      module.bind(false, false, false,
          new DataDesc("data0", Shape.create(1, hiddenSize), DType.Float32(), "NC"),
          new DataDesc("data1", Shape.create(1, hiddenSize), DType.Float32(), "NC"));

      module.setParams(argParams, auxParams, true, true, false);
    }
  }

  public NDArray forward(NDArray input, NDArray hidden) {
    NDArray embedding = NDArray.Embedding(input, embeddingWeight, numWords, hiddenSize, dtype).get();
    return NDArray.swapaxes(embedding, 0, 1).get();
  }

  public NDArray initHidden() {
    return NDArray.zeros(Shape.create(1, 1, hiddenSize), Context.cpu(0), DType.Float32());
  }
}
