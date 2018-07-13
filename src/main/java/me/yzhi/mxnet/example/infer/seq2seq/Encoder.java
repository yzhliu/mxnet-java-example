package me.yzhi.mxnet.example.infer.seq2seq;

import me.yzhi.mxnet.example.infer.ScalaConverter;
import org.apache.mxnet.*;
import scala.Tuple2;
import scala.Tuple3;
import scala.collection.immutable.Map;

import java.io.File;

public class Encoder {
  private NDArray embeddingWeight;
  private GRUCell gru;

  private final int numLayers;
  private final int numWords;
  private final int hiddenSize;

  private final String dtype = "float32";
  private final String layout = "TNC";

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

    gru = new GRUCell(modelDir + File.separator + "encoder_gru", hiddenSize);
  }

  public Tuple2<NDArray, NDArray> predict(NDArray input, NDArray hidden) {
    NDArray embedding = NDArray.Embedding(ScalaConverter.convert(input, embeddingWeight, numWords, hiddenSize, dtype)).get();
    Tuple2<NDArray, NDArray> output = new Tuple2<>(NDArray.swapaxes(ScalaConverter.convert(embedding, 0, 1)).get(), hidden);
    for (int i = 0; i < numLayers; ++i) {
      output = gru.predict(output._1, output._2);
    }
    return output;
  }

  public NDArray initHidden() {
    return NDArray.zeros(Shape.create(1, hiddenSize), Context.cpu(0), DType.Float32());
  }
}
