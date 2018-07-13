package me.yzhi.mxnet.example.infer.seq2seq;

import me.yzhi.mxnet.example.infer.ScalaConverter;
import org.apache.mxnet.*;
import org.apache.mxnet.module.Module;
import scala.Tuple2;
import scala.Tuple3;
import scala.collection.immutable.Map;

import java.io.File;

public class AttnDecoder {
  private NDArray embeddingWeight;
  private Module attn;
  private Module attnCombine;
  private GRUCell gru;
  private Module out;

  private final int hiddenSize;
  private final int outputSize;
  private final int numLayers;
  private final float dropoutProb;

  public AttnDecoder(String modelDir, int hiddenSize, int outputSize, int numLayers) {
    this(modelDir, hiddenSize, outputSize, numLayers, 0.1f);
  }

  public AttnDecoder(String modelDir, int hiddenSize, int outputSize, int numLayers, float dropoutProb) {
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;
    this.numLayers = numLayers;
    this.dropoutProb = dropoutProb;

    {
      Tuple3<Symbol, Map<String, NDArray>, Map<String, NDArray>> model =
          Model.loadCheckpoint(modelDir + File.separator + "decoder_embedding", 0);
      Map<String, NDArray> argParams = model._2();
      embeddingWeight = argParams.get("attndecoderrnn0_embedding0_weight").get();
    }
    {
      Tuple3<Symbol, Map<String, NDArray>, Map<String, NDArray>> model =
          Model.loadCheckpoint(modelDir + File.separator + "decoder_attn", 0);
      attn = new Module.Builder(model._1()).setContext(Context.cpu(0)).setDataNames("data").build();
      attn.bind(false, false, false,
          new DataDesc("data", Shape.create(1, hiddenSize*2), DType.Float32(), "NC"));
      attn.setParams(model._2(), model._3(), true, true, false);
    }
    {
      Tuple3<Symbol, Map<String, NDArray>, Map<String, NDArray>> model =
          Model.loadCheckpoint(modelDir + File.separator + "decoder_attn_combine", 0);
      attnCombine = new Module.Builder(model._1()).setContext(Context.cpu(0)).setDataNames("data").build();
      attnCombine.bind(false, false, false,
          new DataDesc("data", Shape.create(1, hiddenSize*2), DType.Float32(), "NC"));
      attnCombine.setParams(model._2(), model._3(), true, true, false);
    }

    gru = new GRUCell(modelDir + File.separator + "decoder_gru", hiddenSize);

    {
      Tuple3<Symbol, Map<String, NDArray>, Map<String, NDArray>> model =
          Model.loadCheckpoint(modelDir + File.separator + "decoder_out", 0);
      out = new Module.Builder(model._1()).setContext(Context.cpu(0)).setDataNames("data").build();
      out.bind(false, false, false,
          new DataDesc("data", Shape.create(1, 1, hiddenSize), DType.Float32(), "NTC"));
      out.setParams(model._2(), model._3(), true, true, false);
    }
  }

  public Tuple3<NDArray, NDArray, NDArray> predict(NDArray input, NDArray hidden, NDArray encoderOutputs) {
    NDArray embedding = NDArray.Embedding(ScalaConverter.convert(input, embeddingWeight, outputSize, hiddenSize, DType.Float32())).get();
    if (dropoutProb > 0) {
      embedding = NDArray.Dropout(ScalaConverter.convert(embedding, dropoutProb)).get();
    }

    NDArray attnWeights = attn.predict(
        new DataBatch.Builder().setData(
            NDArray.concat(ScalaConverter.convert(embedding, NDArray.flatten(ScalaConverter.convert(hidden)), 2, 1)).get()).build())
        .apply(0);
    attnWeights = NDArray.softmax(ScalaConverter.convert(attnWeights)).get();

    NDArray attnApplied = NDArray.batch_dot(ScalaConverter.convert(
        NDArray.expand_dims(ScalaConverter.convert(attnWeights, 0)).get(),
        NDArray.expand_dims(ScalaConverter.convert(encoderOutputs, 0)).get()
    )).get();


    NDArray output = NDArray.concat(ScalaConverter.convert(
        NDArray.flatten(ScalaConverter.convert(embedding)).get(),
        NDArray.flatten(ScalaConverter.convert(attnApplied)).get(), 2, 1)).get();

    output = attnCombine.predict(new DataBatch.Builder().setData(output).build()).apply(0);
    output = NDArray.expand_dims(ScalaConverter.convert(output, 0)).get();

    for (int i = 0; i < numLayers; ++i) {
      output = NDArray.relu(ScalaConverter.convert(output)).get();
      Tuple2<NDArray, NDArray> gruOutput = gru.predict(output, hidden);
      output = gruOutput._1;
      hidden = gruOutput._2;
    }

    output = out.predict(new DataBatch.Builder().setData(output).build()).apply(0);

    return new Tuple3<>(output, hidden, attnWeights);
  }
}
