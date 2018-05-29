package me.yzhi.mxnet.example.infer;

import me.yzhi.mxnet.example.infer.seq2seq.AttnDecoder;
import me.yzhi.mxnet.example.infer.seq2seq.Encoder;
import me.yzhi.mxnet.example.infer.seq2seq.Lang;
import org.apache.mxnet.Context;
import org.apache.mxnet.DType;
import org.apache.mxnet.NDArray;
import org.apache.mxnet.Shape;
import scala.Tuple2;
import scala.Tuple3;

public class Seq2SeqInference {
  public static void main(String[] args) {
    if (args.length == 0) {
      System.out.println("Please specify the model dir, e.g., MXSeq2Seq/gluon/symbols");
      return;
    }

    final String modelPath = args[0];

    final Lang inputLang = new Lang(new String[]{"SOS", "EOS", "j", "ai", "ans", ".", "je", "vais", "bien", "ca", "va"});
    final Lang outputLang = new Lang(new String[]{"SOS", "EOS", "i", "m", ".", "ok"});

    final int SOS_TOKEN = 0;
    final int EOS_TOKEN = 1;

    final int MAX_LENGTH = 10;
    final int HIDDEN_SIZE = 256;
    final int NUM_LAYERS = 1;

    final int INPUT_LANG_NWORDS = 11;
    final int OUTPUT_LANG_NWORDS = 6;

    float input[] = inputLang.getInput("je vais bien .");
    NDArray modelInput = NDArray.array(input, Shape.create(1, input.length), Context.cpu(0));

    Encoder encoder = new Encoder(modelPath, INPUT_LANG_NWORDS,  HIDDEN_SIZE, NUM_LAYERS);
    AttnDecoder decoder = new AttnDecoder(modelPath, HIDDEN_SIZE, OUTPUT_LANG_NWORDS, NUM_LAYERS);
    Tuple2<NDArray, NDArray> encoderOut = encoder.predict(modelInput, encoder.initHidden());

    NDArray encoderOutputs = encoderOut._1;
    NDArray encoderHidden = encoderOut._2;

    if (input.length < MAX_LENGTH) {
//      HashMap<String, Object> kwargs = new HashMap<>();
//      kwargs.put("dim", 0);
//      NDArray mergedOutputs = NDArray.genericNDArrayFunctionInvoke("stack",
//          JavaConverters.asScalaIteratorConverter(outputs.iterator()).asScala().toSeq(),
//          new ScalaConverter().convert(kwargs)).get();

      encoderOutputs = NDArray.concat(
          NDArray.flatten(encoderOutputs),
          NDArray.zeros(Shape.create(MAX_LENGTH-input.length, HIDDEN_SIZE), Context.cpu(0), DType.Float32()),
          2, 0).get();
    } else {
      encoderOutputs = NDArray.flatten(encoderOutputs).get();
    }

    NDArray decoderInput = NDArray.array(new float[]{SOS_TOKEN}, Shape.create(1), Context.cpu(0));
    NDArray decoderHidden = encoderHidden;

    StringBuilder outputs = new StringBuilder();

    for (int i = 0; i < 50; ++i) {
      Tuple3<NDArray, NDArray, NDArray> decoderOutputs =
          decoder.predict(decoderInput, decoderHidden, encoderOutputs);
      NDArray decoderOutput = decoderOutputs._1();
      decoderHidden = decoderOutputs._2();

      NDArray topi = NDArray.argmax(decoderOutput, 1).get();
      String token = outputLang.getString((int)topi.toScalar());
      if (token.equals(outputLang.getString(EOS_TOKEN))) {
        break;
      }
      outputs.append(token).append(" ");
      decoderInput = NDArray.array(topi.toArray(), Shape.create(1), Context.cpu(0));
    }

    System.out.println(outputs.toString().trim());
  }
}
