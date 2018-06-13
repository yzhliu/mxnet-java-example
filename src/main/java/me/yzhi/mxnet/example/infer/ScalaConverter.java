package me.yzhi.mxnet.example.infer;

import scala.collection.JavaConverters;
import scala.collection.JavaConverters$;
import scala.collection.Seq;
import scala.collection.immutable.Map;

import java.util.Arrays;

public class ScalaConverter {
  public static <K, V> Map<K, V> convert(java.util.Map<K, V> m) {
    return JavaConverters$.MODULE$.mapAsScalaMapConverter(m).asScala().toMap(
        scala.Predef$.MODULE$.<scala.Tuple2<K, V>>conforms()
    );
  }

  public static Seq<Object> convert(Object... args) {
    return JavaConverters.asScalaIteratorConverter(Arrays.asList(args).iterator()).asScala().toSeq();
  }
}
