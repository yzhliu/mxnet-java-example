package me.yzhi.mxnet.debug;

import org.apache.mxnet.Context;
import org.apache.mxnet.DType;
import org.apache.mxnet.NDArray;
import org.apache.mxnet.Shape;

public class EmitNDArray {
  public static void main(String[] args) {
    for (int i = 0; i < 10; ++i) {
      Thread thread = new Thread() {
        public void run() {
          while (true) {
            NDArray arr = NDArray.empty(Shape.create(100, 100, 2), Context.cpu(0), DType.Float32());
            arr.toArray();
            try {
              Thread.sleep(1000);
            } catch (InterruptedException e) {
              e.printStackTrace();
            }
          }
        }
      };
      thread.start();
    }
  }
}
