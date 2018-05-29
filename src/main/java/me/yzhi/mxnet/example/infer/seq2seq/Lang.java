package me.yzhi.mxnet.example.infer.seq2seq;

import java.util.HashMap;
import java.util.Map;

public class Lang {
  private String[] index2word;
  private Map<String, Integer> word2index = new HashMap<>();

  public Lang(String[] index2word) {
    this.index2word = index2word;
    for (int i = 0;  i < index2word.length; i++) {
      word2index.put(index2word[i], i);
    }
  }

  public String getString(int index) {
    if (index >= index2word.length) {
      throw new IllegalArgumentException("Index=" + index + " > Lang.length=" + index2word.length);
    }
    return index2word[index];
  }

  public int getIndex(String word) {
    if (!word2index.containsKey(word)) {
      throw new IllegalArgumentException("word " + word + " not exist");
    }
    return word2index.get(word);
  }
}
