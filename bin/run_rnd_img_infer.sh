#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)

mkdir -p $CURR_DIR/symbol

echo "Run python code ..."
python $CURR_DIR/rnd_img_infer.py

echo "Run Java code ..."
CLASSPATH=$CLASSPATH:$PROJ_DIR/target/*:$PROJ_DIR/target/classes/lib/*
java -Xmx4m -cp $CLASSPATH \
  -Dlog4j.configuration=file://$PROJ_DIR/conf/log4j.properties \
  me.yzhi.mxnet.example.infer.RndImageInference
