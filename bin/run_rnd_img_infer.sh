#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)

mkdir -p symbol

echo "Run python code ..."
python $CURR_DIR/rnd_img_infer.py

echo "Run Java code ..."
CLASSPATH=$CLASSPATH:$PROJ_DIR/target/*:$PROJ_DIR/target/classes/lib/*
java -Xmx128m -cp $CLASSPATH \
  -Dlog4j.configuration=file://$PROJ_DIR/conf/log4j.properties \
  -XX:+PrintGCDetails -XX:+PrintHeapAtGC -XX:+PrintGCDateStamps -XX:+PrintTenuringDistribution \
  -verbose:gc -Xloggc:./gc.log -XX:+PrintGCTimeStamps \
  me.yzhi.mxnet.example.infer.RndImageInference
