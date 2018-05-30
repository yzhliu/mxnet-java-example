#!/bin/bash
CURR_DIR=$(cd $(dirname $0); pwd)
PROJ_DIR=$(cd $(dirname $0)/../; pwd)

CLASSPATH=$CLASSPATH:$PROJ_DIR/target/*:$PROJ_DIR/target/classes/lib/*
java -Xmx2G -cp $CLASSPATH \
  -Dlog4j.configuration=file://$PROJ_DIR/conf/log4j.properties \
  me.yzhi.mxnet.debug.EmitNDArray
