# MXNet Java Examples
Examples for Using MXNet in Java

First clone the MXNet-Scala frontend with Java-friendly APIs:

```bash
git clone --recursive -b java https://github.com/yzhliu/mxnet.git
```

Install Scala frontend to local Maven repository:

```bash
cd mxnet
# to config, refer to https://mxnet.incubator.apache.org/install/index.html
make -j4
make scalapkg
make scalainstall
# optional: if you already pip installed mxnet, can skip following
cd python
python setup.py install --user --force
```

Now clone the Java example:

```bash
git clone https://github.com/yzhliu/mxnet-java-example.git
```

Compile and run:

```bash
cd mxnet-java-example
mvn clean package
bin/run_rnd_img_infer.sh
``` 