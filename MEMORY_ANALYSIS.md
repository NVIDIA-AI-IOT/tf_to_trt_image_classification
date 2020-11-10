## Instructions

clone project and dependencies

```bash
git clone --recursive https://github.com/NVIDIA-AI-IOT/tf_to_trt_image_classification
cd tf_to_trt_image_classification
```

create frozen graphs for models

```bash
source scripts/download_models.sh
python3 models_to_frozen_graphs.py
```

create engines for various models

```bash
./build_engines.sh
```

profile TensorRT for model


```bash
python3 run_engine.py data/engines/inception_v2_int8_bs1.engine
```

profile TensorFlow for model

```bash
python3 run_tf.py data/frozen_graphs/inception_v2.pb --allow_growth
```