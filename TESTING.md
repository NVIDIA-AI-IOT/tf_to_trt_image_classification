**instruction:**

1. follow setup in [README.md](README.md)

**success if:**

1. executing ``import tensorflow`` in python interpreter succeeds without error
2. executing ``import uff`` in python interpreter succeeds without error
3. the build command ``cd build; make ..`` succeeds without error

**instruction:**

1. execute the following from root directory of project

    source scripts/download_models.sh
    source scripts/download_images.sh

**success if:**

1. each of the models in ``scripts/download_models.sh`` is downloaded and extracted to ``data/checkpoints``
2. each of the images in ``scripts/download_images.sh`` is downloaded under ``data/images``

**instruction:**

1. reboot the Jetson TX2
2. execute the following from root directory of project

    python scripts/models_to_frozen_graphs.py

**success if:**

1. the script completes without error
2. a frozen graph is created under ``data/frozen_graphs`` for each model in ``scripts/model_meta.py`` that does not contain ``'exclude': true``

**instruction:**

1. execute the following from root directory of project

    python scripts/frozen_graphs_to_plans.py

**success if:**

1. the script finished without error
2. a serialized engine (.plan) file is created under ``data/plans`` for each model in ``scripts/model_meta.py`` that does not contain ``'exclude': true``

**instruction:**

1. execute the following to ensure the Jetson TX2 is in max-P mode
    
    sudo nvpmodel -m 3
 
2. execute the following from root directory of project

    python scripts/test_trt.py

**success if:**

1. the script finished without error
2. the file ``data/test_output_trt.txt`` is created
3. timing entries exist in ``data/test_output_trt.txt`` for each model in ``scripts/model_meta.py`` that does not contain ``'exclude': true``
4. the timing entries are approximately equal (+/- 10%) to those found in [README.md](README.md)

**instruction:**

1. reboot the Jetson TX2
2. execute the following from the root directory of the project

    python scripts/test_tf.py

**success if:**

1. the script finished without error
2. the file ``data/test_output_tf.txt`` is created
3. timing entries exist in ``data/test_output_tf.txt`` for each model in ``scripts/model_meta.py`` that does not contain ``'exclude': true``
4. the timing entries are approximately equal (+/- 10%) to those found in [README.md](README.md)
