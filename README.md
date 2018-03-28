## Tensorflow High Level API Example

This repository includes an example project with tensorflow high level API (based on version 1.6). Here are some features.

* Adopt tensorflow high level API to train and evaluate binary/multi-classes models.

* Directly use tensors as input of model function without specifying feature columns.

* Use tensorflow's Data Set pipeline followed by scikit learn classifiers.

Before running the example project, generate the data files (e.g., epsilon_normalized) as follows.

```
# Download data from LIBSVM Data page
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2

bzip2 -d epsilon_normalized.bz2

split -l 20000 epsilon_normalized epsilon_normalized_

mkdir epsilon_normalized_pieces

mv epsilon_normalized_* epsilon_normalized_pieces/

cd epsilon_normalized_pieces

find . | xargs -I {} gzip {}
```

Generate a9a data files similarly (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a).

Here are some usage options.

```
python -m tf_highlevelapi_example.main --task read_data --input_files ${DATA_DIR}/a9a_pieces/*.gz --feature_size 123 --label_size 2
```

```
python -m tf_highlevelapi_example.main --task train --input_files ${DATA_DIR}/a9a_pieces/*.gz --model_dir ${MODEL_DIR} --feature_size 123 --label_size 2

python -m tf_highlevelapi_example.main --task eval --input_files ${DATA_DIR}/a9a_pieces/*.gz --model_dir ${MODEL_DIR} --feature_size 123 --label_size 2

python -m tf_highlevelapi_example.main --task export --model_dir ${MODEL_DIR} --export_dir ${EXPORT_DIR}
```

```
python -m tf_highlevelapi_example.main --task train --gen_synthesis_data --input_files ${DATA_DIR}/a9a_pieces/*.gz --model_dir ${MODEL_DIR} --feature_size 15 --label_size 2

python -m tf_highlevelapi_example.main --task eval --gen_synthesis_data --input_files ${DATA_DIR}/a9a_pieces/*.gz --model_dir ${MODEL_DIR} --feature_size 15 --label_size 2
```

```
python -m tf_highlevelapi_example.main --task train_sklearn_lr  --input_files ${DATA_DIR}/epsilon_normalized_pieces/*.gz --model_dir ${MODEL_DIR} --feature_size 2000 --label_size 2 --prefetch_size 128 --batch_size 32

python -m tf_highlevelapi_example.main --task eval_sklearn_lr  --input_files ${DATA_DIR}/epsilon_normalized_pieces/*.gz --model_dir ${MODEL_DIR} --feature_size 2000 --label_size 2 --prefetch_size 128 --batch_size 32
```
