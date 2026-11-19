# Running the GPU pipeline (Google Colab or any CUDA machine)

All scripts are in `src/` and import each other from the same directory. On Colab, upload the contents of `src/` to `/content` (or clone this repository and `cd src`), choose a GPU runtime, and set

```
MS_OUT   directory for results and logs (e.g. a mounted Google Drive folder)
WIDE_DIR directory for the generated training data (about 100 MB)
BENCH_DIR directory for the benchmark sets
```

## 1. Training data, training and evaluation

```
python wide_data.py val                                        # small check: regenerated coefficients vs reference values
python wide_data.py train val_hi train_hi val_xhi train_xhi    # all splits, k from 32 to 2048
python wide_train.py local_ctx_hi_torch                        # model trained for k <= 512
python wide_train.py local_ctx_hi_xhi_torch                    # model trained for k <= 2048
python wide_eval.py                                            # far-field errors, k = 32 ... 2048
```

`wide_data.py` stops with an error if the regenerated chart coefficients differ from reference values of the original data by more than 1e-4 (relative).

## 2. Beyond the training band, against hybrid numerical-asymptotic methods

```
NSHAPE=2 OUT=band_quick.json python band_eval.py 512,1024 1   # quick check on two shapes
python band_eval.py 512,1024,2048 1
```

## 3. Neural-operator baselines

```
pip install neuraloperator
python bench_data_colab.py test_id test_ood val train          # benchmark sets from seeds
MODELS=FNO VARIANTS=phase EPOCHS=20 OUT=sota_quick.json python run_sota_bench_colab.py   # quick check
VARIANTS=phase python run_sota_bench_colab.py
```

Every script prints its progress line by line and writes JSON results to `$MS_OUT/results`.
