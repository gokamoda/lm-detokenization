

# GPT2 Detokenization
This repository contains code for the paper  
[Weight-based Analysis of Detokenization in Language Models:
Understanding the First Stage of Inference Without Inference](https://arxiv.org/abs/2501.15754)  
by Go Kamoda, Benjamin Heinzerling, Tatsuro Inaba, Keito Kudo, Keisuke Sakaguchi and Kentaro Inui.

## Create environment

```
make install
```

## Visualizations

- $T^{ee}$
    ```
    python src/visualize.py \
        --mode l0_tee \
        --heads 1 7 \
        --n-samples 100
    ```

- $T^p$
    ```
    python src/visualize.py \
        --mode l0_tp \
        --heads 1 7
    ```

- $T^{pp}$
    ```
    python src/visualize.py \
        --mode l0_tpp \
        --heads 1 7 \
        --pos-i 500
    ```

- $T^p + T^{pp}$
    ```
    python src/visualize.py \
        --mode l0_tp_tpp \
        --heads 1 7 \
        --pos-i 500
    ```

- $T^{e}$
    ```
    python src/visualize.py \
        --mode l0_te \
        --heads 1 7
    ```

- Undertrained pos emb
    ```
    python src/visualize.py \
        --mode l0_tpp_undertrained \
        --heads 0
    ```


## Frequency
```
python src/frequency.py \
    openwebtext \
    --tokenizer gpt2 \
    --mode bitoken \
    --num-workers 21 \
```

## OpenWebText sample
The empirical experiments below use a fixed sample of OpenWebText.
Make it once before running them:
```
bash scripts/make_openwebtext_sample.sh
```
This streams the whole dataset once (about 24GB of download, ~30 min) and saves the 10,000 documents with the smallest sha256 of the text to `outputs/data/openwebtext_sample.jsonl`.
The experiments use the first N documents of this file.

## Emprirical Experiments

- $T^{p} + T^{pp}$
    ```
    python src/empirical.py \
        --mode vs_tptpp \
        --func main
    ```
    ```
    python src/empirical.py \
        --mode vs_tptpp \
        --func vis
    ```
- 6 Terms importance
    ```
    python src/empirical.py \
        --mode six \
        --func main
    ```
    ```
    python src/empirical.py \
        --mode six \
        --func vis
    ```