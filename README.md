

# GPT2 Detokenization
This repository contains code for the paper `Weight-based Analysis of Detokenization in Language Models:
Understanding the First Stage of Inference Without Inference` by Go Kamoda, Tatsuro Inaba, Keito Kudo, Benjamin Heinzerling, Keisuke Sakaguchi and Kentaro Inui.

## Create environment

```
uv sync --no-dev
uv sync --dev --no-build-isolation
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

## Frequency
```
python src/frequency.py \
    openwebtext \
    --tokenizer gpt2 \
    --mode bitoken \
    --num-workers 21 \
```

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