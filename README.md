

## Initial Setup
- Clone the repository
- Modify `pyproject.toml`.

## UV (if not installed)
- Install
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
- Path
    ```
    source .cargo/env
    ```

## Create environmen
- Create .venv
    ```
    uv sync
    ```
- Activate .venv
    ```
    . .venv/bin/activate
    ```
- Install pre-commit
    ```
    pre-commit install
    ```