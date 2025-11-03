# Magnus IA


## Installation

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv python install

uv venv

.venv\Scripts\activate

uv sync
```

### Ajouter une lib
```shell
uv add exemple_lib
```
