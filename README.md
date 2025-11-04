# Magnus IA


## Installation

```shell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```shell
  uv python install
```
```shell
    uv venv
```
```shell
  .venv\Scripts\activate
```
```shell
  uv sync
```

### Ajouter une lib
```shell
  uv add exemple_lib
```

## Lancer le Back

```shell
    mlflow models serve - m "runs:/URI_DU_MODEL/model" - p 1234 - -no - conda
```

### Lancer l'interface web

```shell
  mlflow ui
```

## Predict
### Exemple
```shell
curl -X POST -H "Content-Type:application/json" \
-d '{
      "columns": ["turns","white_rating","black_rating","opening_ply","inc_1","inc_2","eco_A00","eco_B01","opn_RuyLopez","opn_SicilianDefense"],
      "data": [[35,1500,1450,5,0,1,0,1,1,0]]
    }' \
http://127.0.0.1:1234/invocations
```
