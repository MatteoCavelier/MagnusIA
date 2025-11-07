# Guide d'installation

## Prérequis

Listez ici les prérequis système ou logiciels :

* Python 3.10+
* Git
* Node.js

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/MatteoCavelier/MagnusIA
cd MagnusIA
```
### 2. Installez les dépendances python
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

### 3. Ajoutez une lib (optionnel)

```shell
  uv add exemple_lib
```

### 4. Installation des dépendances web
```shell
 cd app
 npm i
```