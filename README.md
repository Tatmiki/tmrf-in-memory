#### Comando para baixar as dependências do projeto assumindo que você está no root do projeto:
- `pip install -e .` 

#### Script para baixar os datasets necessários e colocá-los nos padrões dos scripts de extração de características:
- `python ./scripts/download_data.py`

```
git clone esse_repositorio
git submodule update --init --recursive
pip install -r src/python/external/pyfeats/requirements.txt
```