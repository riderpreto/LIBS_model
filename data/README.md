# Estrutura do Diretório de Dados

Para executar o pipeline a partir dos dados brutos, organize os arquivos `.esf` na pasta `data/raw/` seguindo a hierarquia abaixo:

```
data/
├── raw/
│   ├── Adulterado_00porcento/
│   │   ├── UV/
│   │   │   ├── amostra_a___1.esf
│   │   │   └── ...
│   │   └── VIS/
│   │       ├── amostra_a___1.esf
│   │       └── ...
│   ├── Adulterado_10porcento/
│   │   ├── UV/
│   │   └── VIS/
│   ├── Adulterado_25porcento/
│   │   ├── UV/
│   │   └── VIS/
│   └── Adulterado_50porcento/
│       ├── UV/
│       └── VIS/
└── processed/
    ├── df_limpo_vis.parquet   (gerado pelo Caderno 01)
    └── df_limpo_uv.parquet    (gerado pelo Caderno 01)
```

> **Nota:** Arquivos `.esf` e matrizes binárias `.parquet` são automaticamente ignorados pelo `.gitignore` para manter o repositório leve.
