# CDC BRFSS 2021 public-use file

The raw file is not committed. Download it once (≈82 MB zip, ≈1 GB XPT):

```bash
curl -L -o data/raw/LLCP2021XPT.zip https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip
unzip data/raw/LLCP2021XPT.zip -d data/raw
mv "data/raw/LLCP2021.XPT " data/raw/LLCP2021.XPT   # CDC's archive names the file with a trailing space
```

Source page: https://www.cdc.gov/brfss/annual_data/annual_2021.html
(zip SHA-256 used for the committed results: `e8a6e290cdcb6bff82d8f37dff6c3f3e435d428191eb529a716cf9f5316aa621`).

`python scripts/prepare_data.py` writes the complete-case analytic table to
`data/processed/brfss_2021_multilabel.parquet` (431,843 rows). `scripts/reproduce.py`
builds it automatically if it is missing. Florida is absent from the 2021 national file.
