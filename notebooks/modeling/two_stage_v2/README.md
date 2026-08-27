# Two-stage ensemble v2

This directory is the package and execution root for the two-stage ensemble v2
notebook workflow.

## Environment contract

Run notebook and test commands from this directory:

```powershell
Set-Location notebooks/modeling/two_stage_v2
```

The workflow uses the Python 3.14 interpreter at
`C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe`.
Install the exact direct dependencies from `requirements-v2.txt` with that
interpreter before running the workflow.

The notebook reads `data/Prepared_data.parquet` relative to the project root
and writes generated outputs only to `data/two_stage_v2_artifacts`.

## Smoke test

Run the environment contract test from this directory:

```powershell
& 'C:\Users\savin\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_environment -v
```
