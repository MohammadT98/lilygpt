@echo off
REM Run on all .ly under data\raw and write final outputs to data\normalized
py -3 -m lilynorm.cli run --input data\raw --out-root data %*