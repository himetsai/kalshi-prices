# Are Kalshi prices probabilities?

An independent replication of Kalshi Research's August 2026 calibration study on public
Kalshi data.

Writeup: [Are Kalshi prices probabilities?](http://himetsai.com/blog/kalshi)

## Running

```bash
pip install -r requirements.txt
python collect.py markets --start 2026-04-01 --end 2026-07-01
python collect.py candles --interval 60 --horizon-days 7
python analysis.py
python movement.py
python figures.py
```

`collect.py` pulls settled markets and hourly candles from the public Kalshi API into
`data/`. `analysis.py` prints the calibration
tables, `movement.py` the martingale test, and `figures.py` produces the figures in `figures/`.
