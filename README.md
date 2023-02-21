# rolling-forecast
Prediction of the next rolling 6-month revenue for each hotel based on booking history 


```
>>> python src/cli/forecast.py \
... -m 2 -y 2023 \
... -r 180000 200000 \
... -ht 'Resort Hotel'                          
INFO:root:Requested rolling revenue forecast for Resort Hotel starting from 02-2023
INFO:root:Parameters check passed successfully
INFO:root:Model and transformer was loaded successfully
INFO:root:Resort Hotel rolling 6-month revenue forecast
INFO:root:Actual   2022-12-31   180000.00
INFO:root:Actual   2023-01-31   200000.00
INFO:root:Forecast 2023-02-28   192170.55
INFO:root:Forecast 2023-03-31   245650.90
INFO:root:Forecast 2023-04-30   329826.22
INFO:root:Forecast 2023-05-31   454316.10
INFO:root:Forecast 2023-06-30   529041.16
INFO:root:Forecast 2023-07-31   548446.39
```