
## SuEIR model for forecasting confirmed cases, deaths, and hospitaliztions at nation, state, and county levels.

### How to get forecast results of confirmed cases, deaths at different levels?

Step 1: Run ```validation.py``` to generate validation file for selecting hyperparameters, e.g.,
```python
python validation.py --END_DATE 2020-07-07 --VAL_END_DATE 2020-07-14  --level state
```

Step 2: Generate prediction results by running ```generate_predictions.py```, e.g.,
```python
python generate_predictions.py --END_DATE 2020-07-07 --VAL_END_DATE 2020-07-14 --level state
```
(before runing ```generate_predictions.py```, one should make sure the corresponding validation file, i.e., with the same ```END_DATE```, ```VAL_END_DATE```, and ```level```, has already be generated)


### Arguments:
*```END_DATE```: end date for training data

*```VAL_END_DATE```: end date for validation data

*```level```: can be state, nation, or county

*```state```: validation for one specific state (```level``` should be set as state)

*```nation```: validation for one specific country (```level``` should be set as nation)

*```dataset```: select which data source to use (can be NYtimes and JHU)

