
## SuEIR model for forecasting confirmed cases, deaths, and hospitaliztions at nation, state, and county levels.

### How to get forecast results of confirmed cases, deaths at different levels?

Step 1: Run ```validation.py``` to generate validation file for selecting hyperparameters, e.g.,
```python
python validation.py --END_DATE 2020-07-07 --VAL_END_DATE 2020-07-14  --dataset NYtimes --level state
```

Step 2: Generate prediction results by running ```generate_predictions.py```, e.g.,
```python
python generate_predictions.py --END_DATE 2020-07-07 --VAL_END_DATE 2020-07-14 --dataset NYtimes --level state
```
Before runing ```generate_predictions.py```, one should make sure the corresponding validation file, i.e., with the same ```END_DATE```, ```VAL_END_DATE```, ```dataset```, and ```level```, has already be generated.


### Arguments:
*```END_DATE```: end date for training data

*```VAL_END_DATE```: end date for validation data

*```level```: can be state, nation, or county, default: state

*```state```: validation/prediction for one specific state (```level``` should be set as state), default: all states in the US 

*```nation```: validation/prediction for one specific country (```level``` should be set as nation), default: 26 countries in the world

*```dataset```: select which data source to use (can be NYtimes and JHU), default: NYtimes data

