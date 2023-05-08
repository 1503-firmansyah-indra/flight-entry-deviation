# Experiment Configurations

## Sector 67Y
To generate the results for Sector 67Y, run the following.

```
python train_nn.py --sector 67Y --oversampling-factor 2 --occ-type conv
```

```
python train_nn.py --sector 67Y --oversampling-factor 2 --occ-type flatten
```

```
python train_xgb.py --sector 67Y --oversampling-factor 2
```


## Sector W

```
python train_nn.py --sector W --oversampling-factor 3 --occ-type conv
```

```
python train_nn.py --sector W --oversampling-factor 3 --occ-type flatten
```

```
python train_xgb.py --sector W --oversampling-factor 3
```
