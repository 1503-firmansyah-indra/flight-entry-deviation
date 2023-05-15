# flight-entry-deviation
This is the implementation of paper "Predicting Deviation of Flight Entry into Air Sector using Machine Learning Techniques"

## System Setup

- Machine used: Windows 10
- Python 3.7.15
- Refer to `requirements.txt` for the libraries used

## Downloading Raw Data

The data used in this work can be downloaded from [here](https://data.mendeley.com/datasets/8yn985bwz5). 
Afterwards, unzip the data and arrange the folders in `raw_data` like the structure below:
```text
- raw_data/
    - sectors_info/
        - esmm_acc_sector_6/
        - esmm_acc_sector_Y/
        - esmm_acc_sector_W/
    - scat20161015_20161021/
    - scat20161112_20161118/
    - scat20161210_20161216/
    - scat20170107_20170113/
    - scat20170215_20170221/
    - scat20170304_20170310/
    - scat20170401_20170407/
    - scat20170429_20170505/
    - scat20170527_20170602/
    - scat20170624_20170630/
    - scat20170722_20170728/
    - scat20170819_20170825/
    - scat20170916_20170922/
```

## Pre-processing Raw Data

- Execute the code in notebook `01_filter_flights.ipynb` and `02_occupancy_processing.ipynb`
- For running notebook `02_occupany_processing.ipynb`, two runs have to be done; one with setting `sector_name = "sector_w_esmm"` and
another with `sector_name  = "sector_67Y"`.
  - This is to generate preprocessed data for sector W and 67Y, respectively. 
- After finished running the two notebooks, the processed data  folder `process_data` should have the following folder structure

```text
- processed_data/
    - sector_67Y/
        - intermediate_data/
        - occupancy/
        - trajectory/
        - sector_sector_67Y_buffer15_combined_results.csv
    - sector_w_esmm/
        - intermediate_data/
        - occupancy/
        - trajectory/
        - sector_sector_w_esmm_buffer15_combined_results.csv
```

- Navigate to `<root>/src` and run the following lines in command line

```
python train_nn.py --oversampling-factor=2 --sector=67Y

python train_nn.py --oversampling-factor=3 --sector=W

python train_xgb.py --oversampling-factor=2 --sector=67Y

python train_xgb.py --oversampling-factor=3 --sector=W
```