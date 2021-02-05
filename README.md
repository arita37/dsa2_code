
#### Check here if latest commit is working :

[Testing code ](https://github.com/arita37/dsa2/blob/main/ztest/run_fast.sh)

Main
![Main, test_fast_linux](https://github.com/arita37/dsa2/workflows/test_fast_linux/badge.svg?branch=main)
![Main, test_full](https://github.com/arita37/dsa2/workflows/test_full/badge.svg?branch=main)


Multi
  ![test_fast_linux](https://github.com/arita37/dsa2/workflows/test_fast_linux/badge.svg?branch=multi)
   ![test_full](https://github.com/arita37/dsa2/workflows/test_full/badge.svg?branch=multi)


Preprocessors Check
![test_preprocess](https://github.com/arita37/dsa2/workflows/test_preprocess/badge.svg?branch=multi)


### Looking for contributors
     Maintain and setup roadmap of this excellent Data Science / ML repo.
     Goal is to unified Data Science and Machine Learning .
     Basic idea is to have one single dictionary/json for
            model, compute, data definition,
     --> easy to define, easy to track, easy to modify.
     

### Install 
     pip install -r zrequirements.txt


### Basic usage 
    python  titanic_classifier.py  preprocess    --nsample 1000
    python  titanic_classifier.py  train         --nsample 2000
    python  titanic_classifier.py  predict



### How to train a new dataset ?
    1) Put your data file   in   data/input/mydata/raw/   
[link](https://github.com/arita37/dsa2/tree/multi/data/input/mydata)
       

    2) Update script        in   data/input/mydata/clean.py
       to load column names, basic profile...


    3) run  python clean.py train_test
        which generates train and test data in :   
           data/input/mydata/train/features.parquet   target.parquet  (y label)        
           data/input/mydata/test/features.parquet    target.parquet  (y label)                
                
    4) Copy Paste titanic_classifier.py  into  mydata_classifier.py
    
    5) Modify the script     mydata_classifier.py
        to match your dataset and the models you want to test.
          
    6) Run 
        python  mydata_classifier.py  train
        python  mydata_classifier.py  predict


        
### Examples

      In example/




### List of preprocessor

        #### Data Over/Under sampling ##################################
        prepro_sampler.pd_autoencoder(df,col, pars)
        
        prepro_sampler.pd_col_genetic_transform(df,col, pars)        
        prepro_sampler.pd_colcat_encoder_generic(df,col, pars)
        
        prepro_sampler.pd_filter_resample(df,col, pars)
        prepro_sampler.pd_filter_rows(df,col, pars)


        #### Category, Numerical  #####################################
        source/prepro.py::pd_autoencoder(df,col, pars)
        source/prepro.py::pd_col_genetic_transform(df,col, pars)
        
        source/prepro.py::pd_colcat_bin(df,col, pars)
        source/prepro.py::pd_colcat_encoder_generic(df,col, pars)
        source/prepro.py::pd_colcat_minhash(df,col, pars)
        source/prepro.py::pd_colcat_to_onehot(df,col, pars)
        
        source/prepro.py::pd_colcross(df,col, pars)
        source/prepro.py::pd_coldate(df,col, pars)
        
        source/prepro.py::pd_colnum(df,col, pars)
        source/prepro.py::pd_colnum_bin(df,col, pars)
        source/prepro.py::pd_colnum_binto_onehot(df,col, pars)
        source/prepro.py::pd_colnum_normalize(df,col, pars)
        source/prepro.py::pd_colnum_quantile_norm(df,col, pars)

        
        #### Text    ##################################################    
        source/prepro.py::pd_coltext(df,col, pars)
        source/prepro.py::pd_coltext_clean(df,col, pars)
        source/prepro.py::pd_coltext_universal_google(df,col, pars)
        source/prepro.py::pd_coltext_wordfreq(df,col, pars)
        
        
        #### Target label encoding  ##################################
        source/prepro.py::pd_coly(df,col, pars)
        
        source/prepro.py::pd_filter_resample(df,col, pars)
        source/prepro.py::pd_filter_rows(df,col, pars)
        source/prepro.py::pd_label_clean(df,col, pars)


        #### Time Series   ##########################################
        prepro_tseries.pd_ts_autoregressive(df,col, pars)
        prepro_tseries.pd_ts_basic(df,col, pars)
        prepro_tseries.pd_ts_date(df,col, pars)
        
        prepro_tseries.pd_ts_detrend(df,col, pars)
        prepro_tseries.pd_ts_generic(df,col, pars)
        prepro_tseries.pd_ts_groupby(df,col, pars)
        prepro_tseries.pd_ts_identity(df,col, pars)
        prepro_tseries.pd_ts_lag(df,col, pars)
        prepro_tseries.pd_ts_onehot(df,col, pars)
        prepro_tseries.pd_ts_rolling(df,col, pars)
        prepro_tseries.pd_ts_template(df,col, pars)







