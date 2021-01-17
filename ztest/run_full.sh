

pwd
which python
ls .

# python outlier_predict.py  preprocess  ;
# python outlier_predict.py  train    --nsample 1000     ;
# python outlier_predict.py  predict  --nsample 1000   ;

# python classifier_multi.py  train    --nsample 10000   ;


python example/regress_salary.py  train   --nsample 1000
python example/regress_salary.py  predict  --nsample 1000

python example/regress_cardif.py  train   --nsample 1000


python example/regress_airbnb.py  train   --nsample 20000
python example/regress_airbnb.py  predict  --nsample 5000






python example/classifier_income.py  train    --nsample 1000   ;
python example/classifier_income.py  predict  --nsample 1000   ;




