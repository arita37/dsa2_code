

pwd
which python
ls .

echo $'\n ######## Mlflow register run experiment to local.db ######## \n'

# python example/classifier_mlflow.py preprocess > data/zlog/log_mlflow_prepro.txt 2>&1
# echo $'preprocess finished see data/zlog/log_mlflow_prepro.txt for output\n'

python example/classifier_mlflow.py train > data/zlog/log_mlflow_train.txt 2>&1
echo $'train finished see data/zlog/log_mlflow_train.txt for output\n'

python example/classifier_mlflow.py predict > data/zlog/log_mlflow_predict.txt 2>&1
echo $'predict finished see data/zlog/log_mlflow_predict.txt for output\n'


echo $'######## Serving mlflow server visit http://localhost:5000 ########'
# mlflow server --backend-store-uri sqlite:///local.db --default-artifact-root ./mlruns
