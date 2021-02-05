# pylint: disable=C0321,C0103,C0301,E1305,E1121,C0302,C0330,C0111,W0613,W0611,R1705
# -*- coding: utf-8 -*-
"""
https://github.com/arita37/pytorch_tabular


        The core model which orchestrates everything from initializing the datamodule, the model, trainer, etc.
        Args:
            config (Optional[Union[DictConfig, str]], optional): Single OmegaConf DictConfig object or
                the path to the yaml file holding all the config parameters. Defaults to None.
            data_config (Optional[Union[DataConfig, str]], optional): DataConfig object or path to the yaml file. Defaults to None.
            model_config (Optional[Union[ModelConfig, str]], optional): A subclass of ModelConfig or path to the yaml file.
                Determines which model to run from the type of config. Defaults to None.
            optimizer_config (Optional[Union[OptimizerConfig, str]], optional): OptimizerConfig object or path to the yaml file.
                Defaults to None.
            trainer_config (Optional[Union[TrainerConfig, str]], optional): TrainerConfig object or path to the yaml file.
                Defaults to None.
            experiment_config (Optional[Union[ExperimentConfig, str]], optional): ExperimentConfig object or path to the yaml file.
                If Provided configures the experiment tracking. Defaults to None.
            model_callable (Optional[Callable], optional): If provided, will override the model callable that will be loaded from the config.
                Typically used when providing Custom Models

"""
import os

try :
    from pytorch_tabular import TabularModel
    from pytorch_tabular.models import CategoryEmbeddingModelConfig
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
except :
    os.system("pip install pytorch_tabular[all]")

import numpy as np
# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_deterministic(True)
# from torch.utils import data

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import wget

####################################################################################################
VERBOSE = False

def log(*s):
    print(*s, flush=True)


####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None



class Model(object):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        self.model_pars, self.compute_pars, self.data_pars = model_pars, compute_pars, data_pars

        if model_pars is None:
            self.model = None

        else:
            ###############################################################
            dm          = data_pars['cols_model_group_custom']
            data_config = DataConfig(
              target           = dm['coly'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
              continuous_cols  = dm['colnum'],
              categorical_cols = dm['colcat'],
            )

            model_config     = CategoryEmbeddingModelConfig( **model_pars['model_pars']    )
            trainer_config   = TrainerConfig( **compute_pars['compute_pars'] )
            optimizer_config = OptimizerConfig()

            self.config_pars = { 'data_config' : data_config,
                        'model_config' : model_config,
                        'optimizer_config' : optimizer_config,
                        'trainer_config' : trainer_config,
            }

            self.model = TabularModel(**self.config_pars)
            self.guide = None
            self.pred_summary = None  ### All MC summary

            if VERBOSE: log(self.guide, self.model)


def fit(data_pars=None, compute_pars=None, out_pars=None, **kw):
    """
    """
    global model, session
    session = None  # Session type for compute
    Xtrain, ytrain, Xtest, ytest = get_dataset(data_pars, task_type="train")
    if VERBOSE: log(Xtrain, model.model)

    #Xtrain = torch.tensor(Xtrain.values, dtype=torch.float)
    #Xtest  = torch.tensor(Xtest.values, dtype=torch.float)
    #ytrain = torch.tensor(ytrain.values, dtype=torch.float)
    #ytest  = torch.tensor(ytest.values, dtype=torch.float)


    train = pd.concat((Xtrain,ytrain)).values
    val   = pd.concat((Xtest,ytest)).values

    ###############################################################
    compute_pars2 = compute_pars.get('compute_pars', {})

    model.model.fit(train=train, validation=val)

    #############################################################


def predict(Xpred=None, data_pars={}, compute_pars=None, out_pars={}, **kw):
    global model, session

    compute_pars2 = model.compute_pars if compute_pars is None else compute_pars
    num_samples   = compute_pars2.get('num_samples', 300)

    ###### Data load
    if Xpred is None:
        Xpred = get_dataset(data_pars, task_type="predict")
    cols_Xpred = list(Xpred.columns)

    max_size = compute_pars2.get('max_size', len(Xpred))
    Xpred    = Xpred.iloc[:max_size, :]

    # Xpred_   = torch.tensor(Xpred.values, dtype=torch.float)

    ###### Post processing normalization
    post_process_fun = model.model_pars.get('post_process_fun', None)
    if post_process_fun is None:
        def post_process_fun(y):
            return y


    #####################################################################
    ypred = model.model.predict(Xpred)

    ypred = ypred #### Nornalized

    ypred_proba = None  ### No proba
    if compute_pars.get("probability", False):
         ypred_proba = model.model.predict_proba(Xpred)
    return ypred, ypred_proba



def reset():
    global model, session
    model, session = None, None


def save(path=None, info=None):
    """
       Custom saving
    """

    ""
    global model, session
    import cloudpickle as pickle
    os.makedirs(path, exist_ok=True)

    filename = "model.pkl"
    model.model.save_model(f"{path}/filename")
    pickle.dump(model, open(f"{path}/{filename}", mode='wb'))  # , protocol=pickle.HIGHEST_PROTOCOL )


    model.model.save_model(f"{path}/torch_checkpoint/")


    filename = "info.pkl"
    pickle.dump(info, open(f"{path}/{filename}", mode='wb'))  # ,protocol=pickle.HIGHEST_PROTOCOL )


def load_model(path=""):
    global model, session
    import cloudpickle as pickle
    model0 = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars
    model.data_pars    = model0.data_pars

    model.model        = model0.model.load_from_checkpoint(f"{path}/")
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd


def preprocess(prepro_pars):
    if prepro_pars['type'] == 'test':
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(n_features=10, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)

        # log(X,y)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
        return Xtrain, ytrain, Xtest, ytest

    if prepro_pars['type'] == 'train':
        from sklearn.model_selection import train_test_split
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]
        dfy = df[prepro_pars['coly']]
        Xtrain, Xtest, ytrain, ytest = train_test_split(dfX.values, dfy.values)
        return Xtrain, ytrain, Xtest, ytest

    else:
        df = pd.read_csv(prepro_pars['path'])
        dfX = df[prepro_pars['colX']]

        Xtest, ytest = dfX, None
        return None, None, Xtest, ytest


####################################################################################################
############ Do not change #########################################################################
def get_dataset(data_pars=None, task_type="train", **kw):
    """
      "ram"  :
      "file" :
    """
    # log(data_pars)
    data_type = data_pars.get('type', 'ram')
    if data_type == "ram":
        if task_type == "predict":
            d = data_pars[task_type]
            return d["X"]

        if task_type == "eval":
            d = data_pars[task_type]
            return d["X"], d["y"]

        if task_type == "train":
            d = data_pars[task_type]
            return d["Xtrain"], d["ytrain"], d["Xtest"], d["ytest"]

    elif data_type == "file":
        raise Exception(f' {data_type} data_type Not implemented ')

    raise Exception(f' Requires  Xtrain", "Xtest", "ytrain", "ytest" ')


def get_params(param_pars={}, **kw):
    import json
    # from jsoncomment import JsonComment ; json = JsonComment()
    pp = param_pars
    choice = pp['choice']
    config_mode = pp['config_mode']
    data_path = pp['data_path']

    if choice == "json":
        cf = json.load(open(data_path, mode='r'))
        cf = cf[config_mode]
        return cf['model_pars'], cf['data_pars'], cf['compute_pars'], cf['out_pars']

    else:
        raise Exception(f"Not support choice {choice} yet")





def test(config=''):
    """
       python source/models/model_torch_tabular.py test

    """
    global model, session

    #X = np.random.rand(10000,20)
    #y = np.random.binomial(n=1, p=0.5, size=[10000])

    BASE_DIR = Path.home().joinpath('data/input/covtype/')
    datafile = BASE_DIR.joinpath('covtype.data.gz')
    datafile.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    if not datafile.exists():
        wget.download(url, datafile.as_posix())

    target_name = ["Covertype"]

    colcat = [ "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"
                      ]

    colnum = [ "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
    ]

    feature_columns = (  colnum + colcat + target_name)

    df = pd.read_csv(datafile, header=None, names=feature_columns)

    df.head()
    train, test = train_test_split(df, random_state=42)
    train, val  = train_test_split(train, random_state=42)
    num_classes = len(set(train[target_name].values.ravel()))


    data_config = DataConfig(
        target=target_name,
        continuous_cols=colnum,
        categorical_cols=colcat,
        continuous_feature_transform=None,#"quantile_normal",
        normalize_continuous_features=False
    )
    model_config = CategoryEmbeddingModelConfig(task="classification",
                                                metrics=["f1","accuracy"],
                                                metrics_params=[{"num_classes":num_classes},{}])

    trainer_config = TrainerConfig(gpus=1, fast_dev_run=True)
    experiment_config = ExperimentConfig(project_name="PyTorch Tabular Example",
                                         run_name="node_forest_cov",
                                         exp_watch="gradients",
                                         log_target="wandb",
                                         log_logits=True)
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        # experiment_config=experiment_config,
    )
    
    
    tabular_model.fit(  train=train, validation=val)
    result = tabular_model.evaluate(test)
    print(result)
    
    
    test.drop(columns=target_name, inplace=True)
    pred_df = tabular_model.predict(test)
    pred_df.to_csv("output/temp2.csv")
    # tabular_model.save_model("test_save")
    # new_model = TabularModel.load_from_checkpoint("test_save")
    # result = new_model.evaluate(test)


    ####### Using API
    X = df
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, random_state=2021, stratify=y)
    X_train, X_valid, y_train, y_valid         = train_test_split(X_train_full, y_train_full, random_state=2021, stratify=y_train_full)


    model_pars = {'model_class': 'model_torch_tabular.py::model',
                  'model_pars': {'n_wide_cross': 10,
                                 'n_wide': 10},
                 }
    data_pars = {'train': {'Xtrain': X_train,
                           'ytrain': y_train,
                           'Xtest': X_test,
                           'ytest': y_test},
                 'eval': {'X': X_valid,
                          'y': y_valid},
                 'predict': {'X': X_valid},
                }
    compute_pars = { 'compute_pars' : {} }



    model = Model(model_pars=model_pars, data_pars=data_pars, compute_pars=compute_pars)

    print('\n\nTraining the model..')
    fit(data_pars=data_pars, compute_pars=compute_pars, out_pars=None)
    print('Training completed!\n\n')

    print('Predict data..')
    ypred, ypred_proba = predict(Xpred=None, data_pars=data_pars, compute_pars=compute_pars)
    print(f'Top 5 y_pred: {np.squeeze(ypred)[:5]}')
    print('Data successfully predicted!\n\n')

    print('Evaluating the model..')
    print(eval(data_pars=data_pars, compute_pars=compute_pars))
    print('Evaluating completed!\n\n')

    print('Saving model..')
    save(path='model_dir/')
    print('Model successfully saved!\n\n')

    print('Load model..')
    model, session = load_model(path="model_dir/")
    print('Model successfully loaded!\n\n')

    print('Model architecture:')
    print(model)







if __name__ == "__main__":
    import fire
    fire.Fire(test)

