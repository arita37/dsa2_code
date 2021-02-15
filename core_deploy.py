from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from source.util_feature import load_function_uri
from source.run_preprocess import preprocess_inference as preprocess
from source.run_inference import predict,model_dict_load
from core_run import get_global_pars,get_config_path
import pandas as pd
import os,sys

##### Abspath of main execution file ######
dir_data =  os.path.abspath( sys.modules['__main__'].__file__).replace("\\", "/")
print("core_deploy dir_data",dir_data)

### Import Model Class Serializers. Look at titanitc_classifier.py
BodyOne = load_function_uri(uri_name= dir_data+'::BodyOne')
BodyBatch = load_function_uri(uri_name= dir_data+'::BodyBatch')

##### Config uri extracted from the main file on execution, in this case titanic_classifier.py
config_uri, config_name = get_config_path('')

##### Load global_pars
mdict = get_global_pars(config_uri)
m = mdict['global_pars']
config_path = m['config_path']

############# Extracted from source/run_predict.py run_predict method #############

###### load model_dict trained
model_dict = model_dict_load(None, config_path, config_name, verbose=True)
m          = model_dict['global_pars']

model_class      = model_dict['model_pars']['model_class']
path_pipeline    = m['path_pred_pipeline']
path_model       = m['path_pred_model']

pars = {'cols_group': model_dict['data_pars']['cols_input_type'],
        'pipe_list' : model_dict['model_pars']['pre_process_pars']['pipe_list']}

##### Declare FastAPI Application #####
app = FastAPI()

@app.post("/")
async def post_model(data: BodyOne):
    #### Serialize using pydantic ^ ####
    json_data = jsonable_encoder(data)
    #### Convert as Json Data as DF and preprocess ####
    df = pd.DataFrame(json_data, index=[0])
    dfX, cols_family = preprocess(df, path_pipeline, preprocess_pars=pars)

    #### Make Inference ####
    ypred, yprob = predict(model_class, path_model, dfX, cols_family)
    return {"pred": ypred.tolist()[0]}

@app.post("/batch")
async def post_model(data: BodyBatch):
    #### Serialize using pydantic ^ ####
    json_data = jsonable_encoder(data)
    #### Convert as Json Data as DF and preprocess ####
    df = pd.DataFrame(json_data["Batch"])
    dfX, cols_family = preprocess(df, path_pipeline, preprocess_pars=pars)

    #### Make Inference ####
    ypred, yproba = predict(model_class, path_model, dfX, cols_family)
    return {"pred": ypred.tolist()}

