#!/usr/bin/env python
# coding: utf-8

# # Initialize env

# ### Load libraries

# In[2]:

import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[3]:


import fastapi
from fastapi import FastAPI


# In[4]:


import uvicorn


# In[5]:


import nest_asyncio


# In[6]:




# In[7]:


from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# In[9]:


import librosa
from sklearn.preprocessing import LabelEncoder


# ### Load saved model

# In[10]:


model = keras.models.load_model("model")
model.summary()


# In[219]:

static_dir = "static/"
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file=static_dir + 'templates/images/model_plot.png', show_shapes=True, show_layer_names=True)


# ### Configure labels

# In[11]:


labels = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]


# ### Configure paths

# In[125]:


sample_sounds_zip = static_dir + "sample_sounds.zip"
uploads_dir = static_dir + "uploads/"
get_ipython().system('@mkdir "static/uploads"')


# # Model methods

# In[65]:


def parse(file):
    X, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)        
    return mels


# In[66]:


def predict(file):
    features = parse(file)
    features = features.reshape(-1, 16, 8, 1)
    pred_vector = np.argmax(model.predict([features]), axis=-1)
    pred_class = labels[pred_vector[0]]
    return pred_class


# In[67]:


for i in range(1, 8):
    print(predict(r"static/sample_sounds/" + str(i) + ".wav"))


# # API Routes

# ### Configure FastAPI server

# In[260]:


app = FastAPI()

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory="static/templates")

@app.get("/test")
async def test():
    return {"message": "Testing Endpoint"}


# ### Backend

# In[261]:


from fastapi import File, UploadFile

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(uploads_dir + file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": predict(uploads_dir + file.filename)}


# In[262]:


from fastapi.responses import FileResponse

@app.get("/download-samples")
def download_samples():
    res = FileResponse(path=sample_sounds_zip, filename="sample_sounds.zip", media_type='application/zip')
    res.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
    res.headers["Content-Disposition"] = "attachment; filename=sample_sounds.zip"
    return res


# ### Template serving

# In[263]:


from fastapi import Request

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# In[264]:


from fastapi import Request

@app.get("/sample-data", response_class=HTMLResponse)
async def sample_data(request: Request):
    return templates.TemplateResponse("sample_data.html", {"request": request})


# In[265]:


from fastapi import Request

@app.get("/arch", response_class=HTMLResponse)
async def arch(request: Request):
    return templates.TemplateResponse("arch.html", {"request": request})


# In[266]:


from fastapi import Request

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


# # Start server

# In[267]:


# In[ ]:




