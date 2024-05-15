from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import io

app = FastAPI()


    
@app.get("/")
def read_root():
    return {"status": "ok"}
