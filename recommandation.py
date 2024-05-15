from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np
import io

app = FastAPI()

# Initialize global variables
global interaction_matrix
interaction_matrix = None
global global_user_indices
global_user_indices = None
global global_item_indices
global_item_indices = None

def recommend_popular_items(n):
    # Assuming 'interaction_matrix' is loaded and contains total interactions per item
    item_popularity = interaction_matrix.sum(axis=0).sort_values(ascending=False)
    popular_items = item_popularity.head(n).index.tolist()
    return popular_items


@app.post("/upload-data/")
async def upload_data(file: UploadFile = File(...)):
    global interaction_matrix, global_user_indices, global_item_indices
    try:
        # Read the uploaded file into a DataFrame
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Preprocessing steps
        data['Date'] = pd.to_datetime(data['Date'])
        data.dropna(subset=['IDUser'], inplace=True)
        data['IDUser'] = data['IDUser'].astype(int)
        
        # Creating interaction matrix and storing indices
        pivot_table = data.pivot_table(index='IDUser', columns='IDItem', aggfunc='size', fill_value=0)
        interaction_matrix = csr_matrix(pivot_table.values, dtype=np.float32)
        global_user_indices = pivot_table.index.tolist()
        global_item_indices = pivot_table.columns.tolist()
        
        return {"message": "Data uploaded and interaction matrix updated successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/recommend-items/{user_id}/{n_items}")
async def recommend_items(user_id: int, n_items: int):
    global interaction_matrix, global_user_indices, global_item_indices
    if interaction_matrix is None:
        raise HTTPException(status_code=400, detail="Interaction matrix not available.")
    
    # Check if user_id is in the global_user_indices
    if user_id not in global_user_indices:
        return {"user_id": user_id, "recommended_items": recommend_popular_items(n_items)}
    
    try:
        # SVD computation
        U, sigma, Vt = svds(interaction_matrix, k=min(interaction_matrix.shape)-1)
        sigma = np.diag(sigma)
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        # Ensure DataFrame is constructed correctly with saved indices
        predicted_ratings_df = pd.DataFrame(predicted_ratings, index=global_user_indices, columns=global_item_indices)
        
        # Get recommendations
        user_ratings = predicted_ratings_df.loc[user_id]
        recommended_items = user_ratings.sort_values(ascending=False).head(n_items).index.tolist()
        return {"user_id": user_id, "recommended_items": recommended_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
def read_root():
    return {"status": "ok"}
