from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from model_logic import recommend_recipes
import uvicorn
import os

app = FastAPI(
    title="Recipe Recommender API",
    description="AI-powered recipe recommendation system",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecipeRequest(BaseModel):
    user_ingredients: List[str]
    user_diet: Optional[str] = None
    debug: Optional[bool] = False  # To enable similarity score if needed

@app.get("/")
async def root():
    return {"message": "Recipe Recommender API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "recipe-recommender"}

@app.post("/recommend")
async def recommend_endpoint(payload: RecipeRequest):
    try:
        if not payload.user_ingredients:
            raise HTTPException(status_code=400, detail="user_ingredients cannot be empty")
        
        result_df = recommend_recipes(
            user_ingredients=payload.user_ingredients,
            user_diet=payload.user_diet,
            debug=payload.debug
        )
        
        if result_df.empty:
            return {"message": "No recipes found matching your criteria", "recipes": []}
        
        return {"recipes": result_df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        # Log the error to console
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
