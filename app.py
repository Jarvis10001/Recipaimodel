from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from recipe_recommender.model_logic import recommend_recipes
import uvicorn

app = FastAPI()

class RecipeRequest(BaseModel):
    user_ingredients: List[str]
    user_diet: Optional[str] = None
    debug: Optional[bool] = False  # To enable similarity score if needed

@app.post("/recommend")
async def recommend_endpoint(payload: RecipeRequest):
    try:
        result_df = recommend_recipes(
            user_ingredients=payload.user_ingredients,
            user_diet=payload.user_diet,
            debug=payload.debug
        )
        return result_df.to_dict(orient="records")
    except Exception as e:
        # Log the error to console
        print(f"❌ Error: {e}")
        return {"error": str(e)}
