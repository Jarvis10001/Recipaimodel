# Recipe Recommender API

An AI-powered recipe recommendation system built with FastAPI and scikit-learn.

## Features

- Ingredient-based recipe recommendations
- Dietary preference filtering
- Ingredient substitution suggestions
- Missing ingredient identification
- Cosine similarity-based matching

## API Endpoints

### `GET /`
Health check endpoint that returns API status and version.

### `GET /health`
Service health check endpoint.

### `POST /recommend`
Main recommendation endpoint.

**Request Body:**
```json
{
  "user_ingredients": ["chicken", "rice", "onion"],
  "user_diet": "non-vegetarian",
  "debug": false
}
```

**Response:**
```json
{
  "recipes": [
    {
      "name": "Chicken Biryani",
      "translated_ingredients": ["chicken", "rice", "onion", "spices"],
      "instructions_translated": "Step by step instructions...",
      "diet": "non-vegetarian",
      "course": "main course",
      "cuisine": "Indian",
      "prep_time": "45 minutes",
      "image_url": "https://example.com/image.jpg",
      "missing_ingredients": ["saffron", "yogurt"],
      "suggested_substitutes": {
        "saffron": "turmeric (for color), or omit",
        "yogurt": "plant-based yogurt, sour cream, or lemon juice with water"
      }
    }
  ]
}
```

## Deployment on Render

1. **Push to GitHub**: Push your code to a GitHub repository.

2. **Create Render Account**: Sign up at [render.com](https://render.com)

3. **Create Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure the service:
     - **Name**: `recipe-recommender-api`
     - **Region**: Choose closest to your users
     - **Branch**: `main`
     - **Runtime**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Environment Variables** (if needed):
   - Add any required environment variables in Render dashboard

5. **Deploy**: Click "Create Web Service"

## Using the API in Other Projects

Once deployed, you can use the API in other projects:

### Python Example:
```python
import requests

# Your Render URL
API_URL = "https://your-app-name.onrender.com"

def get_recipe_recommendations(ingredients, diet=None):
    payload = {
        "user_ingredients": ingredients,
        "user_diet": diet,
        "debug": False
    }
    
    response = requests.post(f"{API_URL}/recommend", json=payload)
    return response.json()

# Example usage
recommendations = get_recipe_recommendations(
    ingredients=["chicken", "rice", "onion"],
    diet="non-vegetarian"
)
print(recommendations)
```

### JavaScript Example:
```javascript
const API_URL = 'https://your-app-name.onrender.com';

async function getRecipeRecommendations(ingredients, diet = null) {
    const payload = {
        user_ingredients: ingredients,
        user_diet: diet,
        debug: false
    };
    
    const response = await fetch(`${API_URL}/recommend`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
    });
    
    return await response.json();
}

// Example usage
getRecipeRecommendations(['chicken', 'rice', 'onion'], 'non-vegetarian')
    .then(data => console.log(data));
```

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn app:app --reload
```

3. Access the API at `http://localhost:8000`
4. View API documentation at `http://localhost:8000/docs`

## Project Structure

```
recipe_recommender/
├── app.py                 # FastAPI application
├── model_logic.py         # Recipe recommendation logic
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment config
├── runtime.txt           # Python version specification
├── data/
│   └── processed/
│       └── recipes_translated_full.csv
└── README.md             # This file
```

## Notes

- The API includes CORS middleware for cross-origin requests
- Ingredient matching uses TF-IDF vectorization and cosine similarity
- Common pantry ingredients are automatically excluded from matching
- Ingredient substitutions are provided for missing ingredients
- Error handling includes proper HTTP status codes