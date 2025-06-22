# model_logic.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
import re

# Common pantry ingredients
COMMON_INGREDIENTS = set([
    "salt", "rock salt", "black salt", "water", "hot water", "cold water", "ice", "ice cubes",
    "sugar", "oil", "pepper", "black pepper", "black pepper powder", "chili powder", "turmeric",
    "jeera", "hing", "garam masala", "mustard seeds", "mustard oil", "ghee", "butter", "cumin seeds",
    "cumin powder", "coriander seeds", "coriander powder", "cumin", "coriander", "ginger", "garlic",
    "green chili", "green chillies", "laung", "cloves", "asafoetida", "curry leaves", "chillies",
    "chili flakes", "curry powder", "curry paste", "coconut oil", "vegetable oil", "olive oil",
    "sesame oil", "peanut oil", "soy sauce", "vinegar", "lemon juice", "lime juice", "honey", "maple syrup"
])

# Ingredient substitutions
SUBSTITUTIONS = {
    "ginger": "dry ginger powder, galangal, or omit",
    "garlic": "garlic powder, asafoetida (hing), or omit",
    "green chili": "chili flakes, jalapeño, or black pepper",
    "kasuri methi": "dried oregano, celery leaves, or omit",
    "curry leaves": "bay leaf, kaffir lime leaf, or omit",
    "tamarind paste": "lemon juice, amchur powder, or vinegar",
    "onions": "shallots, leek, or omit with flavor adjustments",
    "fresh coconut": "desiccated coconut, coconut milk, or cashew paste",
    "tomatoes": "tomato puree, canned tomatoes, or red bell pepper puree",
    "paneer": "tofu, cottage cheese, or omit for vegan",
    "milk": "almond milk, soy milk, or water with cashew paste",
    "yogurt": "plant-based yogurt, sour cream, or lemon juice with water",
    "ghee": "butter, coconut oil, or vegetable oil",
    "butter": "ghee, margarine, or neutral oil",
    "mustard seeds": "caraway seeds or omit",
    "fennel seeds": "anise seeds or caraway seeds",
    "ajwain": "thyme or caraway seeds",
    "cloves": "allspice, cinnamon, or omit",
    "cinnamon": "nutmeg or allspice",
    "poppy seeds": "chia seeds or omit",
    "black peppercorns": "white pepper or ground pepper",
    "bay leaf": "curry leaf, basil leaf, or omit",
    "amchur": "lemon juice or tamarind",
    "asafoetida": "garlic or onion (if diet allows)",
    "cardamom": "cinnamon or nutmeg",
    "star anise": "fennel or a pinch of cinnamon",
    "saffron": "turmeric (for color), or omit",
    "jaggery": "brown sugar, coconut sugar, sugar or maple syrup",
    "coconut milk": "almond milk, soy milk, or cashew cream",
    "chickpea flour": "all-purpose flour, almond flour, or omit",
    "besan": "chickpea flour, all-purpose flour, or omit"
}

def normalize_ingredient(ingredient):
    ingredient = ingredient.lower()
    ingredient = re.sub(r"\bred\s*chilis?\b|\bred\s*chillies?\b|\bred\s*chilli\b|\bchilli flakes\b|\bchili powder\b", "chili", ingredient)
    ingredient = re.sub(r"\bgreen\s*chilis?\b|\bgreen\s*chillies?\b", "chili", ingredient)
    ingredient = re.sub(r"\bchillis?\b", "chili", ingredient)
    ingredient = re.sub(r"\b(chopped|sliced|grated|minced|diced|ground|paste|cut into quarters|crushed|powder|fresh|dry|roasted|boiled|blanched|soaked|julienned|peeled|mashed|washed)\b", "", ingredient)
    ingredient = re.sub(r"\b(cup|cups|teaspoon|tablespoon|grams|kg|ml|liters|tbsp|tsp|½|¼|¾|⅓|⅔|dash|pinch|small|large|medium)\b", "", ingredient)
    ingredient = re.sub(r"[\d/]+", "", ingredient)
    ingredient = re.sub(r"[^a-z\s]", "", ingredient)
    ingredient = re.sub(r"\s+", " ", ingredient)
    return " ".join(sorted(set(ingredient.strip().split())))

def exclude_common_ingredients(ingredient_list):
    return [i for i in ingredient_list if i not in COMMON_INGREDIENTS]

def get_substitutes(missing_list):
    return {item: SUBSTITUTIONS.get(item, "no known substitute, optional or skip") for item in missing_list}

def is_similar_tokenwise(ing1, ing2):
    return len(set(ing1.split()) & set(ing2.split())) > 0

def compute_missing_ingredients(recipe_ings, user_ings, common_ings):
    missing = []
    for recipe_ing in recipe_ings:
        if any(is_similar_tokenwise(recipe_ing, user_ing) for user_ing in user_ings):
            continue
        if any(is_similar_tokenwise(recipe_ing, common_ing) for common_ing in common_ings):
            continue
        missing.append(recipe_ing)
    return missing

def recommend_recipes(user_ingredients, user_diet=None, debug=False):
    import os
    
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "processed", "recipes_translated_full.csv")
    
    df = pd.read_csv(csv_path)
    df['translated_ingredients'] = df['translated_ingredients'].apply(ast.literal_eval)
    df['translated_ingredients'] = df['translated_ingredients'].apply(lambda lst: [normalize_ingredient(i) for i in lst])
    user_ingredients = [normalize_ingredient(i) for i in user_ingredients]

    if user_diet:
        df = df[df['diet'].str.lower() == user_diet.lower()]
        if df.empty:
            df = pd.read_csv(csv_path)
            df['translated_ingredients'] = df['translated_ingredients'].apply(ast.literal_eval)
            df['translated_ingredients'] = df['translated_ingredients'].apply(lambda lst: [normalize_ingredient(i) for i in lst])

    df['match_ingredients'] = df['translated_ingredients'].apply(exclude_common_ingredients)
    filtered_user_ingredients = exclude_common_ingredients(user_ingredients)
    recipe_docs = df['match_ingredients'].apply(lambda x: " ".join(x)).tolist()
    user_doc = " ".join(filtered_user_ingredients)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(recipe_docs + [user_doc])
    recipe_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]
    similarities = cosine_similarity(user_vector, recipe_vectors).flatten()

    top_indices = similarities.argsort()[::-1][:10]
    top_recipes = df.iloc[top_indices].copy()
    top_recipes['similarity'] = similarities[top_indices]
    top_recipes = top_recipes[top_recipes['similarity'] > 0.1]

    top_recipes['missing_ingredients'] = top_recipes['match_ingredients'].apply(
        lambda recipe_ings: compute_missing_ingredients(recipe_ings, filtered_user_ingredients, COMMON_INGREDIENTS)
    )
    top_recipes['suggested_substitutes'] = top_recipes['missing_ingredients'].apply(get_substitutes)

    output_columns = ['name', 'translated_ingredients', 'instructions_translated',
                      'diet', 'course', 'cuisine', 'prep_time', 'image_url',
                      'missing_ingredients', 'suggested_substitutes']

    if debug:
        output_columns.insert(-2, 'similarity')

    return top_recipes[output_columns]

# FastAPI wrapper
def get_recommendations(user_ingredients: list, user_diet: str = None, debug: bool = False):
    return recommend_recipes(user_ingredients, user_diet, debug)