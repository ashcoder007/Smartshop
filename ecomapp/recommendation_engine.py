from .models import Product  # Import the Product model
from .models import UserProductRating, Product
import pandas as pd


def get_collaborative_recommendations(user, n_recommendations=8):
    ratings = UserProductRating.objects.all()

    if not ratings.exists():
        return Product.objects.all().order_by('?')[:n_recommendations]

    data = [(r.user.id, r.product.id, r.rating) for r in ratings]
    df = pd.DataFrame(data, columns=['user_id', 'product_id', 'rating'])
    pivot = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

    # 🔐 Safe check: If current user has no ratings, fallback to random
    if user.id not in pivot.index:
        return Product.objects.all().order_by('?')[:n_recommendations]

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=5, random_state=42)
    matrix = svd.fit_transform(pivot)
    # Removed invalid operator
    user_idx = list(pivot.index).index(user.id)
    user_vector = matrix[user_idx]

    predicted_ratings = user_vector.dot(svd.components_)
    predicted_ratings_series = pd.Series(predicted_ratings, index=pivot.columns)

    rated_products = df[df['user_id'] == user.id]['product_id'].tolist()
    predicted_ratings_series = predicted_ratings_series.drop(labels=rated_products, errors='ignore')

    top_product_ids = predicted_ratings_series.sort_values(ascending=False).head(n_recommendations).index
    recommended_products = Product.objects.filter(id__in=top_product_ids)

    return recommended_products
