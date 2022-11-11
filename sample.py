from recsys.datasets import InteractionsDataset
from recsys.external_datasets import Movielens_1M
from recsys.layers import retrieve_nearest_neighbors
from recsys.models.retrieval import DeepRetriever
from recsys.models.scoring import NCF

# Setup dataset
data = Movielens_1M()
ratings, users, movies = data.load()
dataset = InteractionsDataset(
    ratings,
    users,
    movies,
    item_id="movie_id",
    interaction_id="rating",
    sample_negatives=3,
)

# Train a retriever model
retriever = DeepRetriever(dataset.data_schema)
retriever.fit(dataset=dataset, num_epochs=1)

item_alias, item_representations = retriever.generate_item_representations(dataset)
user_alias, user_representations = retriever.generate_user_representations(dataset)
retrieved_items = retrieve_nearest_neighbors(
    candidates=item_representations, query=user_representations[0], k=10
)

# Train a Deep learning feature based scorer
scorer = NCF(dataset.data_schema)
scorer.fit(dataset=dataset, num_epochs=1)

user_features = dataset.get_user_features(user)
item_features = dataset.get_item_features(item)

scorer.score(users, items)
