import bitermplus as btm
import numpy as np
import pandas as pd

from typing import Tuple

# from sklearn.model_selection import GridSearchCV


class TopicFeatureExtractor:
    """Unsupervised BTM topic modelling over input corpus."""

    def __init__(self, num_topics: Tuple = (2, 10, 1), random_state=42):
        """
        Args:
            num_topics: Tuple of (min, max, step size) for number of topics to evaluate during hyperparameter tuning
        """
        self.num_topics = np.arange(num_topics[0], num_topics[1] + 1, num_topics[2])
        self.random_state = random_state
        self.model = None

    def fit_transform(self, corpus):
        """
        Fit BTM model to corpus with hyperparameter tuning over number of topics using cross-validation.
        Uses default BTM parameters for other settings (alpha, beta, iterations, etc.) to reduce tuning complexity.
        """
        texts = corpus["lyrics"]

        vectorizer_params = {
            "token_pattern": r"(?u)\b\w+(?:_\w+)*\b",  # allows underscores inside tokens
            "lowercase": True,
            "stop_words": None,
        }
        n_topics = 150

        self.model = btm.BTMClassifier(
            n_topics, max_iter=200, vectorizer_params=vectorizer_params
        )
        features = self.model.fit_transform(texts)
        features = self._tidy_feature_frame(features)
        return features, self.model

        # param_grid = {
        #     "n_topics": self.num_topics,
        # }

        # grid_search = GridSearchCV(
        #     btm.BTMClassifier(random_state=self.random_state, vectorizer_params=None),
        #     param_grid,
        #     cv=2,
        #     scoring=None,  # Uses model's score method (Coherence)
        # )

        # grid_search.fit(texts)
        # self.model = grid_search.best_estimator_
        # return self.model.transform(corpus["lyrics"])

    def transform(self, corpus):
        """Transform new corpus using fitted BTM model."""
        self._assert_model_fitted()
        features = self.model.transform(corpus["lyrics"])
        return self._tidy_feature_frame(features)

    def get_topic_score(self, corpus):
        self._assert_model_fitted()
        return self.model.score(corpus["lyrics"].to_list())

    def _assert_model_fitted(self):
        if self.model is None:
            raise ValueError("Model not fittet yet. Call fit_transform first.")

    def _tidy_feature_frame(self, features):
        feature_names = [f"topic_{i}" for i in range(features.shape[1])]
        return pd.DataFrame(features, columns=feature_names)
