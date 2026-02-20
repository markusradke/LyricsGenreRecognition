import joblib
import hashlib
import pandas as pd
from pathlib import Path
from typing import Any

class FeatureCache:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self, *args, **kwargs) -> str:
        """Compute MD5 hash of inputs for cache key."""
        hash_input = {
            "args": args, 
            "kwargs": sorted(kwargs.items())
        }
        hash_str = joblib.hash(hash_input)
        return hash_str
    
    def get(self, key: str) -> Any | None:
        """Load cached features if exist."""
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            return joblib.load(cache_file)
        return None
    
    def set(self, key: str, value: Any):
        """Save features to cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        joblib.dump(value, cache_file)
    
    def cache_features(self, extractor_class, corpus, **extractor_kwargs):
        """
        Wrapper: Check cache before fitting extractor.
        Returns cached features OR fits + caches.
        """
        cache_key = self._compute_hash(
            extractor_class.__name__,
            corpus.values.tobytes(),  # Serialize corpus data
            **extractor_kwargs
        )
        
        cached = self.get(cache_key)
        if cached is not None:
            print(f"Cache hit: {cache_key[:8]}...")
            return cached
        
        print(f"Cache miss: Fitting {extractor_class.__name__}...")
        extractor = extractor_class(**extractor_kwargs)
        features = extractor.fit_transform(corpus)
        
        # Cache both features and fitted extractor
        self.set(cache_key, {"features": features, "extractor": extractor})
        return {"features": features, "extractor": extractor}