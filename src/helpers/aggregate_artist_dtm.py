import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def aggregate_dtm_by_artist(X, artist, genre):
    """
    Aggregate track-level DTM to artist-level.

    Reduces sparsity by combining all tracks from the same artist into a single
    document. Genre is assigned based on the artist's most frequent genre, with
    ties broken by selecting the globally rarer genre to promote diversity.

    Parameters
    ----------
    X : sparse matrix, shape (n_tracks, n_features)
        Track-level document-term matrix.
    artist : pd.Series or array-like
        Artist names for each track.
    genre : pd.Series or array-like
        Genre labels for each track.

    Returns
    -------
    X_artist : sparse matrix, shape (n_artists, n_features)
        Artist-level aggregated DTM.
    artist_genres : pd.Series
        Genre label for each artist (index = artist name, sorted alphabetically).
    """
    artist = pd.Series(artist).reset_index(drop=True)
    genre = pd.Series(genre).reset_index(drop=True)

    unique_artists = sorted(artist.unique())
    artist_to_idx = {a: i for i, a in enumerate(unique_artists)}

    global_genre_freq = genre.value_counts()

    artist_genres = _assign_artist_genres(
        artist, genre, global_genre_freq, unique_artists
    )

    if not hasattr(X, "tocsr"):
        X = csr_matrix(X)
    else:
        X = X.tocsr()

    n_tracks = len(artist)
    n_artists = len(unique_artists)

    artist_idx_array = np.array([artist_to_idx[a] for a in artist])

    row_indices = np.arange(n_tracks)
    col_indices = artist_idx_array
    data = np.ones(n_tracks)

    aggregation_matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_tracks, n_artists),
        dtype=np.float64,
    )

    X_artist = (aggregation_matrix.T @ X).astype(np.float64)

    return X_artist, artist_genres


def _assign_artist_genres(artist, genre, global_genre_freq, unique_artists):
    """
    Assign genre to each artist using mode with diversity-promoting tie-breaker.

    For each artist, selects the most frequent genre among their tracks. If
    multiple genres tie for most frequent, selects the one with lowest global
    frequency to promote diversity.

    Parameters
    ----------
    artist : pd.Series
        Artist names for each track.
    genre : pd.Series
        Genre labels for each track.
    global_genre_freq : pd.Series
        Global genre frequencies across all tracks.
    unique_artists : list
        Sorted list of unique artist names.

    Returns
    -------
    artist_genres : pd.Series
        Genre for each artist (index = artist name).
    """
    artist_genre_df = pd.DataFrame({"artist": artist, "genre": genre})

    def select_genre_with_tiebreaker(genres):
        mode_counts = genres.value_counts()
        max_count = mode_counts.max()
        tied_genres = mode_counts[mode_counts == max_count].index.tolist()

        if len(tied_genres) == 1:
            return tied_genres[0]

        tied_freqs = global_genre_freq[tied_genres]
        return tied_freqs.idxmin()

    artist_genres = (
        artist_genre_df.groupby("artist")["genre"]
        .agg(select_genre_with_tiebreaker)
        .reindex(unique_artists)
    )

    return artist_genres
