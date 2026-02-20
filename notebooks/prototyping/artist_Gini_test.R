library(tidyverse)
library(DescTools)

report_genre_distributions <- function(df, granularity, artistcol) {
  if (!granularity %in% c(5, 12, 25, 32)) {
    stop("granularity must be 5, 12, 25, or 32")
  }
  genrecol <- paste0("cat", granularity)

  artist_genres <- get_artist_genres(df, genrecol, artistcol)

  genres_tracks <- df[[genrecol]]
  gini_tracks <- genres_tracks |> table() |> Gini()
  genres_artists <- artist_genres[[genrecol]]
  gini_artists <- genres_artists |> table() |> Gini()

  print_report(genres_tracks, gini_tracks, genres_artists, gini_artists)
}

get_artist_genres <- function(df, genrecol, artistcol) {
  # For each artist, choose the most frequent genre; tie-breaker: less common globally
  genre_freqs <- df |> count(.data[[genrecol]], name = "global_n", sort = TRUE)

  artist_genre_freqs <- df |>
    count(.data[[artistcol]], .data[[genrecol]], name = "n") |>
    left_join(genre_freqs, by = genrecol) |>
    group_by(.data[[artistcol]]) |>
    filter(n == max(n, na.rm = TRUE)) |>
    slice_min(global_n, with_ties = FALSE) |>
    ungroup()

  artist_genre_freqs
}

print_report <- function(
  genres_tracks,
  gini_tracks,
  genres_artists,
  gini_artists
) {
  message("====================================")
  message(sprintf("granularity: %d", genres_tracks |> unique() |> length()))
  message(sprintf(
    "# tracks: %s\n# artists: %s",
    length(genres_tracks) |> format(big.mark = ','),
    length(genres_artists) |> format(big.mark = ',')
  ))
  message(sprintf(
    "Gini tracks: %.2f\nGini artists: %.2f",
    gini_tracks,
    gini_artists
  ))
  message("------------------------------------")
  message("Genre distribution (tracks):")
  print(table(genres_tracks))
  message("Genre distribution (artists):")
  print(table(genres_artists))
}

data <- read_csv(
  "data/poptrag_lyrics_genres_corpus_filtered_english_lemmatized.csv"
)
artistcol <- "track.s.firstartist.name"
report_genre_distributions(data, granularity = 5, artistcol = artistcol)
report_genre_distributions(data, granularity = 12, artistcol = artistcol)
report_genre_distributions(data, granularity = 25, artistcol = artistcol)
report_genre_distributions(data, granularity = 32, artistcol = artistcol)
