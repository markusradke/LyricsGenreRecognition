library(stm)
library(topicmodels)
library(ldatuning)
library(dplyr)
library(readr)
library(quanteda)

# load and preprocess data ----
lyrics_train = read_csv("notebooks/prototyping/corpus_train_replaced_toR.csv")
lyrics_test = read_csv("notebooks/prototyping/corpus_test_replaced_toR.csv")

(n_na_train <- nrow(lyrics_train |> filter(is.na(lyrics))))
(n_na_test <- nrow(lyrics_test |> filter(is.na(lyrics))))

lyrics_train <- filter(lyrics_train, !is.na(lyrics))
lyrics_test <- filter(lyrics_test, !is.na(lyrics))
(relfreq_na_train <- n_na_train / nrow(lyrics_train) * 100 |> round())
(relfreq_na_test <- n_na_test / nrow(lyrics_test) * 100 |> round())


get_statistics_tokens_per_doc <- function(lyrics_frame) {
  tokens_per_doc <- lapply(lyrics_frame$lyrics, function(x) {
    length(stringr::str_split_1(x, ' '))
  }) |>
    unlist()
  message(paste0("# docs: ", tokens_per_doc |> length()))
  message(paste0("median # tokens: ", tokens_per_doc |> median()))
  message(paste0("25% < ", tokens_per_doc |> quantile(0.25)))
  message(paste0("75% < ", tokens_per_doc |> quantile(0.75)))
  tokens_per_doc |> hist()
}

artist_genres <- lyrics_train |>
  group_by(artist) |>
  count(genre) |>
  filter(n == max(n)) |>
  select(artist, artistgenre = genre)

artist_lyrics <- lyrics_train |>
  group_by(artist) |>
  summarize(
    releaseyear = median(releaseyear),
    lyrics = paste(lyrics, collapse = " ")
  ) |>
  ungroup() |>
  inner_join(artist_genres, by = "artist")

lyrics_train <- artist_lyrics |> rename(genre = artistgenre)


# run STM estimation ---
space_tokenizer <- function(x) {
  tokens_list <- strsplit(x, " ", fixed = TRUE)
  tokens_list <- lapply(tokens_list, function(t) t[t != ""])
  return(tokens_list)
}

preprocess_lyrics_for_stm <- function(lyrics, vocab) {
  tokens <- space_tokenizer(lyrics)
  token_to_index <- setNames(seq_along(vocab), vocab)
  documents <- lapply(tokens, function(doc_tokens) {
    if (length(doc_tokens) == 0) {
      return(matrix(numeric(0), nrow = 2))
    }
    inds <- token_to_index[doc_tokens]
    tcounts <- as.integer(tabulate(inds, nbins = length(vocab)))
    term_inds <- which(tcounts > 0)
    rbind(term_inds, tcounts[term_inds])
  })
  documents
}

assign_colnames_to_X <- function(X) {
  colnames(X) <- paste("topic_", seq(ncol(X)))
  X
}


vocab <- space_tokenizer(lyrics_train$lyrics) |> unlist() |> unique() |> sort()
documents <- preprocess_lyrics_for_stm(lyrics_train$lyrics, vocab)


# metadata for stm from genre
meta <- data.frame(
  genre = as.factor(lyrics_train$genre),
  stringsAsFactors = FALSE
)

prep <- stm::prepDocuments(
  documents = documents,
  vocab = vocab,
  meta = meta
)

# fit STM
documents <- heldout$documents
vocab <- heldout$vocab
meta <- prep$meta
# stm_model <- stm(
#   documents = documents,
#   vocab = vocab,
#   K = K,
#   prevalence = ~genre,
#   content = ~genre,
#   data = meta,
#   max.em.its = 500, # increase if needed (default is 500)
#   init.type = "Spectral"
# )
search <- searchK(
  prep$documents,
  prep$vocab,
  K = 16:20,
  prevalence = ~genre,
  data = meta,
  init.type = "Spectral",
  heldout.seed = 32,
)

# SEED 32 for Heldout:
#  K   exclus    semcoh   heldout residual    bound   lbound em.its
#  2 7.921632 -101.2665  -6.94062 4.531806 -2487092 -2487092     10
#  3 8.714494 -54.60164 -6.933906 4.229068 -2485875 -2485873     14
#  4 9.250582 -54.45819 -6.923932 4.129251 -2484954 -2484951     17
#  5 9.391961 -55.53488 -6.924336 4.053822 -2484886 -2484882     18
#  6 9.278343 -59.93877 -6.778203 3.510908 -2425038 -2425031    358
#  7 9.447219 -60.16016 -6.761375 3.382365 -2416169 -2416160    453
#  8 9.506543 -57.67317 -6.749403 3.330328 -2415080 -2415069    331
#  9 9.582509 -57.68232 -6.747771 3.235265 -2410381 -2410368    420
# 10 9.668552 -61.47576 -6.722467 3.194942 -2406497 -2406482    292
# 11 9.677717 -59.93896 -6.721688 3.098902 -2402420 -2402402    307
# 12 9.721164  -63.9872 -6.701063 3.005051 -2396797 -2396777    236
# 13 9.719698 -66.30503 -6.688849 2.957106 -2392914 -2392891    123
# 14 9.734438 -65.19603 -6.670395 2.891317 -2389707 -2389682    155 # BEST
# 15  9.77534 -70.82351 -6.677086 2.870593 -2385794 -2385766    107
# 16 9.758221 -67.75698  -6.68211 2.842772 -2384637 -2384607     69
# 17 9.783388 -66.03211 -6.672833  2.81592 -2383017 -2382984     88
# 18 9.765919 -66.94164 -6.671521 2.721229 -2378621 -2378584     43
# 19 9.780602 -71.59275 -6.667709 2.768847 -2378278 -2378239     59
# 20 9.770057 -69.79013 -6.662946 2.689253 -2376832 -2376790     38
beepr::beep()

K <- 14
stm_model <- stm(
  documents = documents,
  vocab = vocab,
  K = K,
  prevalence = ~genre,
  data = meta,
  max.em.its = 500, # increase if needed (default is 500)
  init.type = "Spectral"
)

# inspect STM
labelTopics(stm_model, n = 10) # top words per topic
plot.STM(stm_model, type = "summary") # topic summary plot

# Estimate effect of genre on topic prevalence
# prep <- estimateEffect(
#   1:K ~ genre,
#   stm_model,
#   meta = meta,
#   uncertainty = "Global"
# )
# summary(prep)
topic_correlation <- topicCorr(stm_model, method = "simple")
plot.topicCorr(topic_correlation)


lyrics_train = read_csv("notebooks/prototyping/corpus_train_replaced_toR.csv")
lyrics_test = read_csv("notebooks/prototyping/corpus_test_replaced_toR.csv")


train_documents <- preprocess_lyrics_for_stm(lyrics_train$lyrics, vocab)
X_train <- fitNewDocuments(stm_model, documents = train_documents)$theta |>
  as.data.frame() |>
  assign_colnames_to_X()

test_documents <- preprocess_lyrics_for_stm(lyrics_test$lyrics, vocab)
X_test <- fitNewDocuments(stm_model, documents = test_documents)$theta |>
  as.data.frame() |>
  assign_colnames_to_X()


write_csv(X_train, file = "notebooks/prototyping/X_train_stm.csv")
write_csv(X_test, file = "notebooks/prototyping/X_test_stm.csv")
