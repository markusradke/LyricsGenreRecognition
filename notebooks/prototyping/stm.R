library(stm)
library(dplyr)
library(readr)
library(quanteda)

lyrics_train = read_csv("notebooks/prototyping/corpus_train_replaced_toR.csv")
lyrics_test = read_csv("notebooks/prototyping/corpus_test_replaced_toR.csv")

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

# fit STM
K <- 10
heldout <- make.heldout(prep$documents, prep$vocab)
documents <- heldout$documents
vocab <- heldout$vocab
meta <- prep$meta
stm_model <- stm(
  documents = documents,
  vocab = vocab,
  K = K,
  prevalence = ~genre,
  data = meta,
  max.em.its = 5, # increase if needed
  init.type = "Spectral"
)
eval.heldout(stm_model, heldout$missing)


# inspect STM
labelTopics(stm_model, n = 10) # top words per topic
plot.STM(stm_model, type = "summary") # topic summary plot

# Estimate effect of genre on topic prevalence
prep <- estimateEffect(
  1:K ~ genre,
  stm_model,
  meta = meta,
  uncertainty = "Global"
)
summary(prep)
plot(stm_model, tpye = "hist")
topic_correlation <- topicCorr(stm_model, method = "simple")
plot.topicCorr(topic_correlation)


X_train <- stm_model$theta |> as.data.frame() |> assign_colnames_to_X()


test_documents <- preprocess_lyrics_for_stm(lyrics_test$lyrics, vocab)
X_test <- fitNewDocuments(stm_model, documents = test_documents)$theta |>
  as.data.frame() |>
  assign_colnames_to_X()

write_csv(X_train, file = "notebooks/prototyping/X_train_stm.csv")
write_csv(X_test, file = "notebooks/prototyping/X_test_stm.csv")
