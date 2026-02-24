library(Matrix)
library(stm)

X_train_ngram <- readMM(
  "notebooks/prototyping/data/experiment_outputs/Monroe_Extractor_Test_fulldata/X_train_from_python.mtx"
)
X_test_ngram <- readMM(
  "notebooks/prototyping/data/experiment_outputs/Monroe_Extractor_Test_fulldata/X_test_from_python.mtx"
)

stm_model <- readRDS(
  "notebooks/prototyping/data/experiment_outputs/STM_Test_fulldata/stm_model/stm_model.rds"
)

# is the order correct? - it is alphabetical, so assuming to be fine for the moment, but should be checked more carefully
colnames(X_train_ngram) <- stm_model$vocab
colnames(X_test_ngram) <- stm_model$vocab

common_terms <- intersect(colnames(X_train_ngram), stm_model$vocab)
term_index <- match(common_terms, stm_model$vocab) # integer indices in original vocab

# Convert each row of X_new_aligned to stm document format:
# each element is a 2-row matrix: row 1 = word indices, row 2 = counts
dgT_to_stm_docs <- function(m, term_index_vec) {
  m <- as(m, "dgTMatrix")
  docs_split <- split(seq_along(m@i), m@i) # split by document (row index)
  docs <- vector("list", nrow(m))
  for (d in seq_len(nrow(m))) {
    idx <- docs_split[[as.character(d - 1)]]
    if (length(idx) == 0L) {
      docs[[d]] <- matrix(integer(0), nrow = 2)
    } else {
      # j = column indices, x = counts
      cols <- m@j[idx] + 1L
      counts <- m@x[idx]
      # map to model vocab indices
      widx <- term_index_vec[cols]
      # remove any NA just in case
      keep <- !is.na(widx)
      widx <- widx[keep]
      counts <- counts[keep]
      if (length(widx) == 0L) {
        docs[[d]] <- matrix(integer(0), nrow = 2)
      } else {
        docs[[d]] <- rbind(widx, counts)
      }
    }
  }
  docs
}

docs_new <- dgT_to_stm_docs(X_train_ngram, term_index)
new_res <- fitNewDocuments(stm_model, documents = docs_new)
X_train_new <- new_res$theta |> as.data.frame()
nrow(X_train_new)

docs_new <- dgT_to_stm_docs(X_test_ngram, term_index)
new_res <- fitNewDocuments(stm_model, documents = docs_new)
X_test_new <- new_res$theta |> as.data.frame()
nrow(X_test_new)

write.csv(
  X_train_new,
  "notebooks/prototyping/data/experiment_outputs/STM_Test_fulldata/X_train_stm.csv",
  row.names = FALSE
)

write.csv(
  X_test_new,
  "notebooks/prototyping/data/experiment_outputs/STM_Test_fulldata/X_test_stm.csv",
  row.names = FALSE
)
