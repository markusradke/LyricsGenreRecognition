library(earth)

X_train <- read.csv(
  "notebooks/prototyping/experiment_outputs/STM_Test_G12_fulldata/X_train_stm.csv"
)
X_test <- read.csv(
  "notebooks/prototyping/experiment_outputs/STM_Test_G12_fulldata/X_test_stm.csv"
)
y_train <- read.csv(
  "notebooks/prototyping/experiment_outputs/STM_Test_G12_fulldata/y_train_stm.csv"
)
y_test <- read.csv(
  "notebooks/prototyping/experiment_outputs/STM_Test_G12_fulldata/y_test_stm.csv"
) |>
  as.factor()

train_data <- cbind(X_train, y_train) |> dplyr::mutate(genre = factor(genre))
test_data <- cbind(X_test, y_test) |> dplyr::mutate(genre = factor(genre))

# train a MARS model with default parameters
mars_model <- earth(
  genre ~ .,
  data = train_data,
  degree = 3, # allow for interactions up to degree 2
  nprune = 50 # limit the number of terms in the model
)

# make predictions on the test set
predictions <- predict(mars_model, newdata = test_data, type = "class") |>
  factor(levels = levels(train_data$genre))

# evaluate the model's performance
confusion_matrix <- table(test_data$genre, predictions)
print(confusion_matrix)

macro_f1_with_zero <- function(truth, estimate, levels) {
  truth <- factor(truth, levels = levels)
  levs <- levels(truth)
  print(levs)
  df <- tibble::tibble(truth = truth, estimate = estimate)
  print(df)

  per_class <- lapply(levs, function(lbl) {
    tp <- sum(df$truth == lbl & df$estimate == lbl)
    fp <- sum(df$truth != lbl & df$estimate == lbl)
    fn <- sum(df$truth == lbl & df$estimate != lbl)

    precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
    recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)
    f1 <- if ((precision + recall) == 0) {
      0
    } else {
      2 * precision * recall / (precision + recall)
    }

    tibble::tibble(
      class = lbl,
      tp = tp,
      fp = fp,
      fn = fn,
      precision = precision,
      recall = recall,
      f1 = f1
    )
  })
  print(per_class)
  per_class <- per_class |>
    dplyr::bind_rows()

  mean_f1 <- mean(per_class$f1)
  list(per_class = per_class, macro_f1 = mean_f1)
}


res <- macro_f1_with_zero(
  truth = y_test$genre,
  estimate = predictions,
  levels = levels(test_data$genre)
)
res$macro_f1
res$per_class
