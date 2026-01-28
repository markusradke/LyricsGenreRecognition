# GenreDiscourseAnalysis - AI Coding Assistant Instructions

## Project Overview
This is an R package for hierarchical genre classification of popular music using network analysis and machine learning. It processes genre tags from multiple music platforms (MusicBrainz, Spotify) to build genre family trees and train multiclass classifiers.

## Architecture & Pipeline

### Sequential Analysis Workflow
The project follows a strict 5-step pipeline (run via `scripts/run_full_reproduction.R`):

1. **01_import_and_prepare.R**: Import PopTraG dataset, denoise tags, filter by artist occurrences
2. **02_build_genre_trees.R**: Build hierarchical genre graphs from tag co-occurrence matrices
3. **03_fold_metagenres.R**: Tune tree folding to create metagenre hierarchies at different resolutions
4. **04_map_metagenres.R**: Map tracks to metagenres, compute assignment probabilities (`p_max`)
5. **05_classify_metagenres.R**: Train classifiers (GLMNET, Random Forest) with cross-validation

### Key Data Structures

- **Long format tags**: `track.s.id`, `tag_name`, `tag_count` (vote-weighted genre assignments)
- **Metagenre mappings**: Track-level assignments with `p_max` confidence scores
- **Artist-based splits**: Train/test splits ensure no artist overlap (use `create_artist_cv_splits()`)
- **Case weights**: `p_max` converted to `hardhat::importance_weights()` stored in `case_wts` column

### Model Storage Pattern
All models save to `models/` with consistent structure:
```
models/{classifier,metagenres,trees}/
  {name}_best_model.rds
  {name}_settings.rds
  {name}_evaluation.rds
  {name}_tuning_*.png
```

## Critical Conventions

### Package Loading
**Always** use `devtools::load_all()` at the start of analysis scripts, never `library(GenreDiscourseAnalysis)`. All scripts assume the package is loaded this way.

### Parallel Processing
- Uses `future::plan(future::multisession)` for parallelization
- Always restore with `on.exit(future::plan(future::sequential))`
- Tuning functions accept `n_cores` parameter (typically set to 19)
- Cross-validation uses `parallel_over = "everything"` in `tune::control_grid()`


### List-Column Persistence
Use custom functions for feather files containing list columns:
- `save_feather_with_lists()`: Serializes list columns to binary
- `read_feather_with_lists()`: Deserializes back to list columns

### Custom Metrics
The project uses a custom metric `macro_f1_with_zeros()` that computes macro-averaged F1 including zero-count classes. Always include in `yardstick::metric_set()` for model evaluation.

## Development Patterns

### Settings Management
Every analysis step uses a `settings` list containing all configuration. Always save settings alongside models:
```r
settings <- list(seed = 42, model_features = ..., n_cores = 19, use_caseweights = TRUE)
saveRDS(settings, "models/classifier/{model_type}_settings.rds")
```

### Factor Level Cleaning
Use `clean_factor_levels_in_folds()` to ensure factor levels are consistent across train/test/CV splits and meet minimum occurrence thresholds (`min_n_factor_level`).

### Data Versioning
Large data files tracked with DVC (`.dvc` files in `data/`, `data-raw/`, `models/metagenres/`). Never commit large `.rds` or `.feather` files directly.

## Common Gotchas

1. **Graph root**: Hierarchies always use `"POPULAR MUSIC"` as root node
2. **Resolution levels**: "low" = 10-15 metagenres, "high" = 25-30 metagenres (set in tree folding step)
3. **Memory management**: Use `rm(list = ls()); gc()` between major pipeline steps
4. **Imputation**: Missing data handled by `missForestPredict` before modeling, stored in `models/classifier/imputation/`

## File Organization

- `R/`: Package functions (exported via roxygen2)
- `analysis/`: Sequential analysis scripts (01-05)
- `inst/`: Quarto reports for each analysis step
- `data-raw/`: Raw input data + data preparation scripts
- `models/`: All trained models and artifacts
- `reports/`: Generated HTML reports from Quarto


# Code Style guidelines
- Always enforce DRY principles and check the whole file. In agent mode, please check the whole R/ folder for duplicated code / functions that can be merged.
- Use consistent indentation (2 spaces).
- Use snake_case for variable and function names.
- Use short functions, ideally with a maximum of 20 code lines per function. 
- Functions should only operate on a single level of abstraction and serve a single purpose.
- Please make sure variable names do not exeed 30 characters in length and code lines do not exeed 80 characters in length. 
- Please order the resulting functions in a way so that the most abstract functions appear at the top; as long as it does not hinder code execution. 
- Please avoid comments where they are not absolutely necessary to understand complex parts of the code that cannot be made more comprehensable by using more suitable names for variables and functions. This is important to me! Especially, I don't want comments that just repeat what the code does.
- Use verbs for methods and nouns for classes / objects
- Make sure functions do not have side effects or else make it a new function
- Make the code as readable as simple to maintain as possible!
- Document only top level function(s) in the R/ folder where applicable.
- Please do not try to run tests yourself 
- Please do not add any additional dependencies to the project without consulting me first
- Please always use the namespace notation except for Base R function
- all random processes should be reproducible by setting the seed 42 where applicable
- Files should not exceed 400 lines of code (ideally not more than 250 line). If they do, please split them into multiple files where applicable.

