library(tidyverse)
# This code runs all R based figures and tables:
# Make sure to run from the project root (the folder the file is in)

# Output folders
folders <- c(
  "./output/figures/descriptive",
  "./output/figures/did",
  "./output/figures/llm_validation",
  "./output/figures/mechanisms",
  "./output/tables/descriptive",
  "./output/tables/did",
  "./output/tables/iv",
  "./output/tables/topics"
)

# Create each folder if it doesn't exist
for (folder in folders) {
  if (!dir.exists(folder)) {
    dir.create(folder, recursive = TRUE)
  }
}

figures <- list.files("./codes/figures", pattern = "\\.R$", full.names = TRUE)

# Order figures by number / appendix
figures_oredered <- figures[order(
  grepl("Figure A", basename(figures)),
  as.numeric(gsub("Figure (A?)(\\d+)\\.R", "\\2", basename(figures))),
  gsub("Figure (A?)(\\d+)\\.R", "\\1", basename(figures))
)]

# Run each figure
figures_oredered %>% walk(function(file){
  message(str_glue("Running {file}"))
  source(file)
  message(str_glue("Completed {file}"))
})

tables <- list.files("./codes/tables", pattern = "\\.R$", full.names = TRUE)

# Order tables by number / appendix
tables_oredered <- tables[order(
  grepl("Table A", basename(tables)),
  as.numeric(gsub("Table (A?)(\\d+)\\.R", "\\2", basename(tables))),
  gsub("Table (A?)(\\d+)\\.R", "\\1", basename(tables))
)]

# Source each table
tables_oredered %>% walk(function(file){
  message(str_glue("Running {file}"))
  source(file)
  message(str_glue("Completed {file}"))
})