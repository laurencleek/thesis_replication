# This file stores settings/constants about the diff in diff estimation that are loaded by some of the file
EVENT_STUDY_NUMBER_LAGS <- 12
EVENT_STUDY_NUMBER_LEADS <- 4

EVENT_STUDY_FILL <- 0

# DATASET GENERATION
TREATMENT_MINIMUM_INTENSITY <- 0.05
TREATMENT_YEARS_RANGE <- (1997 - EVENT_STUDY_NUMBER_LAGS):(2023 + EVENT_STUDY_NUMBER_LEADS)

PARALLEL_TRENDS_TEST_STRING <- "Parallel trends p-value"
LEVEL_OFF_TEST_STRING <- "Levelling off p-value"

# Definition of high CBI:
HIGH_CBI <- 0.8