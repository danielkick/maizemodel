library(foreach)
library(magrittr)
library(ggplot2)

# Settings ---------------------------------------------------------------------
scale_ECs <- TRUE

# WARNING:
# test_on_subset = TRUE  
test_on_subset = FALSE 
test_on_subset_nTrain = 900
test_on_subset_nTest = 100

# Quick Confirmations ----------------------------------------------------------
print("Current pwd is:")
print(getwd())

# Matrix Setup -----------------------------------------------------------------
data_loc <- './data_input/'

## Response Variable and Train/Test Info ---------------------------------------
# Y    <- read.csv(paste0(data_loc, "ForRY.csv"))    #  0.052 sec
# Y    <- as.matrix(Y)[, 'Y']

Idx  <- read.csv(paste0(data_loc, "ForRIdx.csv"))  #  0.059 sec
Idx$X <- Idx$X+1 # account for Python/R's array indexing 

# set up downsampling indices if testing code
if( test_on_subset ){
  # downsize data for testing
  trainIdBig <- sample(Idx[Idx$Set == "Train", "X"], test_on_subset_nTrain)
  testIdBig <- sample(Idx[Idx$Set == "Test", "X"], test_on_subset_nTest)
}

## Predictor Variables ---------------------------------------------------------
# these are massive csvs, esp. weather, but doing it this way avoids feather as 
# a dependency for ATLAS to speed up running models, create the needed matrices 
# only if they don't already exist and only create the ones we need.

## W (ERM) matrix ------------------------------------------------------------
matrix_file <- "Kweather_EC_list.rds"
matrix_file_path = paste0("./data_matrices/", matrix_file)

### Load -------------------------------------------------------------------
Wdat <- read.csv(paste0(data_loc, "ForRW.csv"))    # 97.89 sec
Wdat <- as.matrix(Wdat)
if( test_on_subset ){ # downsample if testing code
  Wdat <- Wdat[c(trainIdBig, testIdBig), ]
}


### Calc -------------------------------------------------------------------    
weather_ECs <- c("N", "P", "K", "TempMin", "TempMean", "TempMax", "DewPointMean", "RelativeHumidityMean", "SolarRadiationMean", "WindSpeedMax", "WindDirectionMean", "WindGustMax", "SoilTempMean", "SoilMoistureMean", "UVLMean", "PARMean", "PhotoperiodMean", "VaporPresEst", "WaterTotalInmm")

# env.weather is a list of 8 objects
# each object contains a matrix of 145 measurements x nObs
# these values are drawn from the input df
env.weather <- setNames(
  foreach(weather_EC=weather_ECs) %do% {
    
    # get the cols for each day's reading in form 
    # "max_temp1", "max_temp2" ... max_temp145"
    indx_of_weather_daily_vals = grep(paste0("^", weather_EC, "_"), colnames(Wdat)) # <- NOTE!!! This underscore is critical to avoid "P" also matching "PARMean" and "PhotoperiodMean
    
    # this is in nObs x Daily value (145)
    as.matrix(
      Wdat[, indx_of_weather_daily_vals]) %>% 
      t # now it's Daily value x nObs
  }, 
  weather_ECs) # setNames() makes the matrices in this list accessible by name



summarise_Env <- function(E, w = 3, summary_functions = NULL){
    # Time bins
    windows <- cut_interval(1:nrow(E), length = w) # this function is from ggplot2
    # each 3 day period gets a new group:
    # [1] [0,3] [0,3] [0,3]     (3,6] (3,6] (3,6]     (6,9] ...  
    # [145] (144,147]
    # Why this isn't done with seq is beyond me. Possibly because it returns cuts?
    
    if (length(unique(windows)) == 1) {
      Z <- matrix(1, nrow = nrow(E), ncol = 1)
      # if the window length is one return a single column of ones
      
    } else {
      Z <- model.matrix( ~ windows)
      # for window size three this becomes
      
      #   intercept win2 win3 ...
      # 1         1    0    0      # The first group is the intercept
      # 2         1    0    0
      # 3         1    0    0
      # 4         1    1    0      # One hot encoded groups
      # 5         1    1    0
      # 6         1    1    0
      # 7         1    0    1
      # 8         1    0    1
      # 9         1    0    1
      # ...
    }
    
    # Average by time bin
    EC <- crossprod(Z,    # 145x49   window design matrix
                    E     # 145xnObs data matrix
    ) %>% # 49xnObs 
      t #%>%          # flip nObsx49
    # scale %>%      # center and scale each column
    # replaced because this is introducing nas into 0s for `N`, `P` where columns are all 0 (i.e. when no fertilizer was applied)
    # scale each column one at a time if and only if the sd != 0
    # This prevents values from becoming NA and thus keeps the dimensions to the expected size
    for(i in seq(1, ncol(EC))){
      if(sd(EC[,i]) != 0){
        EC[,i] = scale(EC[,i])
      } else {
        EC[,i] = scale(EC[,i], scale = FALSE)        
      }
    }
    # and return to processing as normal
    EC <- EC %>% 
      t %>%          # flip 49xnObs
      na.omit
    
    # Summary by time bin
    # This functionality is not used
    if (!is.null(summary_functions)) {
      EC_summary <- foreach(summary_function = summary_functions) %do% {
        summary_by_window <-
          by(E, windows, function(x)
            apply(x, 2, summary_function))
        
        do.call(rbind, summary_by_window) %>% t %>% scale %>% t %>% na.omit
      } %>% as.list
      
      EC <- do.call(rbind, append(list(EC), EC_summary))
    }
    # Environmental relationship matrix
    return(t(EC)) # Return a nObsxWindows matrix
  }


E_summary_list <- list()
for(i in seq(1, length(env.weather))){
  print(paste0(as.character(i), "/", as.character(length(env.weather))))
  tic <- Sys.time()
  E_summary_list[[length(E_summary_list)+1]] <- summarise_Env(env.weather[[i]])  
  toc <- Sys.time()
  print(toc - tic)
}


# Save out R list
saveRDS(E_summary_list , file = matrix_file_path)
# Save each item in the R list separately

for (i in seq(1, length(E_summary_list))){
  WEC <- weather_ECs[i]
  print(WEC)
  write.csv(E_summary_list[i], paste0("./data_matrices/", WEC, "_summary.csv"))
}
