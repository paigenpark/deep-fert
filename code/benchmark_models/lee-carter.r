##### Lee-Carter #####

# vector of required packages
required_packages <- c("demography", "tidyverse", "reshape2", "glue", "here")

# function to check and install missing packages 
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# apply the function to all required packages
invisible(lapply(required_packages, install_if_missing))

# load the libraries
lapply(required_packages, library, character.only = TRUE)


# load and prepare data
path <- here("data")
asfr_training <- read.table(paste(path, "asfr_training.txt", sep = "/"), 
                              header = FALSE)
countries <- unique(asfr_training[,1])
ages <- unique(asfr_training[,3])
forecasted_years <- 2006:2015
colnames(asfr_training) <- c('Country', 'Year', 'Age', 'Rate')

fitted_results <- list()
forecasted_results <- list()


for (iter in 1:5) {
  set.seed(iter)
  for (i in countries) {
      filtered <- asfr_training |>  
        filter(asfr_training[,1] == i) 
      # get number of years available for country/gender combo
      years <- sort(unique(filtered$Year))

      # get fx matrix
      fx_df <- filtered |>
        pivot_wider(names_from = 'Year',
                    values_from = 'Rate') |>
        select(-Age, -Country)
      fx_mat <- as.matrix(fx_df)
      fx_mat[fx_mat == 0 | is.na(fx_mat)] <- 1e-05
      
      # get exposure matrix
      Ext <- matrix(1, nrow = nrow(fx_mat), ncol = ncol(fx_mat))
      
      # create demogdata object for lc function
      data <- demogdata(
        data = fx_mat,
        pop = Ext,
        ages = ages,
        years = years,
        type = "fertility",
        label = i
      )
      
      # run lc fitting function
      lc_output <- lca(data,
                       years = years,
                       ages = ages,
                       adjust = 'none')
      
      # prep fitted results
      fitted <- exp(lc_output$fitted$y)
      df_fitted <- as.data.frame(fitted)
      df_fitted$age <- ages
      df_fitted_long <- melt(df_fitted, id.vars = "age", 
                             variable.name = "year",
                             value.name = "rate")
      df_fitted_long$year <- rep(years, each = length(ages))
      df_fitted_long$country <- i
      fitted_results[[paste(i, sep = "_")]] <- df_fitted_long
      
      # get/prep forecasts  
      forecasted <- forecast(lc_output, h=10) 
      forecasted_rates <- do.call(cbind, forecasted$rate[1])
      df_forecasted <- as.data.frame(forecasted_rates)
      df_forecasted$age <- ages
      df_forecasted_long <- melt(df_forecasted, id.vars = "age", 
                             variable.name = "year",
                             value.name = "rate")
      df_forecasted_long$year <- rep(forecasted_years, each = length(ages))
      df_forecasted_long$country <- i
      forecasted_results[[paste(i, sep = "_")]] <- df_forecasted_long
  }
  
  
  final_fitted_df <- bind_rows(fitted_results)
  final_fitted_df <- final_fitted_df |>
    select(country, year, age, rate)
  
  final_forecasted_df <- bind_rows(forecasted_results)
  final_forecasted_df <- final_forecasted_df |>
    select(country, year, age, rate)  

  # save forecasts
  write.table(final_forecasted_df, paste(path, glue("lc_forecast_{iter}.csv"), sep = "/"), 
              sep=",", col.names = FALSE,
              row.names = FALSE)

  print(glue("Iteration {iter} complete – saved to lc_forecast_{iter}.csv"))
}


