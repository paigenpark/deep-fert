##### Lee-Carter #####

# vector of required packages
required_packages <- c("demography", "tidyverse", "glue", "here")

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



for (i in countries) {
    filtered <- asfr_training |>  
      filter(Country == i) 
    
    ages <- sort(unique(filtered$Age))
    years <- sort(unique(filtered$Year))


    # get fx matrix
    fx_df <- filtered |>
      arrange(Age, Year) |> # Ensure sorting matches ages/years vectors
      pivot_wider(names_from = 'Year',
                  values_from = 'Rate') |>
      select(-Age, -Country)
    fx_mat <- as.matrix(fx_df)
    rownames(fx_mat) <- ages
    colnames(fx_mat) <- years

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
      label = i,
      name = "female"
    )
    
    # run lc fitting function
    lc_output <- lca(data,
                      years = years,
                      ages = ages,
                      adjust = 'none')

    
    prep_results <- function(rates_mat, year_vector, country_id) {
      df <- as.data.frame(rates_mat)
      colnames(df) <- year_vector # Ensure columns are named by year
      df$age <- ages
      
      df_long <- df |> 
        pivot_longer(cols = -age, names_to = "year", values_to = "rate") |>
        mutate(
          year = as.numeric(year), # pivot_longer makes names strings
          country = country_id
        )
      return(df_long)
    }
    
    # prep fitted results
    fitted <- exp(lc_output$fitted$y)
    fitted_results[[i+1]] <- prep_results(fitted, years, i)

    # get/prep forecasts  
    forecasted <- forecast(lc_output, h=10) 
    forecasted_rates <- forecasted$rate$female
    forecasted_results[[i+1]] <- prep_results(forecasted_rates, forecasted_years, i)
}


final_fitted_df <- bind_rows(fitted_results)
final_fitted_df <- final_fitted_df |>
  select(country, year, age, rate)

final_forecasted_df <- bind_rows(forecasted_results)
final_forecasted_df <- final_forecasted_df |>
  select(country, year, age, rate)  

# save forecasts
write.table(final_forecasted_df, paste(path, "lc_forecast.csv", sep = "/"), 
            sep=",", col.names = FALSE,
            row.names = FALSE)

print("Forecasts saved to lc_forecast.csv")



