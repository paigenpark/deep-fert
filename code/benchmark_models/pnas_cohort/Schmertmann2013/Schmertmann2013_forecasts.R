##########################################################################
# Carl Schmertmann
# 15 Nov 2013
#
# For a selected forecast.year (usually 2010, but sometimes 1985), this program 
# calculates the forecast/posterior distribution for a surface over
#   30 ages, 15..44
#   40 cohorts, forecast.year - (54:15)   [ usually 1956..1995]
#   10 initial cohorts forecast.year - (54:15) [ usually 1956..1965] with complete 
#         (or very nearly complete*) data
#
#  using any data that would have been available in that forecast.year
#
#   * if only a few rates are missing for the earliest cohorts at the
#     youngest ages (e.g. Australia in a 2010 forecast), then we do a "freeze-rate" backcast as in 
#     Myrskylä et al 2013 to fill in and make the forecast possible
##########################################################################

rm(list=ls())
graphics.off()
library(MASS)
library(Matrix)

forecast.year = 2010

infile       = "calibrated penalty matrix.dput"
Kjfile       = "unweighted penalty matrices.RData"
forecastFile = paste("Forecast ",format(Sys.time(), "%a %d %b %Y %H%M"),".dput", sep="")

## read data
source("input rate and exposure data.R")

## set up a list that will contain forecast information
forecast        = vector("list", length(rates))
names(forecast) = names(rates)

## retrieve the calibrated penalty matrix
K = dget(infile)

## retrieve the unweighted penalty matrices K1...K90
load(Kjfile)

## loop over countries in the dataset
for (k in names(rates)) {
  
  print(paste("... starting ", full.name[k]))
  
  last.coh = forecast.year - 15
  coh      = last.coh - 39:0
  
  surf  = matrix(NA, 30, 40, dimnames=list(age, coh))
  women = matrix(NA, 30, 40, dimnames=list(age, coh))
  
  Data = rates[[k]]
  Expo = expos[[k]]
  
  ix              = match( dimnames(surf)[[2]], dimnames(Data)[[2]])
  oksurf          = which(!is.na(ix))
  okdata          = ix[!is.na(ix)]
  surf[, oksurf]  = Data[, okdata]
  women[, oksurf] = Expo[, okdata]  
  
  ## backproject rates under 20, if any are missing for the first 5 cohorts
  ##  this will allow projections for a few cases (eg Australia) without any meaningful
  ##  change in the forecast
  
  gap = any( is.na(surf[paste(15:19), 1:5]))
  if (gap) {
    for (x in 15:19) {
      bad = is.na(surf[paste(x), 1:5])
      if (any(bad)) {
	  
        igap  = which(bad)   # which cohort cols 1-5 have missing data?
        inext = 1+max(igap)
		
        surf[paste(x), igap]  = surf[paste(x), inext]
        women[paste(x), igap] = women[paste(x), inext]
		
      } # if any bad
    } # for x
  } # if gap
  
  ## vis is a 30 x 40 logical matrix: =TRUE if a cell is visible in this forecast year
  has.data = !is.na(surf)
  masked   = outer(age,coh,"+") > forecast.year
 
  vis  = has.data & !masked
  nvis = sum(vis)

  if (nvis > 0) {

    try( {
      
      complete.at.forecast = apply( matrix(vis,nrow=A), 2,all)  # at forecast date, which coh had complete histories
      
      V   = diag(length(surf))[vis,]
    
      PSIINV = diag( women[vis] / pmax(surf[vis],1e-6))   # inverse variances of observed estimates on diagonal of large matrix
      
      ## calculate posterior variance matrix VPOST and mean vector MPOST 
      VPOST = solve( t(V) %*% PSIINV %*% V + K)
      MPOST = VPOST %*% t(V) %*% PSIINV %*% surf[vis]
      
      ## convert MPOST to an A x C surface
      mu = matrix(MPOST, nrow=A, dimnames=list(age, coh))
      ss = matrix(sqrt(diag(VPOST)), nrow=A, dimnames=list(age, coh))
  
      ## save results for post-processing if desired
      forecast[[k]]$forecast.year = forecast.year
      forecast[[k]]$last.coh      = last.coh
      
      forecast[[k]]$vis  = vis
      forecast[[k]]$surf = surf
      
      forecast[[k]]$mu  = round(mu,5)
      forecast[[k]]$sd  = round(ss,6)
          
      ##### CFRs ####
      C = diag(40) %x% matrix(1,1,30)
      
      mu.cfr = as.vector( C %*% MPOST)
      sd.cfr = sqrt(diag( C %*% VPOST %*% t(C)    ))
      names(mu.cfr) = coh
      names(sd.cfr) = coh
      
      forecast[[k]]$mu.cfr = round( mu.cfr, 4)
      forecast[[k]]$sd.cfr = round( sd.cfr, 6)
      
      ## penalties pi[1]...pi[J]

      pen = vector("numeric", length(Kj))
      names(pen) = names(Kj)
      for (j in names(pen)) {
        pen[j] = as.numeric( t(MPOST) %*% Kj[[j]] %*% MPOST )
      }
      
      ## save the penalties pi[1]...pi[J] for this forecast
      forecast[[k]]$pi = round(pen,4)
      
      ##### if this is a 2010 forecast, also calculate and save the posterior means and SDs of
      ######  some selected cross-cohort differences in CFR
      
      if (forecast.year == 2010) {
        
        ## D %*% theta gives CFR1970-CFR1960, CFR1980-CFR1970, CFR1990-CFR1980)
        D = rbind( 
          (matrix( 1*(coh==1970), nrow=1) - matrix( 1*(coh==1960), nrow=1)) %x% matrix(1,1,30),
          (matrix( 1*(coh==1980), nrow=1) - matrix( 1*(coh==1970), nrow=1)) %x% matrix(1,1,30),
          (matrix( 1*(coh==1990), nrow=1) - matrix( 1*(coh==1980), nrow=1)) %x% matrix(1,1,30),
          
          (matrix( 1*(coh==1975), nrow=1) - matrix( 1*(coh==1965), nrow=1)) %x% matrix(1,1,30),
          (matrix( 1*(coh==1985), nrow=1) - matrix( 1*(coh==1975), nrow=1)) %x% matrix(1,1,30),
          (matrix( 1*(coh==1995), nrow=1) - matrix( 1*(coh==1985), nrow=1)) %x% matrix(1,1,30)
          
          )
        rownames(D) = c("CFR70-CFR60","CFR80-CFR70","CFR90-CFR80",
                        "CFR75-CFR65","CFR85-CFR75","CFR95-CFR85")
        
        mu.change = as.vector(D %*% MPOST)
        names(mu.change) = rownames(D)
        
        sd.change = sqrt( diag( as.matrix(D %*% VPOST %*% t(D)) ) )
        names(sd.change) = rownames(D)
        
        forecast[[k]]$mu.change = round(mu.change, 3)
        forecast[[k]]$sd.change = round( sd.change, 4)
              
      } # if forecast.year = 2010
    }) # try
  } # if nvis > 0
  } # for k

## write the entire forecast to a file for later processing
dput(forecast, file=forecastFile)

print(forecastFile)
