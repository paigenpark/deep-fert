###############################################################################################
## Carl Schmertmann
## 15 Nov 2013
##
## Calibrate a combined penalty matrix for a 30 age x 40 cohort surface
##  using penalties for "freeze rates", "freeze slopes", and "shapes"
##  save the combined (1200 x 1200) penalty matrix to a file
###############################################################################################

## clear workspace
rm(list=ls())

## the next three commands are Windows housekeeping items. They may require editing
##  if running on a non-Windows system
graphics.off()
windows(record=TRUE)
op = par(no.readonly=TRUE)

## where will the main output be saved?
outfile = "calibrated penalty matrix.dput"

## where will the individual penalty matrices be saved?
##  note: because these are sparse Matrices, they have to
##        be stored with save() rather than dput()
KjFile = "unweighted penalty matrices.RData"

library(MASS)
library(Matrix)

## read rates and exposure, set up data for analysis 
source("input rate and exposure data.R")

#################################################################
## some necessary weights for constructing time-series residuals
## (notice the order!)
##
## freeze slope residuals are
##     -4/30 * theta[t-5] -3/30 * theta[t-4] ... +(10/30+1) * theta[t-1] - theta[t]
##
## freeze rate residuals are
##     theta[t-1] - theta[t]
#################################################################

time.weights = list(
   'freeze.slope' = c( -4/30, -3/30, -2/30, -1/30, 10/30 + 1, -1),
   'freeze.rate'  = c(1,-1)
 )

############################################

### CALIBRATE FREEZE AND PIN PENALTIES (SCALAR VARIANCES, ONE PER AGE)

placeholder        = NA*age
names(placeholder) = age

result        = vector("list",  length(time.weights))
names(result) = names(time.weights)

for (this.country in names(result)) result[[this.country]] = list(mu=placeholder, sd=placeholder)

## loop over the various types of time-series residuals. For each type, calculate
##  age-specific means and SDs over the historical rate data 

for (typ in names(time.weights)) {
  
  wt = time.weights[[typ]]
  n  = length(wt)
  
  for (this.age in paste(age)) {
    
    x = NULL
    
    for (this.country in names(rates)) {
      
      y = rates[[this.country]][this.age, (cohort[[this.country]] %in% 1900:1949)]
      y = y[is.finite(y)]
  
      if (length(y) >= n) {
        AA = matrix(0, length(y)-(n-1), length(y))
        for (j in n:length(y)) {
          AA[ j-n+1, j-((n-1):0)] = wt
        } # for
        x = c(x, AA %*% y)    
      }  # if
  
      } # for this.country
        
    result[[typ]]$mu[this.age] = mean(x, na.rm=TRUE)
    result[[typ]]$sd[this.age] = sd(x, na.rm=TRUE)
    
    } # for this.age
} # for typ
  

#####################################################################
## construct a big list of sparse (1200 x 1200) matrices for a 30x40 surface
##   The time series priors apply only the the last 30 of the 40 cohorts, 
##   NOT the first 10 (which will always be complete)
##
##   1 freeze.rate penalty matrix per age (30 total)
##   1 freeze.slope penalty matrix per age (30 total)
##   1 shape penalty per cohort (40 total)
##

evec = function(i,n) matrix( diag(n)[,i], ncol=1)

Kj     = list()
target = NULL
group  = NULL
ageref = NULL

## shape (1 penalty per cohort 11...40)
Mrank = qr(M)$rank

for (j in 11:40) {
  nm         = paste("shape",j,sep="")
  group[nm]  = 'shape'
  ageref[nm] = 0      # meaning "all"
  target[nm] = Mrank
  MG         = t(evec(j,40)) %x% M
  Kj[[nm]]   = Matrix( t(MG) %*% omega.plus %*% MG, sparse=TRUE)    # text Eq 10
}

## time series penalties (1 per age for each type)

for (typ in names(time.weights)) {
  
  wt = time.weights[[typ]]
  n  = length(wt)
  
  W   = matrix( 0, 40-(n-1), 40)
  cmr = col(W) - row(W)
  for (j in 1:n) W[ cmr==(j-1)] = wt[j]
  
  ## keep only the last 30 rows of W -- i.e., those that apply to cohorts 11..40
  W = W[ nrow(W)-29:0, ]
  
  for (x in age) {
    nm         = paste(typ,x,sep="")
    target[nm] = nrow(W)
    group[nm]  = typ
    ageref[nm] = x
    ix         = which(age==x)
    s2         = ( result[[typ]]$sd[paste(x)])^2
    WH         = W %*% (diag(40) %x% t(evec(ix,30))) 
    Kj[[nm]]   = 1/s2 * Matrix( t(WH) %*% WH, sparse=TRUE)    # text Eq 13 (freeze-rate) or 14 (freeze-slope)
  } # for x
  
} # for typ


##############################################################################################
##  CALIBRATION
##  Iteratively adjust weights on each penalty until the expected value of each penalty
##  matches its historical average (=27 for shape penalties, =30 for time series penalties)
##############################################################################################

tr = function(Z) sum(diag(Z))   # trace function

calibration.history = list(w=NULL, 
                           e=NULL)

new = NULL

G      = unique(group)
ngroup = length(G)

niter = 30

for (i in 1:niter) {
  
    # penalty weights (initialize on first pass, otherwise update)
    if (is.null(new)) { 
          w = rep(1, length(Kj))
        } else { 
          w = new
        }    
    
    ## construct the weighted K matrix using current weights w
    K = Matrix(0, 1200, 1200, sparse=TRUE)
    
    for (j in seq(Kj)) K = K + w[j] * Kj[[j]]
    
    tmp = ( K + t(K))/2             # stabilize numerically
    K = as( tmp, "symmetricMatrix")  
    
    ei     = eigen(K)
    
    if (i==1) {
      r = sum(ei$val > 1e-6)             #only need to do this once
      print(paste('rank of K =',r))
    }  
    
    ## calculate the generalized inverse, K+
    Ur     = ei$vec[,1:r]
    Dr.inv = diag( 1/ei$val[1:r]) 
    
    tmp   = Ur %*% Dr.inv %*% t(Ur)
    tmp   = (tmp + t(tmp))/2  # stabilize
    Kplus = as( tmp, "symmetricMatrix")
    
  
    ## calculate the expected values (over the non-null space) of each penalty with the current weights
    e = vector("numeric", 0)
    
    for (j in names(Kj)) {   e[j]  = tr( Kj[[j]] %*% Kplus)  }
    
    ## create a new set of updated weights
    new = w
    new = w * e/target
    
    ## record the iteration history for possible analysis
    calibration.history$w  = cbind(calibration.history$w, w)  
    calibration.history$e  = cbind(calibration.history$e, e)  
    
    ## print the current state of the search, and make a diagnostic plot of how we're doing so far
    tmp = data.frame(n        = names(e), 
                     target   = target, 
                     expected = e, 
                     new      = new)
    
    print(paste("--------",i,"--------"))
    print(tmp)
    
    plot( e, pch=16, ylim=c(0,40), main=paste("Before Iteration #",i,"of",niter))
    points(calibration.history$e[,1], pch=4)  
    points(seq(target), e, pch=16, col="blue")      
    lines( target, type="s", col="red", lwd=2)  
    text(c(15,45,75), c(0,0,0), c("Shape","Freeze-Rate","Freeze-Slope"), col='grey')
  
} # for i


#########################################
## Final version of weighted matrix
#########################################

w = new

K = Matrix(0, 1200, 1200, sparse=TRUE)

for (j in seq(Kj)) K = K + w[j] * Kj[[j]]

# stabilize, round to 1 decimal place,  and convert to symmetric
#  rounding makes only trivial difference for our data, because non-zero K elements are very large,
#  but this is problem-specific (if unsure, don't round)

K    = (K + t(K))/2 
symK = as( round(K,1), "symmetricMatrix") 

dput( symK, file=outfile )

plot( w, type="h", main="Final penalty weights", ylim=c(0,1))
text(c(15,45,75), c(1,1,1), c("Shape","Freeze-Rate","Freeze-Slope"), col='grey')

## save the unweighted penalty matrices to a file, for later diagnostics
save( "Kj", file=KjFile)

###############################################################
## Diagnostics: summarize the "before" and "after" calibration 
##  information, for text Table 2
###############################################################

print( "------------------------------")
date()

## targets
tapply(target, group, mean)

## beginning w
tapply( calibration.history$w[,1], group, range)

## ending w
tapply( calibration.history$w[,niter], group, range)

## beginning E*
tapply( calibration.history$e[,1], group, range)

## ending E*
tapply( calibration.history$e[,niter], group, range)
