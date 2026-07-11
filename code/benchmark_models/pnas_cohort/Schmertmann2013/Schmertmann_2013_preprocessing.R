###############################################################################################
## Carl Schmertmann
## 6 Nov 2013
##
## This script is intended to be run inside of other R programs, using source( )
## It reads and arranges input data, as follows
##  (1) reads the HFD rate and exposure data 
##  (2) appends similarly formatted non-HFD data from Australia, Belgium, Denmark,
##      Greece, Iceland, Italy, Japan, Korea, Luxembourg, New Zealand, Romania, and Singapore
##  (3) appends differently-formated data from Brazil
##
## Results that are ready for use by other programs after this script is run include:
##  (A) a series of lists, one per country, with ASFR and related data (as described just below)
##  (B) SVD components for schedule shapes [X]
##  (C) residual projection matrix [M]
##  (D) empirical covariance estimate for shape residuals [omega]
##  (E) empirical inverse covariance estimate for shape residuals [omega.plus]
###############################################################################################

## because each country has data over different calendar years and cohorts, 
## data are collected in lists:

rates    = list()   # age x cohort ASFR matrices for each country
crates   = list()   # age x cohort ASFR matrices for each country
cohort   = list()   # numeric vectors containing cohorts included in each country's data (e.g [1917,1918,...1995])
sched    = list()   # a characther version of cohort, with country names appended (mostly for pretty output)
complete = list()   # logical vector indicating which cohort schedules for this country are complete
expos    = list()   # age x cohort exposure matrices for each country
origin   = list()   # where did each country's data come from -- HFD, Myrsklya et al., Lima

age = 15:44         # which ages do we need for a 'complete' schedule?
A   = length(age)   # an important constant

################################### 
#  READ HFD DATA  
###################################

FF  <-  read.table("asfrVV.txt",skip=2,as.is=T, header=T)     # HFD rate estimates (cohort x age)
WW  <-  read.table("exposVV.txt",skip=2,as.is=T, header=T)    # HFD exposure data (cohort x age)

for (k in unique(FF$Code)) {
  
  ## selected subset for rates data
  sel = ( (FF$Code==k) & (FF$ARDY %in% age) )
  
  origin[[k]]    = "HFD download 2 Nov 2011"
  rates[[k]]     = tapply( FF$ASFR[sel], list( FF$ARDY[sel], FF$Cohort[sel]), sum) 
  
  complete[[k]]  = apply( rates[[k]], 2, function(x) !any(is.na(x)) )
  crates[[k]]    = apply( rates[[k]], 2, cumsum)
  cohort[[k]]    = as.numeric(dimnames(rates[[k]])[[2]])
  sched[[k]]     = paste(k, cohort[[k]], sep="")
  
  ## (possibly) different selected subset for exposure data
  sel            = ( (WW$Code==k) & (WW$ARDY %in% age) )
  expos[[k]]     = tapply( WW$Exposure[sel], list( WW$ARDY[sel], WW$Cohort[sel]), sum)  
  
} ## for k

################################################## 
#  READ additional HFD-style DATA 
#    for 12 more countries Belgium...Singapore
##################################################

FF <- read.csv("additional.data.asfr.csv", as.is=T, header=T)
WW <- read.csv("additional.data.expos.csv", as.is=T, header=T)

for (k in unique(FF$Code)) {
  
  ## selected subset for rates data
  sel = ( (FF$Code==k) & (FF$ARDY %in% age) )
  
  origin[[k]]    = "Myrskyla et al. Jan 2013, as in PDR 39(1)"
  rates[[k]]     = tapply( FF$ASFR[sel], list( FF$ARDY[sel], FF$Cohort[sel]), sum) 
  
  complete[[k]]  = apply( rates[[k]], 2, function(x) !any(is.na(x)) )
  crates[[k]]    = apply( rates[[k]], 2, cumsum)
  cohort[[k]]    = as.numeric(dimnames(rates[[k]])[[2]])
  sched[[k]]     = paste(k, cohort[[k]], sep="")
  
  ## (possibly) different selected subset for exposure data
  sel            = ( (WW$Code==k) & (WW$ARDY %in% age) )
  expos[[k]]     = tapply( WW$Exposure[sel], list( WW$ARDY[sel], WW$Cohort[sel]), sum)
  
} ## for k

################################################## 
#  READ BRAZILIAN DATA (formatted differently, 
#     without exposure data)
##################################################

B = read.csv("Brazil 1966-2010 ASFR.csv", as.is=T, skip=4, row.names=1)

brazil.ages  = 15:49
brazil.years = 1966:2010

brazil.cohorts = outer(-brazil.ages, brazil.years,"+" )
clist          = sort( unique( as.vector(brazil.cohorts)))
ncohort        = length(clist)

Brazil = matrix(NA, length(brazil.ages), ncohort, 
                dimnames=list(brazil.ages,clist))

AGE = matrix( brazil.ages[row(B)], ncol=ncol(B))

for (coh in clist) {
  # any non-missing rates
  ff =  as.matrix(B)[ brazil.cohorts==coh]
  aa =  AGE[ brazil.cohorts==coh]
  
  Brazil[paste(aa), paste(coh)] = ff
}

k = "Brazil"

origin[[k]]    = "personal communication from Everton Lima, Fall 2012"
rates[[k]]     = Brazil[paste(age),]   # limit to 15-44
complete[[k]]  = apply( rates$Brazil, 2, function(x) !any(is.na(x)) )
crates[[k]]    = apply( rates[[k]], 2, cumsum)
cohort[[k]]    = as.numeric(dimnames(rates[[k]])[[2]])
sched[[k]]     = paste(k, cohort[[k]], sep="")
expos[[k]]     = matrix( 500000, 30, length(cohort[[k]]), dimnames=list(age,cohort[[k]]))  # an arbitrary approx

###############################################################

full.name = c(
  AUT         = "Austria",
  BGR         = "Bulgaria",
  CAN         = "Canada",
  CZE         = "Czech Rep",
  EST         = "Estonia",
  FIN         = "Finland",
  FRATNP      = "France",
  DEUTNP      = "Germany",
  DEUTW       = "Western Germany",
  DEUTE       = "Eastern Germany",
  HUN         = "Hungary",
  LTU         = "Lithuania",
  NLD         = "Netherlands",
  PRT         = "Portugal",
  RUS         = "Russia",
  SVK         = "Slovakia",        
  SVN         = "Slovenia",
  SWE         = "Sweden",
  CHE         = "Switzerland",
  GBR_NP      = "Great Britain",
  GBRTENW     = "England and Wales",
  GBR_SCO     = "Scotland",
  GBR_NIR     = "Northern Ireland",
  USA         = "USA",   
  Australia   = "Australia",
  Belgium     = "Belgium",
  Denmark     = "Denmark",
  Greece      = "Greece",
  Iceland     = "Iceland",
  Italy       = "Italy",
  Japan       = "Japan",
  Korea       = "Korea",
  Luxembourg  = "Luxembourg",
  New_Zealand = "New Zealand",
  Romania     = "Romania",
  Singapore   = "Singapore",
  Brazil      = "Brazil"
)  
 
## calculate 
##     SVD components for schedule shapes [X]
##     residual projection matrix [M]
##     empirical covariance estimate for shape residuals [omega]
##     empirical inverse covariance estimate for shape residuals [omega.plus]

### CALIBRATE SHAPE PENALTY using data from cohorts born before 1950 ##########

## assemble the historical set of cohorts born 1900-1949 that have complete schedules
PHI = NULL

for (k in names(rates)) {
  keep = complete[[k]] & (cohort[[k]] %in% 1900:1949)
  tmp  = rates[[k]][, keep ]
  PHI  = cbind(PHI, tmp)
}

## Singular value decomposition PHI = U D V'
ss = svd(PHI)

## keep the first 3 components, with signs flipped to make basis functions prettier
X = ss$u[,1:3] %*% diag(c(-1,+1,-1))   

## orthogonal projection matrix for X components
M = diag(A) - X %*% solve( t(X) %*% X) %*% t(X)

## orthogonal residuals 
eps = M %*% PHI

## average outer product (=estimated covariance of shape residuals in the historical data)
omega      = eps %*% t(eps) / ncol(eps)

## slight cleanup -- the last several eigenvalues of omega should be zero, but because of 
##  precision errors they may be extremely small negative numbers.  Because of this,
##   calculate the Moore-Penrose inverse in a way that ensures zero eigenvalues.

nullity = qr(X)$rank
r       = ncol(omega) - nullity

ei = eigen(omega)
Ur = ei$vec[,1:r]        # cols of Ur are eigenvectors corr. to + eigenvalues
Dr = diag( ei$val[1:r])

## generalized inverse of omega
omega.plus = Ur %*% solve(Dr) %*% t(Ur)

