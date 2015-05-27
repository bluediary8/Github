library(darch)

install.packages("gputools")
library(Matrix)
library(devtools)
devtools::install_github('zachmayer/rbm')

setwd("D:\\Desktop\\DNN\\CNN")
final<-read.csv("data.csv")
dim(data)
for(i in 1:(dim(final)[2]-2)){
  final[,i]<-as.numeric(as.vector(final[,i]))
}
head(final)
dim(final)
str(final)
colnames(final)<-c(paste0("X",1:(dim(final)[2]-3)),"Y","실제","파일명")

sam<-sample(1:dim(final)[1],round(dim(final)[1]*0.8,0))
x<-final[sam,-c((dim(final)[2]-2):dim(final)[2])]


x<-x[,1:15]



num_hidden = 10; max_epochs = 10; learning_rate = 0.1;
use_mini_batches = TRUE; batch_size = 20; initial_weights_mean = 0;
initial_weights_sd = 0.1; momentum = 0; dropout = FALSE; dropout_pct = .50;
retx = FALSE; activation_function=NULL; verbose = FALSE


rb<-rbm(x)
str(rb)
dim(x)
head(rb$rotation)
plot.rbm(rb)
p<-predict.rbm(rb, type='states',as.matrix(final[-sam,-c((dim(final)[2]-2):dim(final)[2])]))
str(p)
p@Dim



ss<-matrix(c(1,1,1,0,0,0,
         1,0,1,0,0,0,
         1,1,0,0,0,0,
         0,1,1,0,0,0,
         0,1,1,1,0,0,
             1,0,0,1,0,0),nrow=6,byrow=T)


ssk<-matrix(c(0,0,0,1,1,1,
             0,0,0,1,0,1,
             0,0,1,1,1,0,
             0,0,1,1,0,1,
             0,1,0,1,0,1,
             1,0,0,1,0,1),nrow=6,byrow=T)

ss2<-cbind(1,ss)
ssk2<-cbind(1,ssk)
dim(ss2)
dim(as.matrix(rrr$rotation))
oho<-ss2 %*% as.matrix(rrr$rotation);oho



oho2<-ssk2 %*% as.matrix(rrr$rotation);oho2
apply(oho,2,sum);apply(oho,1,sum)
apply(oho2,2,sum);apply(oho2,1,sum)
str(rrr)

rrr<-rbm(ss,num_hidden=2,max_epochs = 100, learning_rate = 0.01,use_mini_batches = F)



hap<-t(cbind(ss,ssk))
haprrr<-rbm(hap,num_hidden=2,max_epochs = 1000, learning_rate = 0.05,use_mini_batches = F)
haprrr$rotation
hap2<-cbind(1,hap)
oho3<-hap2 %*% as.matrix(haprrr$rotation);oho3
apply(oho3,1,sum)
dim(hap2)
dim(as.matrix(haprrr$rotation))



x<-as.matrix(train2[rows,-c(1)])
num_hidden=30;max_epochs =500; learning_rate = 0.11;use_mini_batches = T;
batch_size = 30
rbm <- function (x, num_hidden = 10, max_epochs = 1000, learning_rate = 0.1, use_mini_batches = TRUE, batch_size = 250, initial_weights_mean = 0, initial_weights_sd = 0.1, momentum = 0, dropout = FALSE, dropout_pct = .50, retx = FALSE, activation_function=NULL, verbose = FALSE, ...) {
  
  #Checks
  stopifnot(length(dim(x)) == 2)
  if(any('data.frame' %in% class(x))){
    if(any(!sapply(x, is.finite))){
      stop('x must be all finite.  rbm does not handle NAs, NaNs, Infs or -Infs')
    }
    if(any(!sapply(x, is.numeric))){
      stop('x must be all finite, numeric data.  rbm does not handle characters, factors, dates, etc.')
    }
    x = Matrix(as.matrix(x), sparse=TRUE)
#     head(x)
#     str(x)
  
    } else if (any('matrix' %in% class(x))){
    x = Matrix(x, sparse=TRUE)
  } else if(length(attr(class(x), 'package')) != 1){
    stop('Unsupported class for rmb: ', paste(class(x), collapse=', '))
  } else if(attr(class(x), 'package') != 'Matrix'){
    stop('Unsupported class for rmb: ', paste(class(x), collapse=', '))
  }
  if(use_mini_batches & nrow(x) < batch_size){
    warning(paste0('Batch size (', batch_size, ') less than rows in x (', nrow(x), '). Re-setting batch size to nrow(x)'))
    batch_size <- nrow(x)
  }
  
  x_range <- range(x)
  scaled <- FALSE
  if (x_range[1] < -0.5 | x_range[2] > 1){
    warning("x is out of bounds, automatically scaling to 0-1, test data will be scaled as well")
    x <- (x - x_range[1]) / (x_range[2] - x_range[1])
    scaled <- TRUE
  }
  
  stopifnot(is.numeric(momentum))
  stopifnot(momentum >= 0 & momentum <=1)
  if(momentum>0){warning('Momentum > 0 not yet implemented.  Ignoring momentum')}
  
  stopifnot(is.numeric(dropout_pct))
  stopifnot(dropout_pct >= 0 & dropout_pct <1)
  if(dropout){warning('Dropout not yet implemented')}
  
  if(is.null(activation_function)){
    activation_function <- function(x){1.0 / (1 + exp(-x))}
  }
  
  #Check if greater than 1
  
  # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
  # a Gaussian distribution with mean 0 and standard deviation 0.1.
  #momentum_speed <- sparseMatrix(1, 1, x=0, dims=c(p, num_hidden))
  weights = matrix(rnorm(num_hidden*ncol(x), 
                         mean=initial_weights_mean, 
                         sd=initial_weights_sd), nrow=ncol(x), ncol=num_hidden)
  # Insert weights for the bias units into the first row and first column.
  weights = cbind(0, weights)
  weights = rbind(0, weights)
  weights = Matrix(weights, sparse=TRUE)
  head(weights)
  # Insert bias units of 1 into the first column.
  x <- cBind(Bias_Unit=1, x)
  head(x)
  dimnames(weights) = list(colnames(x), c('Bias_Unit', paste('Hidden', 1:num_hidden, sep='_')))
  
  #Fit the model
  x <- drop0(x)
 
  error_stream <- runif(max_epochs)
  for (epoch in 1:max_epochs){
    cat("\t",epoch)
    #Sample mini-batch
    if(use_mini_batches){
      train_rows = sample(1:nrow(x), batch_size, replace=TRUE)
      x_sample = x[train_rows,,drop=FALSE]
    } else {
      x_sample = x
    }
    dim(x_sample)
    head(x_sample)
    dim(weights)
    # Clamp to the data and sample from the hidden units.
    # (This is the "positive CD phase", aka the reality phase.)
    pos_hidden_activations = x_sample %*% weights
    if(dropout){
      pos_hidden_activations_dropped = pos_hidden_activations
      pos_hidden_activations_dropped@x[runif(length(pos_hidden_activations_dropped@x)) < dropout_pct] = 0
      pos_hidden_activations_dropped[,1] <- pos_hidden_activations[,1]
      pos_hidden_activations <- pos_hidden_activations_dropped
    }
    dim(pos_hidden_activations)
    pos_hidden_probs = activation_function(pos_hidden_activations)
    pos_hidden_states = pos_hidden_probs > Matrix(runif(nrow(x_sample)*(num_hidden+1)), nrow=nrow(x_sample), ncol=(num_hidden+1))
    
    # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
    # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
    pos_associations = crossprod(x_sample, pos_hidden_probs)
    
    
    dim(weights)
    dim(pos_hidden_states)
    tcrossprod(pos_hidden_states, weights)
  
    # Reconstruct the visible units and sample again from the hidden units.
    # (This is the "negative CD phase", aka the daydreaming phase.)
    neg_visible_activations = tcrossprod(pos_hidden_states, weights)
    neg_visible_probs = activation_function(neg_visible_activations)
    neg_visible_probs[,1] = 1 # Fix the bias unit.
    neg_hidden_activations = neg_visible_probs %*% weights
    neg_hidden_probs = activation_function(neg_hidden_activations)
    
    # Note, again, that we're using the activation *probabilities* when computing associations, not the states
    # themselves.
    neg_associations = crossprod(neg_visible_probs, neg_hidden_probs)
    
    
    head(  pos_associations)
    head(neg_associations)
    # Update weights
    weights = weights + learning_rate * ((pos_associations - neg_associations) / nrow(x_sample))
    
    #Print output
    error = sum((x_sample - neg_visible_probs) ^ 2)
    error_stream[[epoch]] <- error
    if(verbose){
      print(sprintf("Epoch %s: error is %s", epoch, error))
    }
  }
  
  #Return output
  if(retx){
    output_x <- x %*% weights
  } else {
    output_x <- NULL
  }
  out <- list(rotation=weights, activation_function=activation_function, x=output_x, error=error_stream, max_epochs=max_epochs, x_range=x_range, scaled=scaled)
  class(out) <- 'rbm'
  return(out)
}


print.rbm <- function (x, ...) {
  print(x$rotation)
}



plot.rbm <- function (x, ...) {
  plot(x$error, ...)
}


predict.rbm <- function (object, newdata, type='probs', omit_bias=TRUE, ...) {
  if (missing(newdata)) {
    if (!is.null(object$x)) {
      hidden_activations <- object$x
      rows <- nrow(object$x)
    }
    else stop("no scores are available: refit with 'retx=TRUE'")
  } else {
    #Checks
    stopifnot(length(dim(newdata)) == 2)
    stopifnot(type %in% c('activations', 'probs', 'states'))
    if(any('data.frame' %in% class(newdata))){
      if(any(!sapply(newdata, is.numeric))){
        stop('x must be all finite, numeric data.  rbm does not handle characters, factors, dates, etc.')
      }
      x = Matrix(as.matrix(newdata), sparse=TRUE)
    } else if (any('matrix' %in% class(newdata))){
      x = Matrix(newdata, sparse=TRUE)
    } else if(length(attr(class(newdata), 'package')) != 1){
      stop('Unsupported class for rmb: ', paste(class(newdata), collapse=', '))
    } else if(attr(class(newdata), 'package') != 'Matrix'){
      stop('Unsupported class for rmb: ', paste(class(newdata), collapse=', '))
    }
    
    #Scale if scaled during training
    if (!is.null(object$scaled)) {
      if(object$scaled){
        newdata <- (newdata - object$x_range[1]) / (object$x_range[2] - object$x_range[1])
      }
      if(min(newdata) < 0 | max(newdata) > 1){
        stop('newdata outside of the scale of the model training data.  Could not re-scale data to be 0-1')
      }
    }
    
    # Insert bias units of 1 into the first column.
    newdata <- cBind(Bias_Unit=rep(1, nrow(newdata)), newdata)
    
    nm <- rownames(object$rotation)
    if (!is.null(nm)) {
      if (!all(nm %in% colnames(newdata)))
        stop("'newdata' does not have named columns matching one or more of the original columns")
      newdata <- newdata[, nm, drop = FALSE]
    }
    else {
      if (NCOL(newdata) != NROW(object$rotation))
        stop("'newdata' does not have the correct number of columns")
    }
    hidden_activations <- newdata %*% object$rotation
    rows <- nrow(newdata)
  }
  
  if(omit_bias){
    if(type=='activations'){return(hidden_activations[,-1,drop=FALSE])}
    hidden_probs <- object$activation_function(hidden_activations)
    if(type=='probs'){return(hidden_probs[,-1,drop=FALSE])}
    hidden_states <- hidden_probs > Matrix(runif(rows*ncol(object$rotation)), nrow=rows, ncol=ncol(object$rotation))
    return(hidden_states[,-1,drop=FALSE])
  } else{
    if(type=='activations'){return(hidden_activations)}
    hidden_probs <- object$activation_function(hidden_activations)
    if(type=='probs'){return(hidden_probs)}
    hidden_states <- hidden_probs > Matrix(runif(rows*ncol(object$rotation)), nrow=rows, ncol=ncol(object$rotation))
    return(hidden_states)
  }
  
}



rbm_gpu <- function (x, num_hidden = 10, max_epochs = 1000, learning_rate = 0.1, use_mini_batches = TRUE, batch_size = 250, initial_weights_mean = 0, initial_weights_sd = 0.1, momentum = 0, dropout = FALSE, dropout_pct = .50, retx = FALSE, activation_function=NULL, verbose = FALSE, ...) {
  
  if(! require('gputools')){
    stop('The gputools package is required for this function.  Please install it, or us the "rbm" function instead (which does not require gputools)')
  }
  
  #Checks
  stopifnot(length(dim(x)) == 2)
  if(any('data.frame' %in% class(x))){
    if(any(!sapply(x, is.finite))){
      stop('x must be all finite.  rbm does not handle NAs, NaNs, Infs or -Infs')
    }
    if(any(!sapply(x, is.numeric))){
      stop('x must be all finite, numeric data.  rbm does not handle characters, factors, dates, etc.')
    }
    x = as.matrix(x)
  } else if (any('matrix' %in% class(x))){
    sink <- 1
  } else if(length(attr(class(x), 'package')) != 1){
    stop('Unsupported class for rmb: ', paste(class(x), collapse=', '))
  } else if(attr(class(x), 'package') != 'Matrix'){
    stop('Unsupported class for rmb: ', paste(class(x), collapse=', '))
  }
  
  x_range <- range(x)
  scaled <- FALSE
  if (x_range[1] <0 | x_range[2] > 1){
    warning("x is out of bounds, automatically scaling to 0-1, test data will be scaled as well")
    x <- (x - x_range[1]) / (x_range[2] - x_range[1])
    scaled <- TRUE
  }
  
  stopifnot(is.numeric(momentum))
  stopifnot(momentum >= 0 & momentum <=1)
  if(momentum>0){warning('Momentum > 0 not yet implemented.  Ignoring momentum')}
  
  stopifnot(is.numeric(dropout_pct))
  stopifnot(dropout_pct >= 0 & dropout_pct <1)
  if(dropout){warning('Dropout not yet implemented')}
  
  if(is.null(activation_function)){
    activation_function <- function(x){1.0 / (1 + exp(-x))}
  }
  
  # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
  # a Gaussian distribution with mean 0 and standard deviation 0.1.
  #momentum_speed <- sparseMatrix(1, 1, x=0, dims=c(p, num_hidden))
  weights = matrix(rnorm(num_hidden*ncol(x), mean=initial_weights_mean, sd=initial_weights_sd), nrow=ncol(x), ncol=num_hidden)
  # Insert weights for the bias units into the first row and first column.
  weights = cbind(0, weights)
  weights = rbind(0, weights)
  
  # Insert bias units of 1 into the first column.
  x <- cBind(Bias_Unit=1, x)
  dimnames(weights) = list(colnames(x), c('Bias_Unit', paste('Hidden', 1:num_hidden, sep='_')))
  
  #Fit the model
  error_stream <- runif(max_epochs)
  for (epoch in 1:max_epochs){
    
    #Sample mini-batch
    if(use_mini_batches){
      train_rows = sample(1:nrow(x), batch_size, replace=TRUE)
      x_sample = x[train_rows,]
    } else {
      x_sample = x
    }
    
    # Clamp to the data and sample from the hidden units.
    # (This is the "positive CD phase", aka the reality phase.)
    pos_hidden_activations = gpuMatMult(x_sample, weights)
    if(dropout){
      pos_hidden_activations_dropped = pos_hidden_activations
      pos_hidden_activations_dropped@x[runif(length(pos_hidden_activations_dropped@x)) < dropout_pct] = 0
      pos_hidden_activations_dropped[,1] <- pos_hidden_activations[,1]
      pos_hidden_activations <- pos_hidden_activations_dropped
    }
    pos_hidden_probs = activation_function(pos_hidden_activations)
    pos_hidden_states = pos_hidden_probs > matrix(runif(nrow(x_sample)*(num_hidden+1)), nrow=nrow(x_sample), ncol=(num_hidden+1))
    
    # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
    # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
    # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
    pos_associations = gpuCrossprod(x_sample, pos_hidden_probs)
    
    # Reconstruct the visible units and sample again from the hidden units.
    # (This is the "negative CD phase", aka the daydreaming phase.)
    neg_visible_activations = gpuTcrossprod(pos_hidden_states*1, weights)
    neg_visible_probs = activation_function(neg_visible_activations)
    neg_visible_probs[,1] = 1 # Fix the bias unit.
    neg_hidden_activations = gpuMatMult(neg_visible_probs, weights)
    neg_hidden_probs = activation_function(neg_hidden_activations)
    
    # Note, again, that we're using the activation *probabilities* when computing associations, not the states
    # themselves.
    neg_associations = gpuCrossprod(neg_visible_probs, neg_hidden_probs)
    
    # Update weights
    weights = weights + learning_rate * ((pos_associations - neg_associations) / nrow(x_sample))
    
    #Print output
    error = sum((x_sample - neg_visible_probs) ^ 2)
    error_stream[[epoch]] <- error
    if(verbose){
      print(sprintf("Epoch %s: error is %s", epoch, error))
    }
  }
  
  #Return output
  if(retx){
    output_x <- tryCatch(gpuMatMult(x, weights), error=function(e) x %*% weights) #Maybe make this sparse if the gpu fails?
  } else {
    output_x <- NULL
  }
  out <- list(rotation=weights, activation_function=activation_function, x=output_x, error=error_stream, max_epochs=max_epochs, x_range=x_range, scaled=scaled)
  class(out) <- c('rbm_gpu', 'rbm')
  return(out)
}

#' Predict from a Restricted Boltzmann Machine
#'
#' This function takes an RBM and a matrix of new data, and predicts for the new data with the RBM.
#' @param object a RBM object
#' @param newdata a sparse matrix of new data
#' @param type a character vector specifying whether to return the hidden unit activations, hidden unit probs, or hidden unit states.  Activations or probabilities are typically the most useful if you wish to use the RBM features as input to another predictive model (or another RBM!).  Note that the hidden states are stochastic, and may be different each time you run the predict function, unless you set random.seed() before making predictions.  Activations and states are non-stochastic, and will be the same each time you run predict.
#' @param omit_bias Don't return the bias column in the prediciton matrix.
#' @param ... not used
#' @import methods
#' @importFrom Matrix Matrix cBind drop0
#' @importMethodsFrom Matrix %*% crossprod tcrossprod
#' @export
#' @return a sparse matrix
predict.rbm_gpu <- function (object, newdata, type='probs', omit_bias=TRUE, ...) {
  stopifnot(require('gputools'))
  if (missing(newdata)) {
    if (!is.null(object$x)) {
      hidden_activations <- object$x
      rows <- nrow(object$x)
    }
    else stop("no scores are available: refit with 'retx=TRUE'")
  } else {
    #Checks
    stopifnot(length(dim(newdata)) == 2)
    stopifnot(type %in% c('activations', 'probs', 'states'))
    if(any('data.frame' %in% class(newdata))){
      if(any(!sapply(newdata, is.numeric))){
        stop('x must be all finite, numeric data.  rbm does not handle characters, factors, dates, etc.')
      }
      x = as.matrix(newdata)
    } else if (any('matrix' %in% class(newdata))){
    } else if(length(attr(class(newdata), 'package')) != 1){
      stop('Unsupported class for rmb: ', paste(class(newdata), collapse=', '))
    } else if(attr(class(newdata), 'package') != 'Matrix'){
      stop('Unsupported class for rmb: ', paste(class(newdata), collapse=', '))
    }
    
    #Scale if scaled during training
    if (!is.null(object$scaled)) {
      if(object$scaled){
        newdata <- (newdata - object$x_range[1]) / (object$x_range[2] - object$x_range[1])
      }
      if(min(newdata) < 0 | max(newdata) > 1){
        stop('newdata outside of the scale of the model training data.  Could not re-scale data to be 0-1')
      }
    }
    
    # Insert bias units of 1 into the first column.
    newdata <- cbind(Bias_Unit=rep(1, nrow(newdata)), newdata)
    
    nm <- rownames(object$rotation)
    if (!is.null(nm)) {
      if (!all(nm %in% colnames(newdata)))
        stop("'newdata' does not have named columns matching one or more of the original columns")
      newdata <- newdata[, nm, drop = FALSE]
    }
    else {
      if (NCOL(newdata) != NROW(object$rotation))
        stop("'newdata' does not have the correct number of columns")
    }
    hidden_activations <- tryCatch(gpuMatMult(newdata, object$rotation), error=function(e) newdata %*% object$rotation)
    rows <- nrow(newdata)
  }
  
  if(omit_bias){
    if(type=='activations'){return(hidden_activations[,-1,drop=FALSE])}
    hidden_probs <- object$activation_function(hidden_activations)
    if(type=='probs'){return(hidden_probs[,-1,drop=FALSE])}
    hidden_states <- hidden_probs > matrix(runif(rows*ncol(object$rotation)), nrow=rows, ncol=ncol(object$rotation))
    return(hidden_states[,-1,drop=FALSE])
  } else{
    if(type=='activations'){return(hidden_activations)}
    hidden_probs <- object$activation_function(hidden_activations)
    if(type=='probs'){return(hidden_probs)}
    hidden_states <- hidden_probs > matrix(runif(rows*ncol(object$rotation)), nrow=rows, ncol=ncol(object$rotation))
    return(hidden_states)
  }
}


stacked_rbm <- function (x, layers = c(30, 100, 30), learning_rate=0.1, verbose_stack=TRUE, use_gpu=FALSE, ...) {
  if(use_gpu){
    if(require('gputools')){
      rbm <- rbm_gpu
    } else {
      warning('The gputools package is require to train RBMs on the gpu.  RBMs will be trained on the cpu instead.')
    }
  }
  
  #Checks
  stopifnot(length(dim(x)) == 2)
  if(length(learning_rate)==1){
    learning_rate <- rep(learning_rate, length(layers))
  }
  stopifnot(length(layers) == length(learning_rate))
  
  if(any('data.frame' %in% class(x))){
    if(any(!sapply(x, is.finite))){
      stop('x must be all finite.  rbm does not handle NAs, NaNs, Infs or -Infs')
    }
    if(any(!sapply(x, is.numeric))){
      stop('x must be all finite, numeric data.  rbm does not handle characters, factors, dates, etc.')
    }
    x = Matrix(as.matrix(x), sparse=TRUE)
  } else if (any('matrix' %in% class(x))){
    x = Matrix(x, sparse=TRUE)
  } else if(length(attr(class(x), 'package')) != 1){
    stop('Unsupported class for rmb: ', paste(class(x), collapse=', '))
  } else if(attr(class(x), 'package') != 'Matrix'){
    stop('Unsupported class for rmb: ', paste(class(x), collapse=', '))
  }
  
  x_range <- range(x)
  scaled <- FALSE
  if (x_range[1] <0 | x_range[2] > 1){
    warning("x is out of bounds, automatically scaling to 0-1, test data will be scaled as well")
    x <- (x - x_range[1]) / (x_range[2] - x_range[1])
    scaled <- TRUE
  }
  
  if(length(layers) < 2){
    stop('Please use the rbm function to fit a single rbm')
  }
  
  #Fit first RBM
  if(verbose_stack){print('Fitting RBM 1')}
  rbm_list <- as.list(layers)
  if(verbose_stack){message(paste('Input dims:', paste(dim(x), collapse=', ')))}
  if(verbose_stack){message(paste('Input class:', paste(class(x), collapse=', ')))}
  rbm_list[[1]] <- rbm(x, num_hidden=layers[[1]], learning_rate=learning_rate[[1]], retx=TRUE, ...)
  
  #Fit the rest of the RBMs
  #Do we train on the returned x dataset?
  #Or should we just train on the rotation matrix (would be MUCH quicker for large datasets)
  for(i in 2:length(rbm_list)){
    if(verbose_stack){message(paste('Fitting RBM', i))}
    new_train <- predict(rbm_list[[i-1]], type='probs', omit_bias=TRUE)
    if(verbose_stack){message(paste('Input dims:', paste(dim(new_train), collapse=', ')))}
    if(verbose_stack){message(paste('Input class:', paste(class(new_train), collapse=', ')))}
    rbm_list[[i]] <- rbm(new_train, num_hidden=layers[[i]], learning_rate=learning_rate[[i]], retx=TRUE, ...)
  }
  
  #Return result
  out <- list(rbm_list=rbm_list, layers=layers, activation_function=rbm_list[[1]]$activation_function, x_range=x_range, scaled=scaled)
  class(out) <- 'stacked_rbm'
  return(out)
}

#' Predict from a Stacked Restricted Boltzmann Machine
#'
#' This function takes a stacked RBM and a matrix of new data, and predicts for the new data with the RBM.
#'
#' @param object a RBM object
#' @param newdata a sparse matrix of new data
#' @param type a character vector specifying whether to return the hidden unit activations, hidden unit probs, or hidden unit states.  Activations or probabilities are typically the most useful if you wish to use the RBM features as input to another predictive model (or another RBM!).  Note that the hidden states are stochastic, and may be different each time you run the predict function, unless you set random.seed() before making predictions.  Activations and states are non-stochastic, and will be the same each time you run predict.
#' @param omit_bias Don't return the bias column in the prediciton matrix.
#' @param ... not used
#' @export
#' @return a sparse matrix
#' @import methods
#' @importFrom Matrix Matrix cBind drop0
#' @importMethodsFrom Matrix %*% crossprod tcrossprod
predict.stacked_rbm <- function (object, newdata, type='probs', omit_bias=TRUE, ...) {
  
  #If no new data, just return predictions from the final rbm in the stack
  if (missing(newdata)) {
    return(predict(object$rbm_list[[length(object$rbm_list)]], type=type, omit_bias=omit_bias))
  } else {
    if(! type %in% c('probs', 'states')){
      stop('Currently we can only return hidden probabilities or states from a stacked rbm.  Activations are not yet supported')
    }
    hidden_probs <- predict(object$rbm_list[[1]], newdata=newdata, type='probs', omit_bias=TRUE)
    for(i in 2:length(object$rbm_list)){
      omit_bias_in_loop <- TRUE
      if(i==length(object$rbm_list)){
        omit_bias_in_loop <- omit_bias #For the last RBM, use the user_specified omit_bias
      }
      hidden_probs <- predict(object$rbm_list[[i]], newdata=hidden_probs, type='probs', omit_bias=omit_bias_in_loop)
    }
  }
  
  #Scale if scaled during training
  if (!is.null(object$scaled)) {
    if(object$scaled){
      newdata <- (newdata - object$x_range[1]) / (object$x_range[2] - object$x_range[1])
    }
    if(min(newdata) < 0 | max(newdata) > 1){
      stop('newdata outside of the scale of the model training data.  Could not re-scale data to be 0-1')
    }
  }
  
  rows <- nrow(hidden_probs)
  cols <- ncol(hidden_probs)
  
  if(omit_bias){
    if(type=='probs'){
      return(hidden_probs)
    }
    else if(type == 'states'){
      hidden_states <- hidden_probs > Matrix(runif(rows*cols), nrow=rows, ncol=cols)
      return(hidden_states)
    }
  } else{
    if(type=='probs'){
      return(hidden_probs)
    }
    else if(type == 'states'){
      hidden_states <- hidden_probs > Matrix(runif(rows*cols), nrow=rows, ncol=cols)
      return(hidden_states)
    }
  }
}

#' Combine weights from a Stacked Restricted Boltzmann Machine
#'
#' This function takes a stacked RBM and returns the combined weight matrix
#'
#' @param x a RBM object
#' @param layer which RBM to return weights for (usually the final RBM, which will combine all 3 RBMs into a single weight matrix)
#' @param ... not used
#' @export
#' @return a sparse matrix
#' @importFrom Matrix Matrix cBind drop
#' @importMethodsFrom Matrix %*% crossprod tcrossprod
combine_weights.stacked_rbm <- function(x, layer=length(x$rbm_list)){
  x$rbm_list[[1]]$rotation %*% x$rbm_list[[2]]$rotation %*% x$rbm_list[[3]]$rotation
}