source("cdeeply_neural_network.r")

numFeatures <- 10
numSamples <- 100
NNtypes <- c("autoencoder with 1 latent feature", "regressor")


    # generate a training matrix that traces out some noisy curve in Nf-dimensional space (noise ~ 0.1)

dependentVar <- matrix( runif(n = numSamples+1, min = 0, max = 1), nrow = numSamples+1)
trainTestMat <- matrix( rnorm(n = (numSamples+1)*numFeatures, mean = 0, sd = 0.1), nrow = numSamples+1)
for (cf in 1:numFeatures)  {
    featurePhase <- 2*pi*runif(1)
    featureCurvature <- 2*pi*runif(1)
    trainTestMat[, cf] <- trainTestMat[, cf] + sin(featureCurvature*dependentVar + featurePhase)
}


for (c2 in 1:2)  {
    
    
        # train a neural network from our matrix
    
    cat("Generating ", NNtypes[[c2]], "\n")
    if (c2 == 1)  {
        NN <- CDNN_tabular_encoder(trainTestMat[1:numSamples,], "SAMPLE_FEATURE_ARRAY", numEncodingFeatures=1, doEncoder=TRUE, doDecoder=TRUE)
        firstSampleOutputs <- NN(trainTestMat[1,])
        testSampleOutputs <- NN(trainTestMat[numSamples+1,])
    }
    else  {
        NN <- CDNN_tabular_regressor(trainTestMat[1:numSamples,], "SAMPLE_FEATURE_ARRAY", c(numFeatures))
        firstSampleOutputs <- NN(trainTestMat[1, 1:(numFeatures-1)])
        testSampleOutputs <- NN(trainTestMat[numSamples+1, 1:(numFeatures-1)])
    }
    outputsComputedByServer <- NN()[[1]]
    
    
        # run the network on the first training sample
    
    if (abs(firstSampleOutputs[[1]]-outputsComputedByServer[[1]]) > .0001)  {
        stop("  ** Network problem?  Sample 1 output was calculated as ", firstSampleOutputs[[1]], " locally vs ", outputsComputedByServer[[1]], " by the server")
    }
    
    if (c2 == 1)  {
        targetValue <- trainTestMat[[numSamples+1, 1]]
        targetDescription <- "  Reconstructed test sample, feature 1"    }
    else  {
        targetValue <- trainTestMat[[numSamples+1, numFeatures]]
        targetDescription <- "  Test sample output"    }
    cat(targetDescription, " was ", testSampleOutputs[[1]], "; target value was ", targetValue, "\n")
}
        
        
