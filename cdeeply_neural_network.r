# usage:
# 
# 1) Generate a neural network, using either of:
# 
# myNN <- CDNN_tabular_regressor( trainingSamples, indexOrder, outputIndices [, importances=c(),
#               maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX", maxWeightDepth="NO_MAX", maxActivationRate=1.,
#               maxWeightsHardLimit=TRUE, maxHiddenNeuronsHardLimit=TRUE, maxActivationsHardLimit=TRUE, allowedAFs[5]=c(TRUE,TRUE,TRUE,TRUE,TRUE), 
#               ifQuantizeW=FALSE, wQuantBits=0, wQuantZeroInt=0, wQuantRange=1.,
#               ifQuantizeY=FALSE, yQuantBits=0, yQuantZeroInt=0, yQuantRange=1.,
#               sparseWeights=FALSE, allowNegativeWeights=TRUE, hasBias=TRUE, allowIOconnections=TRUE ] )
# 
# myNN <- CDNN_tabular_encoder( trainingSamples, indexOrder [, importances=c(),
#               doEncoder=TRUE, doDecoder=TRUE, numEncodingFeatures=1, numVariationalFeatures=0, variationalDistribution="NORMAL_DIST",
#               maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX", maxWeightDepth="NO_MAX", maxActivationRate=1.,
#               maxWeightsHardLimit=TRUE, maxHiddenNeuronsHardLimit=TRUE, maxActivationsHardLimit=TRUE, allowedAFs[5]=c(TRUE,TRUE,TRUE,TRUE,TRUE), 
#               ifQuantizeW=FALSE, wQuantBits=0, wQuantZeroInt=0, wQuantRange=1.,
#               ifQuantizeY=FALSE, yQuantBits=0, yQuantZeroInt=0, yQuantRange=1.,
#               sparseWeights=FALSE, allowNegativeWeights=TRUE, hasBias=TRUE ] )
# 
# * indexOrder="SAMPLE_FEATURE_ARRAY" for trainingSamples[sampleNo][featureNo] indexing,
#       or "FEATURE_SAMPLE_ARRAY" for trainingSamples[featureNo][sampleNo] indexing.
# * For supervised x->y regression, the sample table contains BOTH 'x' and 'y', the latter specified by outputIndices[].
# * The importances table, if not empty, has dimensions numOutputFeatures and numSamples (ordered by indexOrder),
#       and weights the training cost function:  C = sum(Imp*dy^2).
# * Weight/neuron/etc limits are either positive integers or "NO_MAX".
# * variationalDistribution is either "UNIFORM_DIST" ([0, 1]) or "NORMAL_DIST" (mean=0, variance=1).
# The optional arguments inside [...] should be named since both functions have 'hidden' arguments
# 
# 
# 2) Run the network on a (single) new sample
# 
# oneSampleOutput <- myNN(oneSampleInput)
# 
# where oneSampleInput is a list of length numInputFeatures, and oneSampleOutput is a list of length numOutputFeatures.
# * If it's an autoencoder (encoder+decoder), length(oneSampleInput) and length(oneSampleOutput) equal the size of the training sample space.
#       If it's just an encoder, length(oneSampleOutput) equals numEncodingFeatures; if decoder only, length(oneSampleInput) must equal numEncodingFeatures.
# * If it's a decoder or autoencoder network having numVariationalFeatures > 0, then oneSampleInput = c(sampleInput, variationalLayerInput)
#       where variationalLayerInput[] is a list of length numVariationalFeatures containing random numbers drawn from variationalDistribution.


#install.packages("curl")
library(curl)


CDNN <- function(ifSupervised)
{
    isSupervised <- ifSupervised
    
    CDNN_call = function(trainingSamples, indexOrder, outputIndices=c(), importances=c(),
            doEncoder=TRUE, doDecoder=TRUE, numEncodingFeatures=1, numVariationalFeatures=0, variationalDistribution="NORMAL_DIST",
            maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX", maxWeightDepth="NO_MAX", maxActivationRate=1.,
            maxWeightsHardLimit=TRUE, maxHiddenNeuronsHardLimit=TRUE, maxActivationsHardLimit=TRUE, allowedAFs=c(TRUE,TRUE,TRUE,TRUE,TRUE), 
            ifQuantizeW=FALSE, wQuantBits=0, wQuantZeroInt=0, wQuantRange=1.,
            ifQuantizeY=FALSE, yQuantBits=0, yQuantZeroInt=0, yQuantRange=1.,
            sparseWeights=FALSE, allowNegativeWeights=TRUE, hasBias=TRUE, allowIOconnections=TRUE)
    {
        
        numLayers <- encoderLayer <- variationalLayer <- 0
        layerSize <- layerAFs <- layerInputs <- n0 <- nf <- weights <- c()
        y <- c()
        sparseWeights <- FALSE
        
        fs <- c(
            function(x) { return(x) },
            function(x) { return(max(0., min(1., ceiling(x)))) },
            function(x) { return(max(0., x)) },
            function(x) { return(max(0., min(1., x))) },
            function(x) { return(1. / (1. + exp(-x))) },
            function(x) { return(tanh(x)) }
        )
        
       
        
        
        ifChecked = function(checkedBool)
        {
            if (checkedBool)  return("on")
            else  return("")
        }
        
        maxString = function(maxVar)
        {
            if (maxVar == "NO_MAX")  return("")
            else  return(as.character(maxVar))
        }
        
        
        rowcolStrings <- c("rows", "columns")
        
        data2table = function(data, numIOs)
        {
            if (indexOrder == "FEATURE_SAMPLE_ARRAY")  {
                dim1 <- numIOs
                dim2 <- numSamples
                rowcol <- "rows"     }
            else if (indexOrder == "SAMPLE_FEATURE_ARRAY")  {
                dim1 <- numSamples
                dim2 <- numIOs
                rowcol <- "columns"   }
            
            rowElStrings <- vector(mode="list", dim2)
            tableRowStrings <- vector(mode="list", dim1)
            
            if (dim1 > 0)  {
            for (i1 in 1:dim1)  {
                tableRowStrings[[i1]] <- paste(data[i1, 1:dim2], collapse=",")
            }}
            
            tableStr <- paste(tableRowStrings, collapse="\n")
            
            return(c(tableStr, rowcol))
        }
        
        
        buildCDNN = function(NNdata)
        {
            
            loadNumArray = function(numericString)
            {
                return(as.numeric(unlist(strsplit(numericString, ",", fixed=TRUE))))
            }
            
            
            firstChar <- as.numeric(utf8ToInt(substr(NNdata, 1, 1)))
            if (firstChar < as.numeric(utf8ToInt('0')) || firstChar > as.numeric(utf8ToInt('9')))  {
                stop(NNdata)
            }
            
            NNdataRows <- unlist(strsplit(NNdata, ";", fixed=TRUE))
            
            NNheader <- unlist(strsplit(NNdataRows[[1]], ",", fixed=TRUE))
            numLayers <<- as.numeric(NNheader[[1]])
            encoderLayer <<- as.numeric(NNheader[[2]])
            variationalLayer <<- as.numeric(NNheader[[3]])
            
            layerSize <<- loadNumArray(NNdataRows[[2]])
            layerAFs <<- loadNumArray(NNdataRows[[3]])+1
            numLayerInputs <- loadNumArray(NNdataRows[[4]])
            
            allLayerInputs <- loadNumArray(NNdataRows[[5]])+1
            layerInputs <<- vector(mode="list", numLayers)
            if (sparseWeights)  {
                allWsizes <- loadNumArray(NNdataRows[[6]])
                allL0s <- loadNumArray(NNdataRows[[7]])
                allNfs <- loadNumArray(NNdataRows[[8]])
                wSize <<- vector(mode="list", allWsizes)
                swOffset = 3;
            }
            else  swOffset = 0;
            
            idx <- 0
            for (l in 1:numLayers)  {
                layerInputs[[l]] <<- c(allLayerInputs[idx+(1:numLayerInputs[[l]])])
                if (sparseWeights)  wSize[[l]] <<- c(allWsizes[idx+(1:numLayerInputs[[l]])])
                idx <- idx + numLayerInputs[[l]]
            }
            
            allWs <- loadNumArray(NNdataRows[[6+swOffset]])
            if (sparseWeights)  {
                n0 <<- vector(mode="list", numLayers)
                nf <<- vector(mode="list", numLayers)
            }
            weights <<- vector(mode="list", numLayers)
            idx <- 0
            for (l in 1:numLayers)  {
                if (sparseWeights)  {
                    n0[[l]] <<- vector(mode="list", numLayerInputs[[l]])
                    nf[[l]] <<- vector(mode="list", numLayerInputs[[l]])
                }
                weights[[l]] <<- vector(mode="list", numLayerInputs[[l]])
                if (numLayerInputs[[l]] > 0)  {
                for (li in 1:numLayerInputs[[l]])  {
                    l0 <- layerInputs[[l]][[li]]
                    numWeights <- layerSize[[l0]]*layerSize[[l]]
                    if (sparseWeights)  {
                        n0[[l]][[li]] <<- allN0s[idx+(1:numWeights)]
                        nf[[l]][[li]] <<- allNfs[idx+(1:numWeights)]
                        weights[[l]][[li]] <<- allWs[idx+(1:numWeights)]
                    }
                    else  weights[[l]][[li]] <<- t(matrix(unlist(allWs[idx+(1:numWeights)]), nrow=layerSize[[l0]], ncol=layerSize[[l]]))
                    idx <- idx + numWeights
            }   }}
            
            outputsComputedByServer <- matrix(unlist(loadNumArray(NNdataRows[[7+swOffset]])), nrow=numSamples, ncol=numOutputs)
            if (indexOrder == "FEATURE_SAMPLE_ARRAY")  {
                outputsComputedByServer <- t(outputsComputedByServer)        }
            
            y <<- vector(mode="list", numLayers)
            for (l in 1:numLayers)  {
                y[[l]] <<- vector(mode="double", layerSize[[l]])
            }
            
            return(outputsComputedByServer)
        }
        
        
        runSample = function(oneSampleInput=c())
        {
            if (length(oneSampleInput) == 0)  return(outputsComputedByServer)
            
            if (variationalLayer <= 0)  NNinput <- oneSampleInput
            else  {
                NNinput <- oneSampleInput[[1]]
                NNvariationalInput <- oneSampleInput[[2]]
            }
            
            y[[1]][] <<- 1
            y[[2]] <<- NNinput
            if (variationalLayer > 0)
                y[[variationalLayer]] <<- NNvariationalInput
            
            for (l in 3:numLayers)  {
            if (l != variationalLayer)  {
                y[[l]][] <<- 0
                if (length(layerInputs[[l]]) > 0)  {
                for (li in 1:length(layerInputs[[l]]))  {
                    l0 <- layerInputs[[l]][[li]]
                    if (sparseWeights)  {
                        for (w in 1:length(weights[[l]][[li]]))  {
                            y[[l]][[1,nf[[l]][[li]][[w]]]] <<- y[[l]][[1,nf[[l]][[li]][[w]]]] + weights[[l]][[li]][[w]] * y[[l0]][[1,n0[[l]][[li]][[w]]]]
                    }   }
                    else  {
                        y[[l]] <<- y[[l]] + weights[[l]][[li]] %*% y[[l0]]
                }}  }
                for (n in 1:layerSize[[l]])  y[[l]][[n]] <<- fs[[layerAFs[[l]]]](y[[l]][[n]])
            }}
            
            return(y[[numLayers]])
        }
        
        
        sampleDims <- dim(trainingSamples)
        if (indexOrder == "FEATURE_SAMPLE_ARRAY")  {
            numIOs <- sampleDims[[1]]
            numSamples <- sampleDims[[2]]     }
        else if (indexOrder == "SAMPLE_FEATURE_ARRAY")  {
            numIOs <- sampleDims[[2]]
            numSamples <- sampleDims[[1]]     }
        else  {
            stop("transpose must be either \"FEATURE_SAMPLE_ARRAY\" or \"SAMPLE_FEATURE_ARRAY\"")
        }
        
        importancesString <- ""
        
        lcHandle <- new_handle()
        handle_setopt(lcHandle, post=TRUE)
        
        if (isSupervised)  {
            numOutputs <- length(outputIndices)
            numInputs <- numIOs-numOutputs
            
            d2t <- data2table(trainingSamples, numIOs)
            sampleString = d2t[[1]]
            rowcolString = d2t[[2]]
            if (length(importances) > 0)  importancesString <- data2table(importances, numInputs)
            
            orcStrings <- vector(mode="list", length(outputIndices))
            if (length(outputIndices) > 0)  {
            for (rc in 1:length(outputIndices))  {
                orcStrings[[rc]] <- as.character(outputIndices[[rc]])
            }}
            outputRowsColumnsString <- paste(orcStrings, collapse=",")
            
            handle_setform(lcHandle,
                samples = sampleString,
                importances = importancesString,
                rowscols = rowcolString,
                rowcolRange = outputRowsColumnsString,
                maxWeights = maxString(maxWeights),
                maxNeurons = maxString(maxHiddenNeurons),
                maxLayers = maxString(maxLayers),
                maxWeightDepth = maxString(maxWeightDepth),
                maxActivationRate = as.character(maxActivationRate),
                maxWeightsHardLimit = ifChecked(maxWeightsHardLimit),
                maxNeuronsHardLimit = ifChecked(maxHiddenNeuronsHardLimit),
                maxActivationsHardLimit = ifChecked(maxActivationsHardLimit),
                step = ifChecked(allowedAFs[[1]]),
                ReLU = ifChecked(allowedAFs[[2]]),
                ReLU1 = ifChecked(allowedAFs[[3]]),
                sigmoid = ifChecked(allowedAFs[[4]]),
                tanh = ifChecked(allowedAFs[[5]]),
                quantizeWeights = ifChecked(ifQuantizeW),
                wQuantBits = as.character(wQuantBits),
                wQuantZero = as.character(wQuantZeroInt),
                wQuantRange = as.character(wQuantRange),
                quantizeActivations = ifChecked(ifQuantizeY),
                yQuantBits = as.character(yQuantBits),
                yQuantZero = as.character(yQuantZeroInt),
                yQuantRange = as.character(yQuantRange),
                sparseWeights = ifChecked(sparseWeights),
                allowNegativeWeights = ifChecked(allowNegativeWeights),
                hasBias = ifChecked(hasBias),
                allowIO = ifChecked(allowIOconnections),
                submitStatus = "Submit",
                NNtype = "regressor",
                formSource = "R_API"  )
        }
        
        else  {
            numInputs <- numIOs
            if (doDecoder)  numOutputs <- numInputs
            else  numOutputs <- numEncodingFeatures
            
            d2t <- data2table(trainingSamples, numInputs)
            sampleString = d2t[[1]]
            rowcolString = d2t[[2]]
            if (length(importances) > 0)  importancesString <- data2table(importances, numInputs)
            
            if ((variationalDistribution == "UNIFORM_DIST"))  variationalDistStr <- "uniform"
            else if ((variationalDistribution == "NORMAL_DIST"))  variationalDistStr <- "normal"
            else  stop("Variational distribution must be either \"UNIFORM_DIST\" or \"NORMAL_DIST\"")
            
            handle_setform(lcHandle,
                samples = sampleString,
                importances = importancesString,
                rowscols = rowcolString,
                numFeatures = as.character(numEncodingFeatures),
                doEncoder = ifChecked(doEncoder),
                doDecoder = ifChecked(doDecoder),
                numVPs = as.character(numVariationalFeatures),
                variationalDist = variationalDistStr,
                maxWeights = maxString(maxWeights),
                maxNeurons = maxString(maxHiddenNeurons),
                maxLayers = maxString(maxLayers),
                maxWeightDepth = maxString(maxWeightDepth),
                maxActivationRate = as.character(maxActivationRate),
                maxWeightsHardLimit = ifChecked(maxWeightsHardLimit),
                maxNeuronsHardLimit = ifChecked(maxHiddenNeuronsHardLimit),
                maxActivationsHardLimit = ifChecked(maxActivationsHardLimit),
                step = ifChecked(allowedAFs[[1]]),
                ReLU = ifChecked(allowedAFs[[2]]),
                ReLU1 = ifChecked(allowedAFs[[3]]),
                sigmoid = ifChecked(allowedAFs[[4]]),
                tanh = ifChecked(allowedAFs[[5]]),
                quantizeWeights = ifChecked(ifQuantizeW),
                wQuantBits = as.character(wQuantBits),
                wQuantZero = as.character(wQuantZeroInt),
                wQuantRange = as.character(wQuantRange),
                quantizeActivations = ifChecked(ifQuantizeY),
                yQuantBits = as.character(yQuantBits),
                yQuantZero = as.character(yQuantZeroInt),
                yQuantRange = as.character(yQuantRange),
                sparseWeights = ifChecked(sparseWeights),
                allowNegativeWeights = ifChecked(allowNegativeWeights),
                hasBias = ifChecked(hasBias),
                submitStatus = "Submit",
                NNtype = "autoencoder",
                formSource = "R_API"  )
        }
        
        response <- curl_fetch_memory("https://cdeeply.com/myNN.php", lcHandle)
        
        outputsComputedByServer <- buildCDNN(rawToChar(response$content))
        
        return(runSample)
    }
    
    return(CDNN_call)
}


CDNN_tabular_regressor <- CDNN(TRUE)
CDNN_tabular_encoder <- CDNN(FALSE)

