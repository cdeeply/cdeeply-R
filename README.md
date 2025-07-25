# cdeeply-R
R interface to C Deeply's neural network generator

Put cdeeply_neural_network.r into a reachable directory, then:

1) Call `myNN <- CDNN_tabular_regressor(...)` or `myNN <- CDNN_tabular_encoder(...)` to train a neural network in supervised or unsupervised mode.  *This step requires an internet connection*! as the training is done server-side.
2) Call `output <- myNN(input)` as many times as you want to process new data samples -- one sample per function call.

**Function definitions:**

`myNN <- CDNN_tabular_regressor(trainingSamples, sampleTableTranspose, outputIndices, importances=c(),`  
`        maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX",`  
`        maxWeightDepth="NO_MAX", maxActivationRate=1.,`  
`        maxWeightsHardLimit=TRUE, maxHiddenNeuronsHardLimit=TRUE, maxActivationsHardLimit=TRUE,`  
`        allowedAFs=c(TRUE,TRUE,TRUE,TRUE,TRUE), `  
`        ifQuantizeWeights=FALSE, wQuantBits=0, wQuantZeroInt=0, wQuantRange=1., `  
`        ifQuantizeActivations=FALSE, yQuantBits=0, yQuantZeroInt=0, yQuantRange=1., `  
`        sparseWeights=FALSE, allowNegativeWeights=TRUE, hasBias=TRUE, allowIOconnections=TRUE)`

Generates a x->y prediction network using *supervised* training on `trainingSamples`.
* `trainingSamples` is a matrix having dimensions `numFeatures` and `numSamples`, containing *both* inputs and target outputs.
  * Set `sampleTableTranspose` to `"FEATURE_SAMPLE_ARRAY"` for `trainingSamples[[feature, sample]]` array ordering, or `"SAMPLE_FEATURE_ARRAY"` for `trainingSamples[[sample, feature]]` array ordering.
  * The rows/columns in `trainingSamples` corresponding to the target outputs are specified by `outputIndices`.
* The optional `importances` argument weights the cost function of the target outputs.  Pass as a matrix having dimensions `numTargetOutputs` and `numSamples` (ordered according to `sampleTableTranspose`).
* Optional integer parameters `maxWeights`, `maxHiddenNeurons` and `maxLayers` limit the size of the neural network, and `maxWeightDepth` limits the depth of layer-to-layer connections.  The corresponding `maxWeightsHardLimit` and `maxHiddenNeuronsHardLimit` parameters should be either `TRUE` or `FALSE`.
* `allowedAFs` is a length-5 Boolean vector corresponding to the allowed activation functions:  (step, ReLU, ReLU1, sigmoid, tanh).  Set each Boolean according to whether the activation function should be considered for a given layer of the network.
* `To quantize weights or neural activations, set `ifQuantizeWeights` or `ifQuantizeActivations` to `TRUE` and give values to the following three fields; otherwise set the respective `ifQuantize` to `FALSE`.
* Set `sparseWeights` to either `TRUE` or `FALSE`, depending on whether we are generating sparse weight matrices.
* The parameter `maxActivationRate` should be set to `1.` unless sparsifying the neural activations, in which case this can be a lower positive value.  Sparse activations do not work with sigmoid or tanh activation functions.  Set `maxActivationsHardLimit` to either `TRUE` or `FALSE`.
* The `allowNegativeWeights` and `ifNNhasBias` parameters are either `TRUE` or `FALSE`.  The bias is a constant input to each neuron.
* Set `allowIOconnections` to `FALSE` to forbid the input layer from feeding directly into the output layer.  (Outliers in new input data might cause wild outputs).
* All optional arguments (having default values) must be passed by name to avoid conflict with 'hidden' arguments that are ignored.

`myNN <- CDNN_tabular_encoder(trainingSamples, sampleTableTranspose, importances=c(),`  
`        doEncoder=TRUE, doDecoder=TRUE, numEncodingFeatures=1, numVariationalFeatures=0, variationalDistribution="NORMAL_DIST",`  
`        maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX",`
`        maxWeightDepth="NO_MAX", maxActivationRate=1.,`  
`        maxWeightsHardLimit=TRUE, maxHiddenNeuronsHardLimit=TRUE, maxActivationsHardLimit=TRUE,'  
`        allowedAFs=[TRUE,TRUE,TRUE,TRUE,TRUE], `  
`        ifQuantizeWeights=FALSE, wQuantBits=0, wQuantZeroInt=0, wQuantRange=1., `  
`        ifQuantizeActivations=FALSE, yQuantBits=0, yQuantZeroInt=0, yQuantRange=1., `  
`        sparseWeights=FALSE, allowNegativeWeights=TRUE, hasBias=TRUE)`

Generates an autoencoder (or an encoder or decoder) using *unsupervised* training on `trainingSamples`.
* `trainingSamples` is a matrix having dimensions `numFeatures` and `numSamples`, in the order determined by `sampleTableTranspose`.
* `sampleTableTranspose` and `importances` are set the same way as for `tabular_regressor(...)`.
* The size of the encoding is determined by `numEncodingFeatures`.
  * So-called variational features are extra randomly-distributed inputs used by the decoder, analogous to the extra degrees of freedom a variational autoencoder generates.
  * `variationalDist` is set to `"UNIFORM_DIST"` if the variational inputs are uniformly-(0, 1)-distributed, or `"NORMAL_DIST"` if they are normally distributed (zero mean, unit variance).
* Set `doEncoder` to `FALSE` for a decoder-only network.
* Set `doDecoder` to `FALSE` for an encoder-only network.
* All other parameters are set the same way as for `tabular_regressor(...)`.
* Again, pass all optional arguments (having default values) explicitly by name.

`sampleOutput = myNN(sampleInput)`

Runs the neural network on a *single* sample, and returns the network output.
* If it's an autoencoder (encoder+decoder), `length(sampleInput)` and `length(sampleOutput)` equal the number of features in the training sample space.  If it's just an encoder, `length(sampleOutput)` equals `numEncodingFeatures`; if decoder only, `length(sampleInput)` must equal `numEncodingFeatures`.
* If it's a decoder or autoencoder network with variational features, sample the variational features from the appropriate distribution and append them to `sampleInput`.
* The return value is simply a copy of the last layer of the network `myNN.y[myNN.numLayers]`.
* Calling `myNN()` without arguments returns the training output *as calculated by the server*.  This is mainly a check that the data went through the pipes OK.
