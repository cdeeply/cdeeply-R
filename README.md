# cdeeply-R
R interface to C Deeply's neural network generator

Put cdeeply_neural_network.r into a reachable directory, then:

1) Call `myNN <- CDNN_tabular_regressor(...)` or `myNN <- CDNN_tabular_encoder(...)` to train a neural network in supervised or unsupervised mode.  *This step requires an internet connection*! as the training is done server-side.
2) Call `output <- myNN(input)` as many times as you want to process new data samples -- one sample per function call.

**Function definitions:**

`myNN <- CDNN_tabular_regressor(trainingSamples, sampleTableTranspose, outputRowOrColumnList, importances=c(),`  
`        maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX", maxLayerSkips="NO_MAX",`  
`        ifNNhasBias=TRUE, ifAllowingInputOutputConnections=TRUE)`

Generates a x->y prediction network using *supervised* training on `trainingSamples`.
* `trainingSamples` is a matrix having dimensions `numFeatures` and `numSamples`, containing *both* inputs and target outputs.
  * Set `sampleTableTranspose` to `"FEATURE_SAMPLE_ARRAY"` for `trainingSamples[[feature, sample]]` array ordering, or `"SAMPLE_FEATURE_ARRAY"` for `trainingSamples[[sample, feature]]` array ordering.
  * The rows/columns in `trainingSamples` corresponding to the target outputs are specified by `outputRowOrColumnList`.
* The optional `importances` argument weights the cost function of the target outputs.  Pass as a matrix having dimensions `numTargetOutputs` and `numSamples` (ordered according to `sampleTableTranspose`).
* Optional integer parameters `maxWeights`, `maxHiddenNeurons` and `maxLayers` limit the size of the neural network, and `maxLayerSkips` limits the depth of layer-to-layer connections.
* Set `ifNNhasBias` to `"NO_BIAS"` if you don't want to allow a bias (i.e. constant) term in each neuron's input.
* Set `ifAllowingInputOutputConnections` to `"NO_IO_CONNECTIONS"` to forbid the input layer from feeding directly into the output layer.  (Outliers in new input data might cause wild outputs).
* All optional arguments (having default values) must be passed by name to avoid conflict with 'hidden' arguments that are ignored.

`myNN <- CDNN_tabular_encoder(trainingSamples, sampleTableTranspose, importances=c(),`  
`        ifDoEncoder=TRUE, ifDoDecoder=TRUE, numEncodingFeatures=1, numVariationalFeatures=0, variationalDist="NORMAL_DIST",`  
`        maxWeights="NO_MAX", maxHiddenNeurons="NO_MAX", maxLayers="NO_MAX", maxLayerSkips="NO_MAX", ifNNhasBias=TRUE)`

Generates an autoencoder (or an encoder or decoder) using *unsupervised* training on `trainingSamples`.
* `trainingSamples` is a matrix having dimensions `numFeatures` and `numSamples`, in the order determined by `sampleTableTranspose`.
* `sampleTableTranspose` and `importances` are set the same way as for `tabular_regressor(...)`.
* The size of the encoding is determined by `numEncodingFeatures`.
  * So-called variational features are extra randomly-distributed inputs used by the decoder, analogous to the extra degrees of freedom a variational autoencoder generates.
  * `variationalDist` is set to `"UNIFORM_DIST"` if the variational inputs are uniformly-(0, 1)-distributed, or `"NORMAL_DIST"` if they are normally distributed (zero mean, unit variance).
* Set `ifDoEncoder` to either `"DO_ENCODER"` or `"NO_ENCODER"`, the latter being for a decoder-only network.
* Set `ifDoDecoder` to either `"DO_DECODER"` or `"NO_DECODER"`, the latter being for an encoder-only network.
* The last 5 parameters are set the same way as for `tabular_regressor(...)`.
* Again, pass all optional arguments (having default values) explicitly by name.

`sampleOutput = runSample(sampleInput)`

Runs the neural network on a *single* sample, and returns the network output.
* If it's an autoencoder (encoder+decoder), `length(sampleInput)` and `length(sampleOutput)` equal the number of features in the training sample space.  If it's just an encoder, `length(oneSampleOutput)` equals `numEncodingFeatures`; if decoder only, `length(oneSampleInput)` must equal `numEncodingFeatures`.
* If it's a decoder or autoencoder network with variational features, sample the variational features from the appropriate distribution and append them to `oneSampleInput`.
* The return value is simply a copy of the last layer of the network `myNN.y[myNN.numLayers]`.
* Calling `runSample()` without argumentsj returns the training output *as calculated by the server*.  This is mainly a check that the data went through the pipes OK.
