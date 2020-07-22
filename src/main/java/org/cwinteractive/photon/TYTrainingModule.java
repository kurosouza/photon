package org.cwinteractive.photon;

import picocli.CommandLine;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

@CommandLine.Command(name = "train", description = "Train model")
public class TYTrainingModule implements Runnable {

    private static final Logger log = LoggerFactory.getLogger(TYTrainingModule.class);

    @Override
    public void run() {
        // number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 1024; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 10; // number of epochs to perform
        double rate = 0.0015; // learning rate

        try {

            // Get the DataSetIterators:
            DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
            DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

            log.info("Build model....");
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed) // include a random seed
                                                                                              // for reproducibility
                    .activation(Activation.RELU).weightInit(WeightInit.XAVIER).updater(new Nadam()).l2(rate * 0.005) // regularize
                                                                                                                     // learning
                                                                                                                     // model
                    .list().layer(new DenseLayer.Builder() // create the first input layer.
                            .nIn(numRows * numColumns).nOut(500).build())
                    .layer(new DenseLayer.Builder() // create the second input layer
                            .nIn(500).nOut(100).build())
                    .layer(new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create hidden layer
                            .activation(Activation.SOFTMAX).nOut(outputNum).build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(5)); // print the score with every iteration

            log.info("Train model....");
            model.fit(mnistTrain, numEpochs);

            log.info("Evaluate model....");
            Evaluation eval = model.evaluate(mnistTest);

            log.info(eval.stats());
            log.info("****************Example finished********************");

        } catch (IOException ioe) {
            log.error("IOException: ", ioe);
        }
    }

}