package kaggle.digit_recognizer;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// https://deeplearning4j.org/tutorials/00-quickstart-for-deeplearning4j
public class RunTutorialDigitRecognizer {

	private static final Logger LOGGER = LoggerFactory.getLogger(RunTutorialDigitRecognizer.class);

	public static void main(String[] args) throws IOException {

		int batchSize = 16; // how many examples to simultaneously train in the network
		EmnistDataSetIterator.Set emnistSet = EmnistDataSetIterator.Set.DIGITS;
		EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
		EmnistDataSetIterator emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);

		int outputNum = EmnistDataSetIterator.numLabels(emnistSet); // total output classes
		int rngSeed = 123; // integer for reproducability of a random number generator
		int numRows = 28; // number of "pixel rows" in an mnist digit
		int numColumns = 28;

		int hiddenLayerSize = 1000;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam())
				.l2(1e-4)
				.list()
				.layer(new DenseLayer.Builder().nIn(numRows * numColumns) // Number of input datapoints.
						.nOut(hiddenLayerSize) // Number of output datapoints.
						.activation(Activation.RELU) // Activation function.
						.weightInit(WeightInit.XAVIER) // Weight initialization.
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(hiddenLayerSize)
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.build())
				// .pretrain(false)
				// .backprop(true)
				.build();

		// create the MLN
		MultiLayerNetwork network = new MultiLayerNetwork(conf);
		network.init();

		// pass a training listener that reports score every 10 iterations
		int eachIterations = 5;
		network.addListeners(new ScoreIterationListener(eachIterations));

		boolean singleEpoch = true;
		// fit a dataset for a single epoch
		if (singleEpoch) {
			network.fit(emnistTrain);
		} else {
			// fit for multiple epochs
			int numEpochs = 2;
			network.fit(emnistTrain, numEpochs);
		}

		{
			// evaluate basic performance
			Evaluation eval = network.evaluate(emnistTest);
			eval.accuracy();
			eval.precision();
			eval.recall();

			// evaluate ROC and calculate the Area Under Curve
			ROCMultiClass roc = network.evaluateROCMultiClass(emnistTest, 0);
			roc.calculateAverageAUC();

			int classIndex = 0;
			roc.calculateAUC(classIndex);

			// optionally, you can print all stats from the evaluations
			LOGGER.info("{}", eval.stats());
			LOGGER.info("{}", roc.stats());
		}
	}
}
