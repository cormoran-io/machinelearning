package kaggle.digit_recognizer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This is based on {@link RunTutorialDigitRecognizer} in order to solve Kaggle
 * https://www.kaggle.com/c/digit-recognizer/
 * 
 * @author Benoit Lacelle
 *
 */
public class RunKaggleDigitRecognizer {

	private static final Logger LOGGER = LoggerFactory.getLogger(RunKaggleDigitRecognizer.class);

	// number of "pixel rows" in an mnist digit
	public static final int COLS = 28;

	public static void main(String[] args) throws IOException {
		int batchSize = 16; // how many examples to simultaneously train in the network
		// EmnistDataSetIterator.Set emnistSet = EmnistDataSetIterator.Set.BALANCED;
		KaggleMnistDataSetIterator emnistTrain = new KaggleMnistDataSetIterator(batchSize, true, 123);
		KaggleMnistDataSetIterator emnistTest = new KaggleMnistDataSetIterator(batchSize, false, 234);

		int outputNum = EmnistDataSetIterator.numLabels(EmnistDataSetIterator.Set.MNIST); // total output classes
		int rngSeed = 123; // integer for reproducability of a random number generator
		int numRows = COLS;
		int numColumns = COLS;

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
		int eachIterations = 100;
		network.addListeners(new ScoreIterationListener(eachIterations));

		boolean singleEpoch = false;
		// fit a dataset for a single epoch
		if (singleEpoch) {
			network.fit(emnistTrain);
		} else {
			// fit for multiple epochs
			int numEpochs = 3;
			network.fit(emnistTrain, numEpochs);
		}

		{
			INDArray output = network.output(emnistTest);

			String outputPath =
					System.getProperty("user.dir") + "/src/main/resources/kaggle/digit_recognizer/submission.csv";
			File outputFile = new File(outputPath);
			outputFile.delete();
			outputFile.createNewFile();
			try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
				writer.write("ImageId,Label");
				writer.newLine();

				for (int imageId = 0; imageId < output.rows(); imageId++) {
					int bestDigit = -1;
					float bestScore = -999;

					INDArray row = output.getRow(imageId);
					for (int digitCandidate = 0; digitCandidate < 10; digitCandidate++) {
						float localScore = row.getFloat(digitCandidate);
						if (localScore > bestScore) {
							bestScore = localScore;
							bestDigit = digitCandidate;
						}
					}

					// Kaggle expect first row to have index == 1
					writer.append(Integer.toString(imageId + 1)).append(",").append(Integer.toString(bestDigit));
					writer.newLine();
				}
			}
		}
	}
}
