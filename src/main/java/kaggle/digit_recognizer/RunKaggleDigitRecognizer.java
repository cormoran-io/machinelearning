package kaggle.digit_recognizer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
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

		// Open http://localhost:9000/train/overview
		{
			// Initialize the user interface backend
			UIServer uiServer = UIServer.getInstance();

			// Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in
			// memory.
			// Alternative: new FileStatsStorage(File), for saving
			// and loading later
			StatsStorage statsStorage = new InMemoryStatsStorage();

			// Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
			uiServer.attach(statsStorage);

			// Then add the StatsListener to collect this information from the network, as it trains
			network.setListeners(new StatsListener(statsStorage));
		}

		// pass a training listener that reports score every 10 iterations
		int eachIterations = 100;
		network.addListeners(new ScoreIterationListener(eachIterations));

		boolean singleEpoch = true;
		// fit a dataset for a single epoch
		if (singleEpoch) {
			network.fit(emnistTrain);
		} else {
			// fit for multiple epochs
			int numEpochs = 3;
			network.fit(emnistTrain, numEpochs);
		}

		{
			// TODO We should split the train dataset

			// evaluate basic performance
			Evaluation eval = network.evaluate(emnistTrain);
			eval.accuracy();
			eval.precision();
			eval.recall();

			// evaluate ROC and calculate the Area Under Curve
			ROCMultiClass roc = network.evaluateROCMultiClass(emnistTrain, 0);
			roc.calculateAverageAUC();

			int classIndex = 0;
			roc.calculateAUC(classIndex);

			// optionally, you can print all stats from the evaluations
			LOGGER.info("{}", eval.stats());
			LOGGER.info("{}", roc.stats());
		}

		{

			// TMP: Try checking the result on the trained data
			KaggleMnistDataSetIterator emnistTest = new KaggleMnistDataSetIterator(batchSize,
					KaggleMnistDataFetcher.NUM_EXAMPLES,
					false,
					false,
					false,
					234);

			// Print the data as read by the iterator
			// Useful to spot we are shuffling the test data, and then we are unable to properly answer to Kaggle...
			if (false) {
				INDArray features = emnistTest.next(1).getFeatures();
				System.out.println("---train= " + "?" + "--------");
				for (int i = 0; i < RunKaggleDigitRecognizer.COLS * RunKaggleDigitRecognizer.COLS; i++) {
					if (i % RunKaggleDigitRecognizer.COLS == 0) {
						System.out.println();
					}
					if (features.getFloat(i) < 0.01) {
						System.out.print(' ');
					} else if (features.getFloat(i) < 0.1) {
						System.out.print('.');
					} else if (features.getFloat(i) < 0.5) {
						System.out.print('x');
					} else {
						System.out.print('m');
					}
				}
				System.out.println();
				System.out.println("--------------------");
			}

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

					// We search for the label having maximized the score
					// TODO It seems not to work at all
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

		UIServer.getInstance().stop();
	}
}
