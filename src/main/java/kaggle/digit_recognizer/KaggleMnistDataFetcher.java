package kaggle.digit_recognizer;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.MathUtils;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;

/**
 * Data fetcher for the MNIST dataset, dedicated for Kaggle Dataset. We removed the checksums.
 * 
 * @author Benoit Lacelle
 * @see MnistDataFetcher
 */
public class KaggleMnistDataFetcher extends BaseDataFetcher {
	private static final long serialVersionUID = 7485601686232185806L;

	public static final int NUM_EXAMPLES = 42000;
	public static final int NUM_EXAMPLES_TEST = 28000;

	protected KaggleMnistManager man;
	protected boolean binarize = true;
	protected boolean train;
	protected int[] order;
	protected Random rng;
	protected boolean shuffle;

	protected boolean firstShuffle = true;
	protected final int numExamples;

	/**
	 * Constructor telling whether to binarize the dataset or not
	 * 
	 * @param binarize
	 *            whether to binarize the dataset or not
	 * @throws IOException
	 */
	public KaggleMnistDataFetcher(boolean binarize) throws IOException {
		this(binarize, true, true, System.currentTimeMillis(), NUM_EXAMPLES);
	}

	public KaggleMnistDataFetcher(boolean binarize, boolean train, boolean shuffle, long rngSeed, int numExamples)
			throws IOException {
		Resource dataPath;

		// We commit Kaggle dataset in resources: 'classpath:/kaggle/digit_recognizer'
		if (train) {
			dataPath = new ClassPathResource("/kaggle/digit_recognizer/" + "train.csv");
			totalExamples = NUM_EXAMPLES;
		} else {
			dataPath = new ClassPathResource("/kaggle/digit_recognizer/" + "test.csv");
			totalExamples = NUM_EXAMPLES_TEST;
		}

		man = new KaggleMnistManager(dataPath, train);

		// Is this the number of possible values for single label?
		numOutcomes = 10;

		this.binarize = binarize;
		cursor = 0;
		inputColumns = RunKaggleDigitRecognizer.COLS * RunKaggleDigitRecognizer.COLS;
		this.train = train;
		this.shuffle = shuffle;

		if (train) {
			order = new int[NUM_EXAMPLES];
		} else {
			order = new int[NUM_EXAMPLES_TEST];
		}
		for (int i = 0; i < order.length; i++)
			order[i] = i;
		rng = new Random(rngSeed);
		this.numExamples = numExamples;
		reset(); // Shuffle order
	}

	public KaggleMnistDataFetcher() throws IOException {
		this(true);
	}

	@Override
	public void fetch(int numExamples) {
		if (!hasMore()) {
			throw new IllegalStateException("Unable to get more; there are no more images");
		}

		float[][] featureData = new float[numExamples][0];
		float[][] labelData = new float[numExamples][0];

		int actualExamples = 0;
		for (int i = 0; i < numExamples; i++, cursor++) {
			if (!hasMore())
				break;

			byte[] img = man.readImageUnsafe(order[cursor]);

			int label = man.readLabel(order[cursor]);

			float[] featureVec = new float[img.length];
			featureData[actualExamples] = featureVec;
			labelData[actualExamples] = new float[numOutcomes];
			labelData[actualExamples][label] = 1.0f;

			for (int j = 0; j < img.length; j++) {
				byte currentByte = img[j];
				if (currentByte != 0) {

					float v = Byte.toUnsignedInt(currentByte); // byte is loaded as signed -> convert to unsigned
					if (binarize) {
						if (v > 30.0f)
							featureVec[j] = 1.0f;
						else
							featureVec[j] = 0.0f;
					} else {
						featureVec[j] = v / 255.0f;
					}
				}
			}

			actualExamples++;
		}

		if (actualExamples < numExamples) {
			featureData = Arrays.copyOfRange(featureData, 0, actualExamples);
			labelData = Arrays.copyOfRange(labelData, 0, actualExamples);
		}

		INDArray features = Nd4j.create(featureData);
		INDArray labels = Nd4j.create(labelData);
		curr = new DataSet(features, labels);
	}

	@Override
	public void reset() {
		cursor = 0;
		curr = null;
		if (shuffle) {
			if ((train && numExamples < NUM_EXAMPLES) || (!train && numExamples < NUM_EXAMPLES_TEST)) {
				// Shuffle only first N elements
				if (firstShuffle) {
					MathUtils.shuffleArray(order, rng);
					firstShuffle = false;
				} else {
					MathUtils.shuffleArraySubset(order, numExamples, rng);
				}
			} else {
				MathUtils.shuffleArray(order, rng);
			}
		}
	}

	@Override
	public DataSet next() {
		DataSet next = super.next();
		return next;
	}

}
