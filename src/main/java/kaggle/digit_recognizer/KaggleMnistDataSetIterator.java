package kaggle.digit_recognizer;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

/**
 * MNIST data set iterator restricted to the data provided by Kaggle
 * 
 * https://www.kaggle.com/c/digit-recognizer/data
 *
 * @author Benoit Lacelle
 */
public class KaggleMnistDataSetIterator extends BaseDatasetIterator {
	private static final long serialVersionUID = 7734859035509683992L;

	public KaggleMnistDataSetIterator(int batch, int numExamples) throws IOException {
		this(batch, numExamples, false);
	}

	/**
	 * Get the specified number of examples for the MNIST training data set.
	 * 
	 * @param batch
	 *            the batch size of the examples
	 * @param numExamples
	 *            the overall number of examples
	 * @param binarize
	 *            whether to binarize mnist or not
	 * @throws IOException
	 */
	public KaggleMnistDataSetIterator(int batch, int numExamples, boolean binarize) throws IOException {
		this(batch, numExamples, binarize, true, false, 0);
	}

	/**
	 * Constructor to get the full MNIST data set (either test or train sets) without binarization (i.e., just
	 * normalization into range of 0 to 1), with shuffling based on a random seed.
	 * 
	 * @param batchSize
	 * @param train
	 * @throws IOException
	 */
	public KaggleMnistDataSetIterator(int batchSize, boolean train, int seed) throws IOException {
		this(batchSize,
				(train ? KaggleMnistDataFetcher.NUM_EXAMPLES : KaggleMnistDataFetcher.NUM_EXAMPLES_TEST),
				false,
				train,
				true,
				seed);
	}

	/**
	 * Get the specified number of MNIST examples (test or train set), with optional shuffling and binarization.
	 * 
	 * @param batch
	 *            Size of each patch
	 * @param numExamples
	 *            total number of examples to load
	 * @param binarize
	 *            whether to binarize the data or not (if false: normalize in range 0 to 1)
	 * @param train
	 *            Train vs. test set
	 * @param shuffle
	 *            whether to shuffle the examples
	 * @param rngSeed
	 *            random number generator seed to use when shuffling examples
	 */
	public KaggleMnistDataSetIterator(int batch,
			int numExamples,
			boolean binarize,
			boolean train,
			boolean shuffle,
			long rngSeed) throws IOException {
		super(batch, numExamples, new KaggleMnistDataFetcher(binarize, train, shuffle, rngSeed, numExamples));
	}
}
