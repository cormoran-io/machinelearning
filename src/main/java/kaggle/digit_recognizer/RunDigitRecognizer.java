package kaggle.digit_recognizer;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;

// https://www.kaggle.com/c/digit-recognizer/
// https://deeplearning4j.org/tutorials/00-quickstart-for-deeplearning4j
public class RunDigitRecognizer {
	public static void main(String[] args) throws IOException {

		int batchSize = 16; // how many examples to simultaneously train in the network
		EmnistDataSetIterator.Set emnistSet = EmnistDataSetIterator.Set.BALANCED;
		EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator(emnistSet, batchSize, true);
		EmnistDataSetIterator emnistTest = new EmnistDataSetIterator(emnistSet, batchSize, false);
	}
}
