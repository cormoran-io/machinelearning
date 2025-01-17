/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package deeplearning4j.tutorials.word2vec;

import java.io.File;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import deeplearning4j.tutorials.DataUtilities;

/**
 * Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the
 * words it contains. This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a
 * review is vectorized (using the Word2Vec model) and fed into a recurrent neural network. Training data is the "Large
 * Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/ This data set contains 25,000 training
 * reviews + 25,000 testing reviews
 *
 * Process: 1. Automatic on first run of example: Download data (movie reviews) + extract 2. Load existing Word2Vec
 * model (for example: Google News word vectors. You will have to download this MANUALLY) 3. Load each each review.
 * Convert words to vectors + reviews to sequences of vectors 4. Train network
 *
 * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
 * additional tuning.
 *
 * NOTE / INSTRUCTIONS: You will have to download the Google News word vector model manually. ~1.5GB The Google News
 * vector model available here: https://code.google.com/p/word2vec/ Download the GoogleNews-vectors-negative300.bin.gz
 * file Then: set the WORD_VECTORS_PATH field to point to this location.
 *
 * @author Alex Black
 */
public class Word2VecSentimentRNN {
	private static final Logger LOGGER = LoggerFactory.getLogger(Word2VecSentimentRNN.class);

	/** Location (local file system) for the Google News vectors. Set this manually (~1.5GB). */
	public static final String WORD_VECTORS_PATH =
			System.getProperty("user.home") + "/Downloads" + "/GoogleNews-vectors-negative300.bin.gz";

	/** Data URL for downloading */
	public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
	/** Location to save and extract the training/testing data */
	public static final String DATA_PATH =
			FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");

	/**
	 * This main method should end printing a very negative review like 'Hated it with all my being. Worst movie ever.
	 * Mentally- scarred. Help me. It was that bad.TRUST ME!!!'
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		if (WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")) {
			throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
		}

		// Download and extract data
		downloadData();

		int batchSize = 64; // Number of examples in each minibatch
		int vectorSize = 300; // Size of the word vectors. 300 in the Google News model
		int nEpochs = 1; // Number of epochs (full passes of training data) to train on
		int truncateReviewsToLength = 256; // Truncate reviews with length (# words) greater than this
		final int seed = 0; // Seed for reproducibility

		Nd4j.getMemoryManager().setAutoGcWindow(10000); // https://deeplearning4j.org/workspaces

		// Set up network configuration
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.updater(new Adam(5e-3))
				.l2(1e-5)
				.weightInit(WeightInit.XAVIER)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0)
				.list()
				.layer(new LSTM.Builder().nIn(vectorSize).nOut(256).activation(Activation.TANH).build())
				.layer(new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
						.lossFunction(LossFunctions.LossFunction.MCXENT)
						.nIn(256)
						.nOut(2)
						.build())
				.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		// DataSetIterators for training and testing respectively
		WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
		SentimentExampleIterator train =
				new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, true);
		SentimentExampleIterator test =
				new SentimentExampleIterator(DATA_PATH, wordVectors, batchSize, truncateReviewsToLength, false);

		LOGGER.info("Starting training");
		net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(test, 1, InvocationType.EPOCH_END));
		net.fit(train, nEpochs);

		// After training: load a single example and generate predictions
		File shortNegativeReviewFile = new File(FilenameUtils.concat(DATA_PATH, "aclImdb/test/neg/12100_1.txt"));
		String shortNegativeReview = FileUtils.readFileToString(shortNegativeReviewFile, (Charset) null);

		INDArray features = test.loadFeaturesFromString(shortNegativeReview, truncateReviewsToLength);
		INDArray networkOutput = net.output(features);
		long timeSeriesLength = networkOutput.size(2);
		INDArray probabilitiesAtLastWord =
				networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

		LOGGER.info("\n\n-------------------------------");
		LOGGER.info("Short negative review: \n" + shortNegativeReview);
		LOGGER.info("\n\nProbabilities at last time step:");
		LOGGER.info("p(positive): " + probabilitiesAtLastWord.getDouble(0));
		LOGGER.info("p(negative): " + probabilitiesAtLastWord.getDouble(1));

		LOGGER.info("----- Example complete -----");

		LOGGER.info("Plot TSNE....");
		BarnesHutTsne tsne = new BarnesHutTsne.Builder().setMaxIter(1000)
				.stopLyingIteration(250)
				.learningRate(500)
				.useAdaGrad(false)
				.theta(0.5)
				.setMomentum(0.5)
				.normalize(true)
				// .usePca(false)
				.build();
		wordVectors.lookupTable().plotVocab(tsne, 100, Files.createTempFile("kaggle", ".jpg").toFile());

	}

	public static void downloadData() throws Exception {
		// Create directory if required
		File directory = new File(DATA_PATH);
		if (!directory.exists())
			directory.mkdir();

		// Download file:
		String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
		File archiveFile = new File(archizePath);
		String extractedPath = DATA_PATH + "aclImdb";
		File extractedFile = new File(extractedPath);

		if (!archiveFile.exists()) {
			LOGGER.info("Starting data download (80MB)...");
			FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
			LOGGER.info("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
			// Extract tar.gz file to output directory
			DataUtilities.extractTarGz(archizePath, DATA_PATH);
		} else {
			// Assume if archive (.tar.gz) exists, then data has already been extracted
			LOGGER.info("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
			if (!extractedFile.exists()) {
				// Extract tar.gz file to output directory
				DataUtilities.extractTarGz(archizePath, DATA_PATH);
			} else {
				LOGGER.info("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
			}
		}
	}

}