package kaggle.digit_recognizer;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import org.datavec.image.mnist.MnistManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;

/**
 * Enable loading of MNIST images in the format provided by Kaggle
 */
public class KaggleMnistManager {
	private static final Logger LOGGER = LoggerFactory.getLogger(KaggleMnistManager.class);

	private byte[][] imagesArr;
	private int[] labelsArr;

	/**
	 * Writes the given image in the given file using the PPM data format.
	 *
	 * @param image
	 * @param ppmFileName
	 * @throws IOException
	 */
	public static void writeImageToPpm(int[][] image, String ppmFileName) throws IOException {
		MnistManager.writeImageToPpm(image, ppmFileName);
	}

	public KaggleMnistManager(Resource dataPath, boolean train) throws IOException {
		this(dataPath, train, train ? KaggleMnistDataFetcher.NUM_EXAMPLES : KaggleMnistDataFetcher.NUM_EXAMPLES_TEST);
	}

	public KaggleMnistManager(Resource dataPath, boolean train, int numExamples) throws IOException {
		imagesArr = new byte[numExamples][];
		labelsArr = new int[numExamples];

		try (Stream<String> stream = Files.lines(dataPath.getFile().toPath())) {
			AtomicInteger rowIndex = new AtomicInteger();

			// Skip the header row
			stream.skip(1).forEach(row -> {
				int rowI = rowIndex.getAndIncrement();

				if (rowI % 10000 == 0) {
					LOGGER.info("Parsed #{}", rowI);
				}

				String[] pixels = row.split(",");

				// We have the label only in the train data
				int readShift;
				if (train) {
					if (pixels.length != 1 + RunKaggleDigitRecognizer.COLS * RunKaggleDigitRecognizer.COLS) {
						throw new IllegalArgumentException("Invalid row");
					}

					readShift = 1;
				} else {
					if (pixels.length != RunKaggleDigitRecognizer.COLS * RunKaggleDigitRecognizer.COLS) {
						throw new IllegalArgumentException("Invalid row");
					}
					readShift = 0;
				}

				byte[] bytes = new byte[pixels.length - readShift];
				for (int i = readShift; i < pixels.length; i++) {
					// https://stackoverflow.com/questions/7401550/how-to-convert-int-to-unsigned-byte-and-back
					bytes[i - readShift] = (byte) Integer.parseInt(pixels[i]);
				}
				imagesArr[rowI] = bytes;

				if (train) {
					labelsArr[rowI] = Integer.parseInt(pixels[0]);
				} else {
					// We set the label 0 as -1 would lead to issues as it a not-existant label
					labelsArr[rowI] = 0;

					if (rowI < 100) {
						for (int i = 0; i < RunKaggleDigitRecognizer.COLS * RunKaggleDigitRecognizer.COLS; i++) {
							if (i % RunKaggleDigitRecognizer.COLS == 0) {
								System.out.println();
							}
							if (bytes[i] == 0) {
								System.out.print(' ');
							} else {
								System.out.print('x');
							}
						}
						System.out.println("--------------------");
					}
				}
			});
		} catch (IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	public byte[] readImageUnsafe(int i) {
		return imagesArr[i];
	}

	public int readLabel(int i) {
		return labelsArr[i];
	}

}
