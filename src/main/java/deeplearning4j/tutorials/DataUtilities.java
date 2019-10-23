package deeplearning4j.tutorials;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;

/**
 * Common data utility functions.
 * 
 * @author fvaleri
 */
public class DataUtilities {

	/**
	 * Download a remote file if it doesn't exist.
	 * 
	 * @param remoteUrl
	 *            URL of the remote file.
	 * @param localPath
	 *            Where to download the file.
	 * @return True if and only if the file has been downloaded.
	 * @throws Exception
	 *             IO error.
	 */
	public static boolean downloadFile(String remoteUrl, String localPath) throws IOException {
		boolean downloaded = false;
		if (remoteUrl == null || localPath == null)
			return downloaded;
		File file = new File(localPath);
		if (!file.exists()) {
			file.getParentFile().mkdirs();

			try (BufferedInputStream in = new BufferedInputStream(new URL(remoteUrl).openStream())) {
				Files.copy(in, file.toPath(), StandardCopyOption.REPLACE_EXISTING);
			}
			downloaded = true;
		}
		if (!file.exists())
			throw new IOException("File doesn't exist: " + localPath);
		return downloaded;
	}

	/**
	 * Extract a "tar.gz" file into a local folder.
	 * 
	 * @param inputPath
	 *            Input file path.
	 * @param outputPath
	 *            Output directory path.
	 * @throws IOException
	 *             IO error.
	 */
	public static void extractTarGz(String inputPath, String outputPath) throws IOException {
		if (inputPath == null || outputPath == null)
			return;
		final int bufferSize = 4096;
		if (!outputPath.endsWith("" + File.separatorChar))
			outputPath = outputPath + File.separatorChar;
		try (TarArchiveInputStream tais = new TarArchiveInputStream(
				new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(inputPath))))) {
			TarArchiveEntry entry;
			while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
				if (entry.isDirectory()) {
					new File(outputPath + entry.getName()).mkdirs();
				} else {
					int count;
					byte data[] = new byte[bufferSize];
					FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
					BufferedOutputStream dest = new BufferedOutputStream(fos, bufferSize);
					while ((count = tais.read(data, 0, bufferSize)) != -1) {
						dest.write(data, 0, count);
					}
					dest.close();
				}
			}
		}
	}

}