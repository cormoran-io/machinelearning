package kaggle.titanic;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.TransformProcess.Builder;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Rickard Edén (github.com/neph1)
 *
 *         In words: https://www.kaggle.com/c/titanic/discussion/61841
 *
 */
public class Titanic {

	private static Logger LOGGER = LoggerFactory.getLogger(Titanic.class);

	public static void main(String[] args) throws IOException, InterruptedException {
		// int labelIndex = 6;
		int numClasses = 2; // survived or not
		int batchSize = 892; // size of the dataset

		// Constant seed for reproducibility
		long seed = 12;

		MultiLayerNetwork model;

		try (RecordReader recordReader = makeCsvRecordReader()) {
			recordReader.initialize(new FileSplit(new ClassPathResource("kaggle/titanic/train.csv").getFile()));

			TransformProcess.Builder builder = new TransformProcess.Builder(makeSchema(true));
			TransformProcess transformProcess = testAndTestTransform(builder).removeColumns("PassengerId").build();
			RecordReader transformedRecordReader = new TransformProcessRecordReader(recordReader, transformProcess);

			// As we excluded passengerId, the survived column (i.e. the label) is at index 0
			int labelIndex = 0;
			DataSetIterator iterator = new RecordReaderDataSetIterator.Builder(transformedRecordReader, batchSize)
					.classification(labelIndex, numClasses)
					.build();

			DataSet allData = iterator.next();
			allData.shuffle(25);

			// Keep 90% of entries for training, and 10% for testing
			SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.9);

			DataSet trainingData = testAndTrain.getTrain();

			final int numInputs = 7;
			int outputNum = 2;

			LOGGER.info("Build model....");
			model = getModel(seed, numInputs, outputNum);
			model.init();
			model.setListeners(new ScoreIterationListener(100));
			model.pretrain(iterator);
			for (int i = 0; i < 10000; i++) {
				trainingData.shuffle(25);
				model.fit(trainingData);
			}

			// evaluate the model on the test set
			{
				Evaluation eval = new Evaluation(2);

				DataSet testData = testAndTrain.getTest();
				INDArray output = model.output(testData.getFeatures());
				eval.eval(testData.getLabels(), output);
				LOGGER.info(eval.stats());
			}
		}

		{
			INDArray output;
			INDArray features;
			try (CSVRecordReader recordReader = makeCsvRecordReader()) {
				recordReader.initialize(new FileSplit(new ClassPathResource("kaggle/titanic/test.csv").getFile()));

				TransformProcess.Builder builder = new TransformProcess.Builder(makeSchema(false));
				TransformProcess testTransformProcess = testAndTestTransform(builder)
						// .removeColumns("PassengerId")
						.build();
				RecordReader transformedRecordReader =
						new TransformProcessRecordReader(recordReader, testTransformProcess);
				DataSetIterator iterator =
						new RecordReaderDataSetIterator.Builder(transformedRecordReader, 418).build();

				DataSet verifyData = iterator.next();
				List<String> labelNames = new ArrayList<>();
				labelNames.add("Dead");
				labelNames.add("Alive");
				verifyData.setLabelNames(labelNames);
				// normalizer.transform(verifyData); //Apply normalization to the test data. This is using statistics
				// calculated
				// from the *training* set
				features = verifyData.getFeatures();

				// Exclude the passengerId column at index 0
				output = model.output(features.getColumns(IntStream.range(1, features.columns()).toArray()));
			}

			String outputPath =
					System.getProperty("user.dir") + "/src/main/resources/kaggle/titanic/gender_submission.csv";
			File outputFile = new File(outputPath);
			outputFile.delete();
			outputFile.createNewFile();
			try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
				writer.write("PassengerId,Survived");
				writer.newLine();

				for (int i = 0; i < output.rows(); i++) {
					boolean alive = output.getRow(i).getFloat(1) > output.getRow(i).getFloat(0);
					int passengerId = features.getRow(i).getInt(0);
					writer.append(Integer.toString(passengerId)).append(",").append(Integer.toString(alive ? 1 : 0));
					writer.newLine();
				}
			}
		}
	}

	// Schema():
	// idx name type meta data
	// 0 "Survived" Integer IntegerMetaData(name="Survived",)
	// 1 "Pclass" Categorical CategoricalMetaData(name="Pclass",stateNames=["1","2","3"])
	// 2 "Name" String StringMetaData(name="Name",)
	// 3 "Sex" Categorical CategoricalMetaData(name="Sex",stateNames=["female","male"])
	// 4 "Age" Integer IntegerMetaData(name="Age",)
	// 5 "Siblings/Spouses Aboard" Integer IntegerMetaData(name="Siblings/Spouses Aboard",)
	// 6 "Parents/Children Aboard" Integer IntegerMetaData(name="Parents/Children Aboard",)
	// 7 "SibSp" String StringMetaData(name="SibSp",)
	// 8 "Parch" String StringMetaData(name="Parch",)
	// 9 "Ticket" String StringMetaData(name="Ticket",)
	// 10 "Fare" Double DoubleMetaData(name="Fare",allowNaN=false,allowInfinite=false)
	// 11 "Cabin" String StringMetaData(name="Cabin",)
	// 12 "Embarked" String StringMetaData(name="Embarked",)

	// PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
	private static Schema makeSchema(boolean withSurvivedColumn) {
		Schema.Builder schemaBuilder = new Schema.Builder();
		schemaBuilder.addColumnInteger("PassengerId");

		if (withSurvivedColumn) {
			schemaBuilder.addColumnInteger("Survived");
		}

		schemaBuilder.addColumnCategorical("Pclass", Arrays.asList("1", "2", "3"))
				.addColumnString("Name")
				.addColumnCategorical("Sex", Arrays.asList("male", "female"))
				.addColumnsInteger("Age", "Siblings/Spouses Aboard", "Parents/Children Aboard")
				.addColumnString("Ticket")
				.addColumnDouble("Fare")
				.addColumnString("Cabin")
				.addColumnCategorical("Embarked", Arrays.asList("", "S", "C", "Q"));
		return schemaBuilder.build();
	}

	private static Builder testAndTestTransform(TransformProcess.Builder builder) {
		return builder.removeColumns("Name", "Fare")
				.categoricalToInteger("Sex", "Embarked")
				.categoricalToOneHot("Pclass")
				.removeColumns("Pclass[1]")
				// Cabin is sometimes empty
				.removeColumns("Cabin")
				// Ticket has a weird format: A/5 21171
				.removeColumns("Ticket")
				// We suppose people with no age are 18
				.conditionalReplaceValueTransform("Age",
						new Text("18"),
						new StringColumnCondition("Age", ConditionOp.Equal, ""));
	}

	private static CSVRecordReader makeCsvRecordReader() {
		return new CSVRecordReader(1, ',');
	}

	private static MultiLayerNetwork getModel(long seed, int numInputs, int outputNum) {
		int layerOne = 9;
		int layerTwo = 9;
		int layerThree = 4;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.activation(Activation.RELU)
				.weightInit(WeightInit.XAVIER)
				// .updater(new Sgd(0.18))
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Nesterovs(0.17, 0.25))
				.list()
				.layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(layerOne).build())
				.layer(1, new DenseLayer.Builder().nIn(layerOne).nOut(layerTwo).build())
				.layer(2, new DenseLayer.Builder().nIn(layerTwo).nOut(layerThree).build())
				.layer(3,
						new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID)
								.nIn(layerThree)
								.nOut(outputNum)
								.build())
				// .backprop(true).pretrain(false)
				.build();

		return new MultiLayerNetwork(conf);
	}

}
