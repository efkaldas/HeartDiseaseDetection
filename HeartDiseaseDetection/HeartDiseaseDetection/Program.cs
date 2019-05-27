using System;
using System.Globalization;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using HeartDiseasePredictionConsoleApp.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HeartDiseasePredictionConsoleApp
{
    public class TransformedData
    {
        public float[] NumericVector { get; set; }
    }

    public class InputData
    {
        [VectorType(14)]
        public float[] NumericVector { get; set; }
    };

    public class Program
    {
        const int crossValidatioN = 10;
        private static string BaseDatasetsRelativePath = @"../../../../Data";
        private static string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/HeartTraining.csv";
        private static string TestDataRelativePath = $"{BaseDatasetsRelativePath}/HeartTest.csv";

        private static string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
        private static string TestDataPath = GetAbsolutePath(TestDataRelativePath);


        private static string BaseModelsRelativePath = @"../../../../MLModels";
        private static string ModelRelativePath = $"{BaseModelsRelativePath}/HeartClassification.zip";

        private static string ModelPath = GetAbsolutePath(ModelRelativePath);

        public static void Main(string[] args)
        {
            var mlContext = new MLContext();

            bool naudojaNaujaMetoda = true;
            if (naudojaNaujaMetoda)
            {
                BuildTrainEvaluateAndSaveModelNEZINAU(mlContext);
              //  crossValidation();
            }
            else
            {
                BuildTrainEvaluateAndSaveModel(mlContext);
                TestPrediction(mlContext);
            }

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        /// <summary>
        /// Returns a few rows of data.
        /// </summary>
        private static IEnumerable<InputData> GetTrainData()
        {
            var trainData = new List<InputData>();
            string[] trainDataLines = File.ReadAllLines(TrainDataPath);
            for (int i = 0; i < trainDataLines.Length; i++)
            {
                string[] values = trainDataLines[i].Split(';', StringSplitOptions.RemoveEmptyEntries);
                float[] numericVector = new float[values.Length];
                for (int j = 0; j < values.Length; j++)
                {
                    numericVector[j] = float.Parse(values[j], CultureInfo.InvariantCulture.NumberFormat);
                }
                InputData inputdata = new InputData
                {
                    NumericVector = numericVector
                };
                trainData.Add(inputdata);
            }
            return trainData;
        }

        /// <summary>
        /// Returns a few rows of data.
        /// </summary>
        private static IEnumerable<InputData> GetTestData()
        {
            var testData = new List<InputData>();
            string[] testDataLines = File.ReadAllLines(TestDataPath);
            for (int i = 0; i < testDataLines.Length; i++)
            {
                string[] values = testDataLines[i].Split(';', StringSplitOptions.RemoveEmptyEntries);
                float[] numericVector = new float[values.Length];
                for (int j = 0; j < values.Length; j++)
                {
                    numericVector[j] = float.Parse(values[j], CultureInfo.InvariantCulture.NumberFormat);
                }
                InputData inputdata = new InputData
                {
                    NumericVector = numericVector
                };
                testData.Add(inputdata);
            }
            return testData;
        }
        /// <summary>
        /// Returns a few rows of data.
        /// </summary>
        private static IEnumerable<InputData> GetTestDatacross(int count)
        {
            var testData = new List<InputData>();
            string[] testDataLines = File.ReadAllLines(TestDataPath);
            int readcount = (int)(testDataLines.Length * (1.0M - (1.0M / crossValidatioN)));
            for (int i = count; i < testDataLines.Length; i++)
            {
                string[] values = testDataLines[i].Split(';', StringSplitOptions.RemoveEmptyEntries);
                float[] numericVector = new float[values.Length];
                for (int j = 0; j < values.Length; j++)
                {
                    numericVector[j] = float.Parse(values[j], CultureInfo.InvariantCulture.NumberFormat);
                }
                InputData inputdata = new InputData
                {
                    NumericVector = numericVector
                };
                testData.Add(inputdata);
            }
            return testData;
        }

        /*
         * !!! Nežinau, ar gerai pritaikiau dimensijų sumažinimą, tai kol kas sukūriau atskirą metodą :D
         * !!! Trūksta mašininio mokymosi metodo
         */
        private static void BuildTrainEvaluateAndSaveModelNEZINAU(MLContext mlContext)
        {
            // Reikšmės ilgis spausdinant
            int valueLength = 4;

            // Spausdinamų duomenų (eilučių) kiekis
            int dataPrintCount = 10;

            // Slenkstis
            int threshold = 100;

            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawTrainData = GetTrainData();
            var rawTestData = GetTestData();

            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromEnumerable(rawTrainData);
            var testDataView = mlContext.Data.LoadFromEnumerable(rawTestData);

            // Spausdina duomenis PRIEŠ dimensijų sumažinimą
            Console.WriteLine("=============== Input data BEFORE FeatureSelection(threshold: {0}) ===============", threshold);
            int count = 0;
            foreach (var item in rawTrainData)
            {
                count++;
                if (count <= dataPrintCount)
                {
                    Console.Write("|");
                    Console.WriteLine(string.Concat(item.NumericVector.Select(i => string.Format("{0, " + valueLength + "}|", i))));
                }
                else
                {
                    break;
                }
            }

            // STEP 2: We will use the SelectFeaturesBasedOnCount transform estimator, to retain only those slots which have at least 'count' non-default values per slot
            // FeatureSelection(): dimensijų sumažinimas (nežinau, ar gerai padariau)
            var pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(new InputOutputColumnPair[] { new InputOutputColumnPair("NumericVector") }, count: threshold);

            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("=============== Training the model ===============");
            ITransformer trainedModel = pipeline.Fit(trainingDataView);
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("=============== Finish the train model. Push Enter ===============");
            Console.WriteLine("");
            Console.WriteLine("");

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            Console.WriteLine("");
            var predictions = trainedModel.Transform(testDataView);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(predictions, true);

            // Spausdina duomenis PO dimensijų sumažinimo
            Console.WriteLine("");
            Console.WriteLine("=============== Input data AFTER FeatureSelection(threshold: {0}) ===============", threshold);
            count = 0;
            foreach (var item in convertedData)
            {
                count++;
                if (count <= dataPrintCount)
                {
                    Console.Write("|");
                    Console.WriteLine(string.Concat(item.NumericVector.Select(i => string.Format("{0, " + valueLength + "}|", i))));
                }
                else
                {
                    break;
                }
            }

            // !!! metrics neveikia, dar nežinau kaip sutvarkyt
            /*
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {trainedModel.ToString()} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Area Under Roc Curve:      {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       Area Under PrecisionRecall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            Console.WriteLine($"************************************************************");
            */
            Console.WriteLine("");
            Console.WriteLine("");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("=============== Model Saved ============= ");
        }

        private static void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
        {
            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataPath, hasHeader: true, separatorChar: ';');
            var testDataView = mlContext.Data.LoadFromTextFile<HeartData>(TestDataPath, hasHeader: true, separatorChar: ';');

            // STEP 2: Concatenate the features and set the training algorithm
            var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));

            Console.WriteLine("=============== Training the model ===============");
            ITransformer trainedModel = pipeline.Fit(trainingDataView);
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("=============== Finish the train model. Push Enter ===============");
            Console.WriteLine("");
            Console.WriteLine("");

            Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
            var predictions = trainedModel.Transform(testDataView);

            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {trainedModel.ToString()} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"*       Area Under Roc Curve:      {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*       Area Under PrecisionRecall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");
            Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");
            Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");
            Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            Console.WriteLine($"************************************************************");
            Console.WriteLine("");
            Console.WriteLine("");

            Console.WriteLine("=============== Saving the model to a file ===============");
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
            Console.WriteLine("");
            Console.WriteLine("");
            Console.WriteLine("=============== Model Saved ============= ");
        }


        private static void TestPrediction(MLContext mlContext)
        {
            ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<HeartData, HeartPrediction>(trainedModel);                   

            foreach (var heartData in HeartSampleData.heartDataList)
            {
                var prediction = predictionEngine.Predict(heartData);

                Console.WriteLine($"=============== Single Prediction  ===============");
                Console.WriteLine($"Age: {heartData.Age} ");
                Console.WriteLine($"Sex: {heartData.Sex} ");
                Console.WriteLine($"Cp: {heartData.Cp} ");
                Console.WriteLine($"TrestBps: {heartData.TrestBps} ");
                Console.WriteLine($"Chol: {heartData.Chol} ");
                Console.WriteLine($"Fbs: {heartData.Fbs} ");
                Console.WriteLine($"RestEcg: {heartData.RestEcg} ");
                Console.WriteLine($"Thalac: {heartData.Thalac} ");
                Console.WriteLine($"Exang: {heartData.Exang} ");
                Console.WriteLine($"OldPeak: {heartData.OldPeak} ");
                Console.WriteLine($"Slope: {heartData.Slope} ");
                Console.WriteLine($"Ca: {heartData.Ca} ");
                Console.WriteLine($"Thal: {heartData.Thal} ");
                Console.WriteLine($"Prediction Value: {prediction.Prediction} ");
                Console.WriteLine($"Prediction: {(prediction.Prediction ? "A disease could be present" : "Not present disease" )} ");
                Console.WriteLine($"Probability: {prediction.Probability} ");
                Console.WriteLine($"==================================================");
                Console.WriteLine("");
                Console.WriteLine("");
            }

        }
        //Veikia ne visai kaip reikia bet dirbam :D
        private static void crossValidation()
        {
            int threshold = 100;
            // Reikšmės ilgis spausdinant
            int valueLength = 4;
            int countt = 0;
            // Spausdinamų duomenų (eilučių) kiekis
            int dataPrintCount = 10;
            for (int x = 0; x < crossValidatioN; x++)
            {
                var mlContext = new MLContext();
                var testData = GetTestDatacross(countt);
                ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);
                string[] testDataLines = File.ReadAllLines(TestDataPath);
                var testDataView = mlContext.Data.LoadFromEnumerable(testData);
                int testDataCount = testDataLines.Length;
                mlContext.Data.LoadFromEnumerable(testData);
                int testToRead = testDataCount - (int)(testDataCount * (1.0M - (1.0M / crossValidatioN)));

                for (int i = x * testDataCount; i < x * testDataCount + testDataCount - 2; i++)
                {


                    var predictions = trainedModel.Transform(testDataView);

                    var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(predictions, true);

                    Console.WriteLine("");
                    Console.WriteLine("=============== Input data AFTER FeatureSelection(threshold: {0}) ===============", threshold);
                    int count = 0;
                    foreach (var item in convertedData)
                    {
                        count++;
                        if (count <= dataPrintCount)
                        {
                            Console.Write("|");
                            Console.WriteLine(string.Concat(item.NumericVector.Select(z => string.Format("{0, " + valueLength + "}|", z))));
                        }
                        else
                        {
                            break;
                        }
                    }



                }
                countt += testToRead;
            }
        }


        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;

        }
    }
}
