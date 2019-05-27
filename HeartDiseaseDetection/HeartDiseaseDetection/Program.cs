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
        [VectorType(13)]
        public float[] NumericVector { get; set; }
    }

    public class Program
    {
        // Reikšmės ilgis spausdinant
        private static int valueLength = 4;

        // Spausdinamų duomenų (eilučių) kiekis
        private static int dataPrintCount = 10;

        // Slenkstis, naudojamas dimensijų sumažinimui, t.y. nurodo, kad stulpelyje turi būti bent nurodytas kiekis (>= slenkstis) non-default elementų
        // Jei stulpelyje non-default elementų skaičius yra mažesnis už slenkstį, stulpelis (atributas) bus atmetamas
        // Pavyzdys: slenkstis = 3; NaN yra default reikšmė
        // PRIEŠ dimensijų sumažinimą
        // 4, NaN,   6
        // 4,   5,   6
        // 4,   5,   6
        // 4, NaN, NaN
        // PO dimensijų sumažinimo: antras stulpelis (atributas) atmetamas, nes jame yra mažiau nei 3 (< slenkstis) non-default (ne NaN) reikšmės
        // 4,        6
        // 4,        6
        // 4,        6
        // 4,        NaN
        private static int threshold = 282;

        // Kryžminės patikros fold count
        private static int foldCount = 5;

        // k-means metodo cluster count
        private static int clusterCount = 3;

        // Metodo numeris
        // methodNumber =  1: FieldAwareFactorizationMachine()
        // methodNumber =  2: AveragedPerceptron()
        // methodNumber =  3: LbfgsLogisticRegression()
        // methodNumber =  4: LinearSvm()
        // methodNumber =  5: Prior()
        // methodNumber =  6: SdcaLogisticRegression()
        // methodNumber =  7: SdcaNonCalibrated()
        // methodNumber =  8: SgdCalibrated()
        // methodNumber =  9: SgdNonCalibrated()
        // methodNumber = 10: FastForest()
        // methodNumber = 11: FastTree()
        // methodNumber = 12: Gam()
        // methodNumber = 13: KMeans()
        private static int methodNumber = 11;

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

            bool naudojaNaujaMetoda = false;
            bool naudojaClusteringKmeans = true;
            if (naudojaClusteringKmeans)
            {
                ClusteringKmeans();
            }
            else if (naudojaNaujaMetoda)
            {
                BuildTrainEvaluateAndSaveModelNEZINAU(mlContext);
                TestPrediction(mlContext);
            }
            else
            {
                //BuildTrainEvaluateAndSaveModel(mlContext);
                //BuildTrainEvaluateAndSaveModelGam(mlContext);
                BuildTrainEvaluateAndSaveModelLbfgs(mlContext);
                TestPrediction(mlContext);
            }

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        public static void ClusteringKmeans()
        {
            var mlContext = new MLContext(seed: 0);
            IDataView dataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataRelativePath, hasHeader: false, separatorChar: ';');
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms.Concatenate(featuresColumnName, "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));
            var model = pipeline.Fit(dataView);

            using (var fileStream = new FileStream(ModelRelativePath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }


            var predictor = mlContext.Model.CreatePredictionEngine<HeartData, HeartPredictionClusters>(model);
            for (int i = 0; i < HeartSampleData.heartDataList.Count; i++)
            {

                var prediction = predictor.Predict(HeartSampleData.heartDataList[i]);

                Console.WriteLine($"Disease Prediction {i + 1}");
                Console.WriteLine($"Probabilities: {prediction.Probability[0] / 100}");
            }
        }

        /// <summary>
        /// Returns a few rows of data.
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        private static IEnumerable<InputData> GetData(string filePath)
        {
            var data = new List<InputData>();
            string[] dataLines = File.ReadAllLines(filePath);
            for (int i = 0; i < dataLines.Length; i++)
            {
                string[] values = dataLines[i].Split(';', StringSplitOptions.RemoveEmptyEntries);

                // Imam [values.Length - 1], nes nereikia paskutinio atributo (the predicted attribute)
                float[] numericVector = new float[values.Length - 1];
                for (int j = 0; j < values.Length - 1; j++)
                {
                    numericVector[j] = float.Parse(values[j], CultureInfo.InvariantCulture.NumberFormat);
                }
                InputData inputdata = new InputData
                {
                    NumericVector = numericVector
                };
                data.Add(inputdata);
            }
            return data;
        }


        /// <summary>
        /// Gauna likusius stulpelių (atributų) vardus, kurių neatmetė dimensijų sumažinimas.
        /// </summary>
        /// <param name="convertedData"></param>
        /// <param name="filePath"></param>
        /// <returns></returns>
        private static string[] GetRemainingColumns(IEnumerable<TransformedData> convertedData, string filePath)
        {
            IDictionary<string, List<float>> allColumns = new Dictionary<string, List<float>>
            {
                { "Age", new List<float>() },
                { "Sex", new List<float>() },
                { "Cp", new List<float>() },
                { "TrestBps", new List<float>() },
                { "Chol", new List<float>() },
                { "Fbs", new List<float>() },
                { "RestEcg", new List<float>() },
                { "Thalac", new List<float>() },
                { "Exang", new List<float>() },
                { "OldPeak", new List<float>() },
                { "Slope", new List<float>() },
                { "Ca", new List<float>() },
                { "Thal", new List<float>() }
            };
            IDictionary<int, List<float>> remainingColumns = new Dictionary<int, List<float>>();
            for (int i = 0; i < convertedData.ElementAt(0).NumericVector.Length; i++)
            {
                remainingColumns.Add(i, new List<float>());
            }
            foreach (var item in convertedData)
            {
                for (int i = 0; i < item.NumericVector.Length; i++)
                {
                    remainingColumns[remainingColumns.Keys.ElementAt(i)].Add(item.NumericVector[i]);
                }
            }
            string[] dataLines = File.ReadAllLines(filePath);
            for (int i = 0; i < dataLines.Length; i++)
            {
                string[] values = dataLines[i].Split(';', StringSplitOptions.RemoveEmptyEntries);

                // Imam [values.Length - 1], nes nereikia paskutinio atributo (the predicted attribute)
                for (int j = 0; j < values.Length - 1; j++)
                {
                    allColumns[allColumns.Keys.ElementAt(j)].Add(float.Parse(values[j], CultureInfo.InvariantCulture.NumberFormat));
                }
            }
            string[] remainingColumnNames = new string[remainingColumns.Count];
            for (int i = 0; i < remainingColumns.Count; i++)
            {
                for (int j = 0; j < allColumns.Count; j++)
                {
                    if (remainingColumns[remainingColumns.Keys.ElementAt(i)].SequenceEqual(allColumns[allColumns.Keys.ElementAt(j)]))
                    {
                        remainingColumnNames[i] = allColumns.Keys.ElementAt(j);
                        break;
                    }
                }
            }
            return remainingColumnNames;
        }

        /*
         * !!! Nežinau, ar gerai pritaikiau dimensijų sumažinimą ir kryžminę patikrą, tai kol kas sukūriau atskirą metodą :D
         * !!! Reikia dar nuspręst dėl mašininio mokymosi metodų
         */
        private static void BuildTrainEvaluateAndSaveModelNEZINAU(MLContext mlContext)
        {
            // Get a small dataset as an IEnumerable and convert it to an IDataView.
            var rawTrainData = GetData(TrainDataPath);
            var rawTestData = GetData(TestDataPath);

            // Common data loading configuration
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

            // We will use the SelectFeaturesBasedOnCount transform estimator, to retain only those slots which have at least 'count' non-default values per slot
            var pipeline = mlContext.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(new InputOutputColumnPair[] { new InputOutputColumnPair("NumericVector") }, count: threshold);

            var transformedData = pipeline.Fit(trainingDataView).Transform(trainingDataView);

            var convertedData = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, true);

            // Spausdina duomenis PO dimensijų sumažinimo
            Console.WriteLine("");
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

            // Gauna likusius stulpelių (atributų) vardus, kurių neatmetė dimensijų sumažinimas
            string[] remainingColumns = GetRemainingColumns(convertedData, TrainDataPath);

            // Nauji duomenys po dimensijų sumažinimo
            var newTrainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataPath, hasHeader: true, separatorChar: ';');
            var newTestDataView = mlContext.Data.LoadFromTextFile<HeartData>(TestDataPath, hasHeader: true, separatorChar: ';');

            double crossValidationSum;
            double crossValidationAverage;

            // Metodo numeris
            // methodNumber =  1: FieldAwareFactorizationMachine()
            // methodNumber =  2: AveragedPerceptron()
            // methodNumber =  3: LbfgsLogisticRegression()
            // methodNumber =  4: LinearSvm()
            // methodNumber =  5: Prior()
            // methodNumber =  6: SdcaLogisticRegression()
            // methodNumber =  7: SdcaNonCalibrated()
            // methodNumber =  8: SgdCalibrated()
            // methodNumber =  9: SgdNonCalibrated()
            // methodNumber = 10: FastForest()
            // methodNumber = 11: FastTree()
            // methodNumber = 12: Gam()
            // methodNumber = 13: KMeans()
            if (methodNumber == 1)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for FieldAwareFactorizationMachine()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 2)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for AveragedPerceptron()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 3)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for LbfgsLogisticRegression()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 4)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for LinearSvm()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 5)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.Prior(labelColumnName: "Label"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for Prior()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 6)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for SdcaLogisticRegression()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 7)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for SdcaNonCalibrated()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 8)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.SgdCalibrated(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for SgdCalibrated()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 9)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.SgdNonCalibrated(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for SgdNonCalibrated()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 10)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for FastForest()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 11)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for FastTree()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else if (methodNumber == 12)
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.BinaryClassification.Trainers.Gam(labelColumnName: "Label", featureColumnName: "Features"));

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.BinaryClassification.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.Accuracy);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for Gam()      ");
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
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
            else
            {
                var newPipeLine = mlContext.Transforms.Concatenate("Features", remainingColumns).Append(mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: clusterCount));

                // !!! Kryžminė patikra su KMeans kažkodėl neveikia

                // Evaluate the model using cross-validation
                // Cross-validation splits our dataset into 'folds', trains a model on some folds and evaluates it on the remaining fold
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
                var crossValidationResults = mlContext.Clustering.CrossValidate(data: newTrainingDataView, estimator: newPipeLine, numberOfFolds: foldCount);
                IEnumerable<double> accuracy = crossValidationResults.Select(fold => fold.Metrics.AverageDistance);
                count = 0;
                crossValidationSum = 0;
                crossValidationAverage = 0;
                foreach (var acc in accuracy)
                {
                    count++;
                    crossValidationSum += acc;
                    Console.WriteLine("Accuracy (k = {0}): {1} = {2} %", count, acc, acc * 100);
                }
                crossValidationAverage = crossValidationSum / accuracy.Count();
                Console.WriteLine("Cross validation average: {0} = {1} %", crossValidationAverage, crossValidationAverage * 100);

                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Training the model ===============");
                ITransformer trainedModel = newPipeLine.Fit(newTrainingDataView);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Finish the train model. Push Enter ===============");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
                var predictions = trainedModel.Transform(newTestDataView);

                var metrics = mlContext.Clustering.Evaluate(predictions, scoreColumnName: "Score", featureColumnName: "Features");
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine($"************************************************************");
                Console.WriteLine($"*       Metrics for KMeans()      ");
                Console.WriteLine($"*-----------------------------------------------------------");
                Console.WriteLine($"*       AverageDistance: {metrics.AverageDistance}");
                Console.WriteLine($"*       DaviesBouldinIndex:      {metrics.DaviesBouldinIndex}");
                Console.WriteLine($"*       NormalizedMutualInformation:  {metrics.NormalizedMutualInformation}");
                Console.WriteLine($"************************************************************");
                Console.WriteLine("");
                Console.WriteLine("");

                Console.WriteLine("=============== Saving the model to a file ===============");
                mlContext.Model.Save(trainedModel, newTrainingDataView.Schema, ModelPath);
                Console.WriteLine("");
                Console.WriteLine("");
                Console.WriteLine("=============== Model Saved ============= ");
                Console.WriteLine("");
                Console.WriteLine("");
            }
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

        private static void BuildTrainEvaluateAndSaveModelGam(MLContext mlContext)
        {
            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataPath, hasHeader: true, separatorChar: ';');
            var testDataView = mlContext.Data.LoadFromTextFile<HeartData>(TestDataPath, hasHeader: true, separatorChar: ';');

            // STEP 2: Concatenate the features and set the training algorithm
            var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
                .Append(mlContext.BinaryClassification.Trainers.Gam(labelColumnName: "Label", featureColumnName: "Features"));

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

        private static void BuildTrainEvaluateAndSaveModelLbfgs(MLContext mlContext)
        {
            // STEP 1: Common data loading configuration
            var trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataPath, hasHeader: true, separatorChar: ';');
            var testDataView = mlContext.Data.LoadFromTextFile<HeartData>(TestDataPath, hasHeader: true, separatorChar: ';');

            // STEP 2: Concatenate the features and set the training algorithm
            var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

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

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;

        }
    }
}
