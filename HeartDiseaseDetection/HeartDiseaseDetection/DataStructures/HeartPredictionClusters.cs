using Microsoft.ML.Data;

namespace HeartDiseasePredictionConsoleApp.DataStructures
{
    public class HeartPredictionClusters
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;


        [ColumnName("Score")]
        public float[] Probability;
    }
}
