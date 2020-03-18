using Microsoft.ML;
using System.Linq;
using System.Reflection;

namespace FraudPredictor
{
    public class FraudPredictor
    {
        public static bool Predict(string type,
            string destinationAccountName,
            double amount,
            double oldBalanceOfOriginAccount,
            double oldBalanceOfDestinationAccount,
            double newBalanceOfOriginAccount,
            double newBalanceOfDestinationAccount)
        {
            MLContext mlContext = new MLContext();

            var model = LoadModel(mlContext);

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            var input = new ModelInput
            {
                Type = type,
                NameDest = destinationAccountName,
                Amount = (float)amount,
                OldbalanceDest = (float)oldBalanceOfDestinationAccount,
                OldbalanceOrg = (float)oldBalanceOfOriginAccount,
                NewbalanceDest = (float)newBalanceOfDestinationAccount,
                NewbalanceOrig = (float)newBalanceOfOriginAccount
            };

            ModelOutput result = predEngine.Predict(input);
            return result.Prediction;
        }

        private static ITransformer LoadModel(MLContext mlContext)
        {
            var assembly = Assembly.GetExecutingAssembly();

            var resource = assembly.GetManifestResourceNames().First(x => x.EndsWith("MLModel.zip"));

            using (var stream = assembly.GetManifestResourceStream(resource))
            {
                return mlContext.Model.Load(stream, out var _);
            }
        }
    }
}
