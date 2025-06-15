import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Flower, Award, Target, AlertTriangle, Cloud, Server } from "lucide-react"

interface ModelPrediction {
  model: string
  prediction: string
  confidence: number
  probabilities: Record<string, number>
  warning?: string
}

interface PredictionResultsProps {
  predictions: ModelPrediction[]
}

export default function PredictionResults({ predictions }: PredictionResultsProps) {
  if (predictions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Prediction Results</CardTitle>
          <CardDescription>Results will appear here after prediction</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-48 text-gray-500">
          <div className="text-center space-y-2">
            <Flower className="h-12 w-12 mx-auto opacity-50" />
            <p>No predictions yet</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const getSpeciesColor = (species: string) => {
    switch (species) {
      case "Iris-setosa":
        return "bg-green-100 text-green-800 border-green-200"
      case "Iris-versicolor":
        return "bg-blue-100 text-blue-800 border-blue-200"
      case "Iris-virginica":
        return "bg-purple-100 text-purple-800 border-purple-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  const getSpeciesIcon = (species: string) => {
    switch (species) {
      case "Iris-setosa":
        return "🌸"
      case "Iris-versicolor":
        return "🌺"
      case "Iris-virginica":
        return "🌻"
      default:
        return "🌼"
    }
  }

  const hasWarnings = predictions.some((pred) => pred.warning)
  const isVercelDeployment = typeof window !== "undefined" && window.location.hostname.includes("vercel.app")

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Target className="h-5 w-5" />
          Prediction Results
        </CardTitle>
        <CardDescription>
          {predictions.length === 1 ? "Single model prediction" : `Comparison of ${predictions.length} models`}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {hasWarnings && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              {isVercelDeployment ? (
                <>
                  <Cloud className="h-4 w-4 inline mr-1" />
                  <strong>Vercel Deployment:</strong> Using intelligent simulation mode. Real models require a dedicated
                  Python backend service. The predictions are based on iris classification patterns and provide accurate
                  demonstrations.
                </>
              ) : (
                <>
                  <Server className="h-4 w-4 inline mr-1" />
                  <strong>Local Development:</strong> Python models not available. Using simulation mode for
                  demonstration. Run <code className="bg-red-100 px-1 rounded">python scripts/train_models.py</code> to
                  enable real models.
                </>
              )}
            </AlertDescription>
          </Alert>
        )}

        {predictions.map((pred, index) => (
          <div key={index} className="border rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Award className="h-4 w-4 text-gray-500" />
                <span className="font-medium">{pred.model}</span>
                {pred.warning && (
                  <Badge variant="outline" className="text-xs">
                    {isVercelDeployment ? "Intelligent Simulation" : "Simulated"}
                  </Badge>
                )}
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-2xl">{getSpeciesIcon(pred.prediction)}</span>
              <div className="flex-1">
                <Badge className={getSpeciesColor(pred.prediction)}>{pred.prediction}</Badge>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Confidence</span>
                <span className="font-medium">{(pred.confidence * 100).toFixed(1)}%</span>
              </div>
              <Progress value={pred.confidence * 100} className="h-2" />
            </div>

            {/* Show all class probabilities */}
            {pred.probabilities && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700">Class Probabilities:</h4>
                {Object.entries(pred.probabilities).map(([species, prob]) => (
                  <div key={species} className="flex justify-between items-center text-sm">
                    <span className="flex items-center gap-1">
                      <span className="text-lg">{getSpeciesIcon(species)}</span>
                      {species}
                    </span>
                    <div className="flex items-center gap-2">
                      <Progress value={prob * 100} className="h-1 w-16" />
                      <span className="font-medium w-12 text-right">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {predictions.length > 1 && (
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-blue-900 mb-2">Consensus</h4>
            <p className="text-sm text-blue-800">
              {(() => {
                const predictionCounts = predictions.reduce(
                  (acc, pred) => {
                    acc[pred.prediction] = (acc[pred.prediction] || 0) + 1
                    return acc
                  },
                  {} as Record<string, number>,
                )

                const mostCommon = Object.entries(predictionCounts).sort(([, a], [, b]) => b - a)[0]

                return `${mostCommon[1]} out of ${predictions.length} models predict: ${mostCommon[0]}`
              })()}
            </p>
          </div>
        )}

        {/* Deployment Info */}
        {isVercelDeployment && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-center gap-2 text-blue-800">
              <Cloud className="h-4 w-4" />
              <span className="font-medium text-sm">Vercel Deployment</span>
            </div>
            <p className="text-xs text-blue-700 mt-1">
              This demo uses intelligent simulation based on iris classification patterns. For production use with real
              models, consider deploying the Python backend separately.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
