"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Flower2, BarChart3, Target, TrendingUp, AlertCircle, Settings } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import ModelComparison from "@/components/model-comparison"
import PredictionResults from "@/components/prediction-results"
import ModelStatus from "@/components/model-status"
import Link from "next/link"

interface PredictionInput {
  sepalLength: number
  sepalWidth: number
  petalLength: number
  petalWidth: number
}

interface ModelPrediction {
  model: string
  prediction: string
  confidence: number
  probabilities: Record<string, number>
  warning?: string
}

export default function IrisClassifier() {
  const [input, setInput] = useState<PredictionInput>({
    sepalLength: 5.1,
    sepalWidth: 3.5,
    petalLength: 1.4,
    petalWidth: 0.2,
  })

  const [selectedModel, setSelectedModel] = useState<string>("Random Forest")
  const [predictions, setPredictions] = useState<ModelPrediction[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const models = ["Random Forest", "SVM", "KNN", "Decision Tree", "Logistic Regression"]

  const handlePredict = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch("/api/predict-all", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sepal_length: input.sepalLength,
          sepal_width: input.sepalWidth,
          petal_length: input.petalLength,
          petal_width: input.petalWidth,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setPredictions(data.predictions)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during prediction")
      console.error("Prediction error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const handlePredictSingle = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: selectedModel,
          sepal_length: input.sepalLength,
          sepal_width: input.sepalWidth,
          petal_length: input.petalLength,
          petal_width: input.petalWidth,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setPredictions([data])
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred during prediction")
      console.error("Prediction error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-2">
            <Flower2 className="h-8 w-8 text-green-600" />
            <h1 className="text-4xl font-bold text-gray-900">Iris Species Classifier</h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Classify iris flowers using trained machine learning models
          </p>
          <div className="hover:text-blue-600">
            <Link href="https://github.com/bryanjohn05/Iris-Classification/blob/main/dataset%20and%20Phase1/Iris.csv">View Dataset</Link>
          </div>
        </div>

        <Tabs defaultValue="classify" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="classify" className="flex items-center gap-2">
              <Target className="h-4 w-4" />
              Classify
            </TabsTrigger>
            <TabsTrigger value="compare" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Compare Models
            </TabsTrigger>
            {/* <TabsTrigger value="performance" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="status" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Status
            </TabsTrigger> */}
          </TabsList>

          <TabsContent value="classify" className="space-y-6">
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Form */}
              <Card>
                <CardHeader>
                  <CardTitle>Flower Measurements</CardTitle>
                  <CardDescription>Enter the measurements of the iris flower in centimeters</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="sepal-length">Sepal Length (cm)</Label>
                      <Input
                        id="sepal-length"
                        type="number"
                        step="0.1"
                        min="4.0"
                        max="8.0"
                        value={input.sepalLength}
                        onChange={(e) =>
                          setInput((prev) => ({ ...prev, sepalLength: Number.parseFloat(e.target.value) || 0 }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="sepal-width">Sepal Width (cm)</Label>
                      <Input
                        id="sepal-width"
                        type="number"
                        step="0.1"
                        min="2.0"
                        max="4.5"
                        value={input.sepalWidth}
                        onChange={(e) =>
                          setInput((prev) => ({ ...prev, sepalWidth: Number.parseFloat(e.target.value) || 0 }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="petal-length">Petal Length (cm)</Label>
                      <Input
                        id="petal-length"
                        type="number"
                        step="0.1"
                        min="1.0"
                        max="7.0"
                        value={input.petalLength}
                        onChange={(e) =>
                          setInput((prev) => ({ ...prev, petalLength: Number.parseFloat(e.target.value) || 0 }))
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="petal-width">Petal Width (cm)</Label>
                      <Input
                        id="petal-width"
                        type="number"
                        step="0.1"
                        min="0.1"
                        max="2.5"
                        value={input.petalWidth}
                        onChange={(e) =>
                          setInput((prev) => ({ ...prev, petalWidth: Number.parseFloat(e.target.value) || 0 }))
                        }
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label>Model Selection</Label>
                      <Select value={selectedModel} onValueChange={setSelectedModel}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {models.map((model) => (
                            <SelectItem key={model} value={model}>
                              {model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex gap-2">
                      <Button onClick={handlePredictSingle} disabled={isLoading} className="flex-1">
                        {isLoading ? "Predicting..." : `Predict with ${selectedModel}`}
                      </Button>
                      <Button onClick={handlePredict} disabled={isLoading} variant="outline" className="flex-1">
                        {isLoading ? "Comparing..." : "Compare All Models"}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Results */}
              <PredictionResults predictions={predictions} />
            </div>
          </TabsContent>

          <TabsContent value="compare">
            <ModelComparison />
          </TabsContent>

          <TabsContent value="performance">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {models.map((model, index) => {
                const accuracies = [0.97, 0.95, 0.93, 0.91, 0.94]
                const precisions = [0.96, 0.94, 0.92, 0.9, 0.93]
                const recalls = [0.97, 0.95, 0.93, 0.91, 0.94]

                return (
                  <Card key={model}>
                    <CardHeader>
                      <CardTitle className="text-lg">{model}</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Accuracy</span>
                        <Badge variant="secondary">{(accuracies[index] * 100).toFixed(1)}%</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Precision</span>
                        <Badge variant="secondary">{(precisions[index] * 100).toFixed(1)}%</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Recall</span>
                        <Badge variant="secondary">{(recalls[index] * 100).toFixed(1)}%</Badge>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </TabsContent>

          <TabsContent value="status">
            <ModelStatus />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
