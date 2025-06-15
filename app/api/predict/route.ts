import { type NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import path from "path"

interface PredictionRequest {
  model: string
  sepal_length: number
  sepal_width: number
  petal_length: number
  petal_width: number
}

function runPythonScript(scriptName: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(process.cwd(), "scripts", scriptName)
    const pythonProcess = spawn("python", [scriptPath, ...args])

    let output = ""
    let errorOutput = ""

    pythonProcess.stdout.on("data", (data) => {
      output += data.toString()
    })

    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString()
    })

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        resolve(output.trim())
      } else {
        reject(new Error(`Python script failed: ${errorOutput}`))
      }
    })

    pythonProcess.on("error", (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`))
    })
  })
}

function simulateModelPrediction(input: PredictionRequest): any {
  const { sepal_length, sepal_width, petal_length, petal_width, model } = input

  let prediction = "Iris-setosa"
  let confidence = 0.85

  if (petal_length > 2.5) {
    if (petal_width > 1.7) {
      prediction = "Iris-virginica"
      confidence = 0.92
    } else {
      prediction = "Iris-versicolor"
      confidence = 0.88
    }
  }

  // Model-specific adjustments
  const modelAccuracies = {
    "Random Forest": 0.97,
    SVM: 0.95,
    KNN: 0.93,
    "Decision Tree": 0.91,
    "Logistic Regression": 0.94,
  }

  if (model === "Decision Tree" && petal_length < 2.0) confidence += 0.05
  if (model === "SVM" && prediction === "Iris-virginica") confidence += 0.03
  if (model === "KNN" && petal_width > 1.5) confidence -= 0.02

  confidence = Math.min(0.99, Math.max(0.75, confidence))

  const probabilities: Record<string, number> = {
    "Iris-setosa": 0.1,
    "Iris-versicolor": 0.1,
    "Iris-virginica": 0.1,
  }
  probabilities[prediction] = confidence

  const remaining = 1 - confidence
  const otherSpecies = Object.keys(probabilities).filter((s) => s !== prediction)
  otherSpecies.forEach((species) => {
    probabilities[species] = remaining / otherSpecies.length
  })

  return {
    model,
    prediction,
    confidence,
    probabilities,
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { model, sepal_length, sepal_width, petal_length, petal_width } = body

    // Validate input
    if (
      !model ||
      typeof sepal_length !== "number" ||
      typeof sepal_width !== "number" ||
      typeof petal_length !== "number" ||
      typeof petal_width !== "number"
    ) {
      return NextResponse.json({ error: "Invalid input parameters" }, { status: 400 })
    }

    try {
      // Try to use Python script for real model prediction
      const args = [
        "predict-single",
        model,
        sepal_length.toString(),
        sepal_width.toString(),
        petal_length.toString(),
        petal_width.toString(),
      ]

      const result = await runPythonScript("model_predictor.py", args)
      const prediction = JSON.parse(result)

      return NextResponse.json(prediction)
    } catch (pythonError) {
      console.log("Python prediction failed, using simulation:", pythonError)

      // Fallback to simulation
      const simulatedPrediction = simulateModelPrediction({
        model,
        sepal_length,
        sepal_width,
        petal_length,
        petal_width,
      })

      return NextResponse.json({
        ...simulatedPrediction,
        warning: "Using simulated prediction - Python models unavailable",
      })
    }
  } catch (error) {
    console.error("Prediction API error:", error)
    return NextResponse.json({ error: "Failed to make prediction" }, { status: 500 })
  }
}
