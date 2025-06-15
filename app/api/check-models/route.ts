import { NextResponse } from "next/server"
import { spawn } from "child_process"
import path from "path"
import fs from "fs"

function checkModelFiles(): { available: boolean; files: Record<string, boolean> } {
  const modelFiles = {
    scaler: "models/scaler.joblib",
    "Random Forest": "models/RandomForest_model.joblib",
    SVM: "models/SVM_model.joblib",
    KNN: "models/KNN_model.joblib",
    "Decision Tree": "models/DecisionTree_model.joblib",
    "Logistic Regression": "models/LogisticRegression_model.joblib",
  }

  const fileStatus: Record<string, boolean> = {}
  let allAvailable = true

  for (const [name, filePath] of Object.entries(modelFiles)) {
    const exists = fs.existsSync(filePath)
    fileStatus[name] = exists
    if (!exists) allAvailable = false
  }

  return { available: allAvailable, files: fileStatus }
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

export async function GET() {
  try {
    // Check if model files exist
    const modelStatus = checkModelFiles()

    // Try to test Python environment
    let pythonAvailable = false
    let pythonError = ""

    try {
      await runPythonScript("model_predictor.py", ["test"])
      pythonAvailable = true
    } catch (error) {
      pythonError = error instanceof Error ? error.message : "Unknown error"
    }

    return NextResponse.json({
      models: modelStatus,
      python: {
        available: pythonAvailable,
        error: pythonError || null,
      },
      status: modelStatus.available && pythonAvailable ? "ready" : "simulation_mode",
    })
  } catch (error) {
    console.error("Model check error:", error)
    return NextResponse.json({ error: "Failed to check model status" }, { status: 500 })
  }
}
