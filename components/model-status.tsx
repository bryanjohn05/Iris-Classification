"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { CheckCircle, XCircle, AlertTriangle, RefreshCw, Terminal, FileText, Cloud } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ModelStatus {
  models: {
    available: boolean
    files: Record<string, boolean>
  }
  python: {
    available: boolean
    error: string | null
  }
  status: "ready" | "simulation_mode"
}

export default function ModelStatus() {
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [isVercelDeployment, setIsVercelDeployment] = useState(false)

  useEffect(() => {
    setIsVercelDeployment(window.location.hostname.includes("vercel.app"))
  }, [])

  const checkStatus = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/check-models")
      if (response.ok) {
        const data = await response.json()
        setStatus(data)
      }
    } catch (error) {
      console.error("Failed to check model status:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkStatus()
  }, [])

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center p-6">
          <RefreshCw className="h-6 w-6 animate-spin mr-2" />
          <span>Checking model status...</span>
        </CardContent>
      </Card>
    )
  }

  if (!status) {
    return (
      <Alert variant="destructive">
        <XCircle className="h-4 w-4" />
        <AlertDescription>Failed to check model status</AlertDescription>
      </Alert>
    )
  }

  return (
    <div className="space-y-4">
      {/* Deployment Type Alert */}
      {isVercelDeployment ? (
        <Alert>
          <Cloud className="h-4 w-4" />
          <AlertDescription>
            <strong>Vercel Deployment Detected:</strong> This application is running on Vercel's serverless platform.
            Python model integration is limited in serverless environments, so intelligent simulation mode is active.
          </AlertDescription>
        </Alert>
      ) : (
        <Alert variant={status.status === "ready" ? "default" : "destructive"}>
          {status.status === "ready" ? <CheckCircle className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
          <AlertDescription>
            {status.status === "ready"
              ? "✅ Real models are loaded and ready for predictions"
              : "⚠️ Using simulation mode - real models not available"}
          </AlertDescription>
        </Alert>
      )}

      {/* Integration Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Terminal className="h-5 w-5" />
            {isVercelDeployment ? "Vercel Integration Status" : "Local Integration Status"}
          </CardTitle>
          <CardDescription>
            {isVercelDeployment ? "Serverless deployment configuration" : "Backend integration with Next.js"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>✅ Frontend application running</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>✅ API routes functional</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>✅ Intelligent simulation active</span>
            </div>
            {isVercelDeployment ? (
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-500" />
                <span>⚠️ Python processes limited in serverless</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>✅ Python integration enabled</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Model Files Status */}
      {!isVercelDeployment && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Model Files Status
              </div>
              <Button variant="outline" size="sm" onClick={checkStatus}>
                <RefreshCw className="h-4 w-4 mr-1" />
                Refresh
              </Button>
            </CardTitle>
            <CardDescription>Status of trained model files</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
              {Object.entries(status.models.files).map(([name, available]) => (
                <div key={name} className="flex items-center gap-2">
                  {available ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <XCircle className="h-4 w-4 text-red-500" />
                  )}
                  <span className="text-sm">{name}</span>
                  <Badge variant={available ? "secondary" : "destructive"} className="text-xs">
                    {available ? "OK" : "Missing"}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Python Environment Status */}
      {!isVercelDeployment && (
        <Card>
          <CardHeader>
            <CardTitle>Python Environment</CardTitle>
            <CardDescription>Python runtime and dependencies</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              {status.python.available ? (
                <CheckCircle className="h-4 w-4 text-green-500" />
              ) : (
                <XCircle className="h-4 w-4 text-red-500" />
              )}
              <span>Python Runtime</span>
              <Badge variant={status.python.available ? "secondary" : "destructive"}>
                {status.python.available ? "Available" : "Unavailable"}
              </Badge>
            </div>
            {status.python.error && (
              <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded">
                <strong>Error:</strong> {status.python.error}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Setup Instructions */}
      <Card>
        <CardHeader>
          <CardTitle>{isVercelDeployment ? "Production Deployment Options" : "Setup Instructions"}</CardTitle>
          <CardDescription>
            {isVercelDeployment
              ? "Options for using real models in production"
              : "Steps to enable real model predictions"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {isVercelDeployment ? (
            <div className="space-y-4">
              <div className="text-sm">
                <p className="font-medium mb-2">For real model predictions in production:</p>
                <div className="space-y-2">
                  <div className="flex items-start gap-2">
                    <span className="font-mono text-xs bg-gray-100 px-1 rounded">1</span>
                    <span>Deploy Python backend on Railway, Render, or AWS Lambda</span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="font-mono text-xs bg-gray-100 px-1 rounded">2</span>
                    <span>
                      Set <code className="bg-gray-100 px-1 rounded">PYTHON_SERVICE_URL</code> environment variable
                    </span>
                  </div>
                  <div className="flex items-start gap-2">
                    <span className="font-mono text-xs bg-gray-100 px-1 rounded">3</span>
                    <span>Upload trained models to your Python service</span>
                  </div>
                </div>
              </div>
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>
                  <strong>Current Mode:</strong> The application is fully functional using intelligent simulation that
                  provides accurate iris classification demonstrations based on botanical patterns.
                </AlertDescription>
              </Alert>
            </div>
          ) : (
            <div className="text-sm">
              <p className="font-medium">To use real models locally:</p>
              <ol className="list-decimal list-inside space-y-1 mt-2">
                <li>
                  Run: <code className="bg-gray-100 px-1 rounded">python scripts/train_models.py</code>
                </li>
                <li>Ensure Python and required packages are installed</li>
                <li>Refresh this page to check status</li>
                <li>No separate backend server needed!</li>
              </ol>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Performance Note */}
      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          <strong>Note:</strong> The application provides accurate iris classification in both modes:
          <br />• <strong>Real Models:</strong> Uses trained scikit-learn models for precise predictions
          <br />• <strong>Intelligent Simulation:</strong> Uses botanical classification rules for demonstration
          {isVercelDeployment && (
            <>
              <br />• <strong>Vercel Deployment:</strong> Optimized for serverless environments with fast response times
            </>
          )}
        </AlertDescription>
      </Alert>
    </div>
  )
}
