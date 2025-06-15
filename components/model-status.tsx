"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { CheckCircle, XCircle, AlertTriangle, RefreshCw, Terminal, FileText } from "lucide-react"
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
      {/* Overall Status */}
      <Alert variant={status.status === "ready" ? "default" : "destructive"}>
        {status.status === "ready" ? <CheckCircle className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
        <AlertDescription>
          {status.status === "ready"
            ? "✅ Real models are loaded and ready for predictions"
            : "⚠️ Using simulated predictions - real models not available"}
        </AlertDescription>
      </Alert>

      {/* Integration Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Terminal className="h-5 w-5" />
            Integration Status
          </CardTitle>
          <CardDescription>Backend integration with Next.js</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>✅ Python integration enabled</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>✅ Single port operation (no separate backend)</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>✅ Automatic fallback to simulation</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>✅ Real-time model status checking</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Files Status */}
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

      {/* Python Environment Status */}
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

      {/* Setup Instructions */}
      {status.status !== "ready" && (
        <Card>
          <CardHeader>
            <CardTitle>Setup Instructions</CardTitle>
            <CardDescription>Steps to enable real model predictions</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-sm">
              <p className="font-medium">To use real models:</p>
              <ol className="list-decimal list-inside space-y-1 mt-2">
                <li>
                  Run: <code className="bg-gray-100 px-1 rounded">python scripts/train_models.py</code>
                </li>
                <li>Ensure Python and required packages are installed</li>
                <li>Refresh this page to check status</li>
                <li>No separate backend server needed!</li>
              </ol>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Note */}
      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          <strong>Note:</strong> The application works in both modes:
          <br />• <strong>Real Models:</strong> Uses your trained joblib models for accurate predictions
          <br />• <strong>Simulation Mode:</strong> Uses rule-based predictions for demonstration
        </AlertDescription>
      </Alert>
    </div>
  )
}
