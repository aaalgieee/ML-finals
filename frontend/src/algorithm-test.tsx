import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Button } from "./components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Label } from "./components/ui/label"
import { Brain, Activity, LineChart, Network, TreeDeciduous, ArrowLeft } from 'lucide-react'

const algorithms = [
  { id: 'linear-regression', name: 'Linear Regression', icon: LineChart, description: 'Predicting patient recovery times', inputs: ['Age', 'Treatment Duration', 'Severity Score'] },
  { id: 'naive-bayes', name: 'Naive Bayes', icon: Brain, description: 'Classifying patient symptoms for diagnosis', inputs: ['Symptom 1', 'Symptom 2', 'Symptom 3'] },
  { id: 'knn', name: 'K-Nearest Neighbors', icon: Network, description: 'Identifying similar patient profiles', inputs: ['Age', 'BMI', 'Blood Pressure'] },
  { id: 'svm', name: 'Support Vector Machine', icon: Activity, description: 'Classifying medical images', inputs: ['Image URL'] },
  { id: 'decision-tree', name: 'Decision Tree', icon: TreeDeciduous, description: 'Guiding treatment decisions', inputs: ['Age', 'Gender', 'Condition Severity'] },
  { id: 'ann', name: 'Artificial Neural Network', icon: Brain, description: 'Predicting disease outbreaks', inputs: ['Location', 'Season', 'Population Density'] },
]

export default function AlgorithmTest() {
  const { id } = useParams<{ id: string }>()
  const [algorithm, setAlgorithm] = useState<typeof algorithms[0] | undefined>()
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [result, setResult] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const foundAlgorithm = algorithms.find(algo => algo.id === id)
    setAlgorithm(foundAlgorithm)
    if (foundAlgorithm) {
      const initialInputs = foundAlgorithm.inputs.reduce((acc, input) => ({ ...acc, [input]: '' }), {})
      setInputs(initialInputs)
    }
  }, [id])

  const handleInputChange = (name: string, value: string) => {
    setInputs(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000))
    setResult(`${algorithm?.name} prediction: ${Math.random().toFixed(2)}`)
    setIsLoading(false)
  }

  if (!algorithm) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-b from-blue-50 to-blue-100">
        <div className="text-center p-8 rounded-lg bg-white shadow-lg">
          <p className="text-lg text-gray-800 font-medium">Algorithm not found</p>
          <Link to="/" className="mt-4 text-blue-600 hover:text-blue-700 inline-flex items-center">
            <ArrowLeft className="h-4 w-4 mr-2" />
            Return home
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-blue-100 py-8">
      <div className="container mx-auto px-6 max-w-2xl">
        <Link 
          to="/" 
          className="group inline-flex items-center text-blue-600 hover:text-blue-700 mb-6 transition-colors font-medium"
        >
          <ArrowLeft className="h-4 w-4 mr-2 transform group-hover:-translate-x-1 transition-transform duration-200" />
          Back to Home
        </Link>
        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center text-2xl text-gray-900">
              <algorithm.icon className="h-8 w-8 mr-3 text-blue-600" />
              {algorithm.name}
            </CardTitle>
            <CardDescription className="text-gray-600 mt-1">{algorithm.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {algorithm.inputs.map((input) => (
                <div key={input} className="space-y-2">
                  <Label htmlFor={input} className="text-gray-700">{input}</Label>
                  <Input
                    id={input}
                    value={inputs[input]}
                    onChange={(e) => handleInputChange(input, e.target.value)}
                    required
                    className="bg-gray-50 hover:bg-gray-100 focus:bg-white"
                    placeholder={`Enter ${input.toLowerCase()}`}
                  />
                </div>
              ))}
              <Button 
                type="submit" 
                className={`w-full transition-all duration-200 ${
                  isLoading ? 'bg-blue-500' : 'bg-blue-600 hover:bg-blue-700'
                }`}
                disabled={isLoading}
              >
                {isLoading ? (
                  <span className="flex items-center justify-center space-x-2">
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Processing...</span>
                  </span>
                ) : (
                  <span className="flex items-center justify-center space-x-2">
                    <span>Run Algorithm</span>
                  </span>
                )}
              </Button>
            </form>
            {result && (
              <div className="mt-6 p-4 bg-blue-50 rounded-md">
                <h3 className="font-semibold text-lg mb-2 text-gray-900">Result</h3>
                <p className="text-gray-800 font-medium">{result}</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}