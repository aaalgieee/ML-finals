import { useState, useEffect, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import { Button } from "./components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Label } from "./components/ui/label"
import { Brain, Activity, LineChart, Network, TreeDeciduous, ArrowLeft } from 'lucide-react'
import { useDropzone } from 'react-dropzone'
import { Scatter } from 'react-chartjs-2'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const algorithms = [
  { id: 'linear-regression', name: 'Linear Regression', icon: LineChart, description: 'Predicting patient recovery times', inputs: ['Age', 'Treatment Duration', 'Severity Score'] },
  { id: 'naive-bayes', name: 'Naive Bayes', icon: Brain, description: 'Classifying patient symptoms for diagnosis', inputs: ['Symptom 1', 'Symptom 2', 'Symptom 3'] },
  { id: 'knn', name: 'K-Nearest Neighbors', icon: Network, description: 'Identifying similar patient profiles', inputs: ['Gender', 'Age', 'Hypertension', 'Heart Disease', 'Smoking History', 'BMI', 'HbA1c Level', 'Blood Glucose Level'] },
  { id: 'svm', name: 'Support Vector Machine', icon: Activity, description: 'Classifying medical images', inputs: ['Image URL'] },
  { id: 'decision-tree', name: 'Decision Tree', icon: TreeDeciduous, description: 'Guiding treatment decisions', inputs: ['Age', 'Gender', 'Condition Severity'] },
  { id: 'ann', name: 'Artificial Neural Network', icon: Brain, description: 'Predicting disease outbreaks', inputs: ['Location', 'Season', 'Population Density'] },
]

type PredictionResponse = {
  success: boolean;
  prediction: number;
  message: string;
  raw_prediction: string;
}

export default function AlgorithmTest() {
  const { id } = useParams<{ id: string }>()
  const [algorithm, setAlgorithm] = useState<typeof algorithms[0] | undefined>()
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [result, setResult] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [scatterData, setScatterData] = useState<any>(null)

  useEffect(() => {
    const foundAlgorithm = algorithms.find(algo => algo.id === id)
    setAlgorithm(foundAlgorithm)
    if (foundAlgorithm) {
      const initialInputs = foundAlgorithm.inputs.reduce((acc, input) => ({ ...acc, [input]: '' }), {})
      setInputs(initialInputs)
    }
  }, [id])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    setImageFile(file)
    setImagePreview(URL.createObjectURL(file))
    setResult(null)  // Reset result when new image is uploaded
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    multiple: false,
    onDragEnter: () => {},
    onDragLeave: () => {},
    onDragOver: () => {}
  })

  const handleInputChange = (name: string, value: string) => {
    setInputs(prev => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    
    if (algorithm?.id === 'svm' && imageFile) {
      try {
        const formData = new FormData()
        formData.append('image', imageFile)

        console.log('Sending file:', imageFile.name, imageFile.type)
        
        const response = await fetch('http://127.0.0.1:5000/api/svm/predict', {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data: PredictionResponse = await response.json()
        console.log('Response:', data)
        
        setResult(`Classification: ${data.message}`)
      } catch (error) {
        console.error('Upload error:', error)
        setResult(`Error: ${error instanceof Error ? error.message : 'Error processing image'}`)
      } finally {
        setIsLoading(false)
      }
    } else if (algorithm?.id === 'knn') {
      try {
        const transformedInputs = {
          gender: inputs['Gender'],
          age: Number(inputs['Age']),
          hypertension: inputs['Hypertension'] === 'true' ? 1 : 0,
          heart_disease: inputs['Heart Disease'] === 'true' ? 1 : 0,
          smoking_history: inputs['Smoking History'],
          bmi: parseFloat(inputs['BMI']),
          HbA1c_level: parseFloat(inputs['HbA1c Level']),
          blood_glucose_level: parseInt(inputs['Blood Glucose Level'], 10)
        }

        console.log('Sending data:', transformedInputs);

        const response = await fetch('http://127.0.0.1:5000/api/knn/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ features: transformedInputs })
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        console.log('Received data:', data);

        if (data.prediction) {
          setResult(`Prediction: ${data.prediction} (Risk of Diabetes)`)
          
          if (data.scatter_plot_data) {
            const scatterPlotData = {
              datasets: [
                {
                  label: 'Patient Data',
                  data: data.scatter_plot_data || [],
                  backgroundColor: data.prediction === 'Diabetic' ? 'red' : 'green',
                  pointRadius: 8,
                  pointStyle: 'circle',
                  borderColor: 'black',
                  borderWidth: 1
                }
              ],
              options: {
                plugins: {
                  title: {
                    display: true,
                    text: 'Patient Classification'
                  },
                  legend: {
                    display: true,
                    position: 'top'
                  }
                },
                scales: {
                  x: { 
                    title: { display: true, text: 'Age' },
                    grid: { display: true }
                  },
                  y: { 
                    title: { display: true, text: 'BMI' },
                    grid: { display: true }
                  }
                }
              }
            }
            setScatterData(scatterPlotData)
          }
        } else {
          setResult('Error: Could not generate prediction')
        }
      } catch (error) {
        console.error('Prediction error:', error)
        setResult(`Error: ${error instanceof Error ? error.message : 'Error processing prediction'}`)
      } finally {
        setIsLoading(false)
      }
    } else {
      await new Promise(resolve => setTimeout(resolve, 1000))
      setResult(`${algorithm?.name} prediction: ${Math.random().toFixed(2)}`)
      setIsLoading(false)
    }
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
              {algorithm.id === 'svm' ? (
                <div className="space-y-4">
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors
                      ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-500'}`}
                  >
                    <input {...getInputProps()} type="file" style={{ display: 'none' }} />
                    <div className="space-y-2">
                      <div className="flex justify-center">
                        <svg className="h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                          <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4-4m4-12h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                      </div>
                      <div className="text-gray-600">
                        <span className="font-medium">Click to upload</span> or drag and drop
                        <p className="text-xs">PNG, JPG, GIF up to 10MB</p>
                      </div>
                    </div>
                  </div>
                  
                  {imagePreview && (
                    <div className="mt-4 relative">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="max-h-64 mx-auto rounded-lg shadow-md"
                      />
                      <button
                        type="button"
                        onClick={() => {
                          setImageFile(null)
                          setImagePreview(null)
                        }}
                        className="absolute top-2 right-2 bg-red-500 text-white p-1 rounded-full hover:bg-red-600"
                      >
                        <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {algorithm.inputs.map((input) => (
                    <div key={input} className="space-y-2">
                      <Label htmlFor={input} className="text-gray-700">{input}</Label>
                      {['Gender', 'Hypertension', 'Heart Disease', 'Smoking History'].includes(input) ? (
                        <select
                          id={input}
                          value={inputs[input]}
                          onChange={(e) => handleInputChange(input, e.target.value)}
                          required
                          className="bg-gray-50 hover:bg-gray-100 focus:bg-white w-full p-2 border border-gray-300 rounded-md"
                        >
                          {input === 'Gender' && (
                            <>
                              <option value="">Select Gender</option>
                              <option value="Male">Male</option>
                              <option value="Female">Female</option>
                            </>
                          )}
                          {input === 'Hypertension' && (
                            <>
                              <option value="">Have Hypertension?</option>
                              <option value="true">Yes</option>
                              <option value="false">No</option>
                            </>
                          )}
                          {input === 'Heart Disease' && (
                            <>
                              <option value="">Have Heart Disease?</option>
                              <option value="true">Yes</option>
                              <option value="false">No</option>
                            </>
                          )}
                          {input === 'Smoking History' && (
                            <>
                              <option value="">Smoking History</option>
                              <option value="never">never</option>
                              <option value="former">former</option>
                              <option value="current">current</option>
                              <option value="not current">not current</option>
                              <option value="No Info">No Info</option>  {/* Changed from "no info" to "No Info" */}
                              <option value="ever">ever</option>
                            </>
                          )}
                        </select>
                      ) : (
                        <Input
                          id={input}
                          value={inputs[input]}
                          onChange={(e) => handleInputChange(input, e.target.value)}
                          required
                          className="bg-gray-50 hover:bg-gray-100 focus:bg-white"
                          placeholder={`Enter ${input.toLowerCase()}`}
                        />
                      )}
                    </div>
                  ))}
                </>
              )}
              <Button 
                type="submit" 
                className={`w-full transition-all duration-200 ${
                  isLoading ? 'bg-blue-500' : 'bg-blue-600 hover:bg-blue-700'
                }`}
                disabled={isLoading || (algorithm.id === 'svm' && !imageFile)}
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
                {scatterData && (
                  <div className="mt-4">
                    <Scatter data={scatterData} />
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}