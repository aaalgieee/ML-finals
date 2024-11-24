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
  { 
    id: 'linear-regression', 
    name: 'Linear Regression', 
    icon: LineChart, 
    description: 'Calculate your risk of developing hypertension using advanced prediction models',
    inputs: [
      { name: 'Age', type: 'number', min: 0, max: 120, placeholder: 'Enter your age', help: 'Your current age in years' },
      { name: 'Gender', type: 'select', options: [
        { value: 'Male', label: 'Male' },
        { value: 'Female', label: 'Female' }
      ]},
      { name: 'Current Smoker', type: 'select', options: [
        { value: 'true', label: 'Yes' },
        { value: 'false', label: 'No' }
      ]},
      { name: 'Cigarettes Per Day', type: 'number', min: 0, max: 100, placeholder: 'Number of cigarettes', help: 'Average number of cigarettes smoked per day' },
      { name: 'BP Medications', type: 'select', options: [
        { value: 'true', label: 'Yes' },
        { value: 'false', label: 'No' }
      ]},
      { name: 'Diabetes', type: 'select', options: [
        { value: 'true', label: 'Yes' },
        { value: 'false', label: 'No' }
      ]},
      { name: 'Total Cholesterol', type: 'number', min: 0, max: 600, placeholder: 'Cholesterol level', help: 'Total cholesterol in mg/dL' },
      { name: 'Systolic BP', type: 'number', min: 0, max: 300, placeholder: 'Systolic blood pressure', help: 'Upper blood pressure number in mmHg' },
      { name: 'Diastolic BP', type: 'number', min: 0, max: 200, placeholder: 'Diastolic blood pressure', help: 'Lower blood pressure number in mmHg' },
      { name: 'BMI', type: 'number', min: 0, max: 100, step: 0.1, placeholder: 'Body Mass Index', help: 'Your Body Mass Index (weight/height²)' },
      { name: 'Heart Rate', type: 'number', min: 0, max: 250, placeholder: 'Heart rate', help: 'Resting heart rate in beats per minute' },
      { name: 'Glucose', type: 'number', min: 0, max: 500, placeholder: 'Blood glucose', help: 'Blood glucose level in mg/dL' }
    ]
  },
  { 
    id: 'naive-bayes', 
    name: 'Naive Bayes', 
    icon: Brain, 
    description: 'Assess your heart disease risk with our intelligent diagnostic tool',
    inputs: [
      { name: 'age', label: 'Age', type: 'number', min: 0, max: 120, placeholder: 'Enter your age', help: 'Your current age in years' },
      { name: 'sex', label: 'Gender', type: 'select', options: [
        { value: '1', label: 'Female' },
        { value: '0', label: 'Male' }
      ], help: 'Select your gender' },
      { name: 'cp', label: 'Chest Pain Type', type: 'select', options: [
        { value: '0', label: 'Typical Angina' },
        { value: '1', label: 'Atypical Angina' },
        { value: '2', label: 'Non-anginal Pain' },
        { value: '3', label: 'Asymptomatic' }
      ], help: 'Type of chest pain you experience' },
      { name: 'trestbps', label: 'Resting Blood Pressure', type: 'number', min: 0, max: 300, placeholder: 'Enter BP value', help: 'Blood pressure (in mm Hg) while resting', unit: 'mmHg' },
      { name: 'chol', label: 'Cholesterol Level', type: 'number', min: 0, max: 600, placeholder: 'Enter cholesterol', help: 'Serum cholesterol in mg/dl', unit: 'mg/dl' },
      { name: 'fbs', label: 'Fasting Blood Sugar > 120 mg/dl', type: 'select', options: [
        { value: '1', label: 'Yes' },
        { value: '0', label: 'No' }
      ]},
      { name: 'restecg', label: 'Resting ECG Results', type: 'select', options: [
        { value: '0', label: 'Normal' },
        { value: '1', label: 'ST-T Wave Abnormality' },
        { value: '2', label: 'Left Ventricular Hypertrophy' }
      ]},
      { name: 'thalach', label: 'Maximum Heart Rate', type: 'number', min: 0, max: 250 },
      { name: 'exang', label: 'Exercise Induced Angina', type: 'select', options: [
        { value: '1', label: 'Yes' },
        { value: '0', label: 'No' }
      ]},
      { name: 'oldpeak', label: 'ST Depression', type: 'number', min: 0, max: 10, step: 0.1 },
      { name: 'slope', label: 'Slope of Peak Exercise ST', type: 'select', options: [
        { value: '0', label: 'Upsloping' },
        { value: '1', label: 'Flat' },
        { value: '2', label: 'Downsloping' }
      ]},
      { name: 'ca', label: 'Number of Major Vessels', type: 'select', options: [
        { value: '0', label: '0' },
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
        { value: '4', label: '4' }
      ]},
      { name: 'thal', label: 'Thalium Stress Test', type: 'select', options: [
        { value: '0', label: 'Normal' },
        { value: '1', label: 'Fixed Defect' },
        { value: '2', label: 'Reversible Defect' },
        { value: '3', label: 'Not Described' }
      ]}
    ]
  },
  { 
    id: 'knn', 
    name: 'K-Nearest Neighbors', 
    icon: Network, 
    description: 'Screen for diabetes risk based on your health metrics',
    inputs: [
      { name: 'Gender', type: 'select', options: [
        { value: 'Male', label: 'Male' },
        { value: 'Female', label: 'Female' }
      ], help: 'Select your gender' },
      { name: 'Age', type: 'number', min: 0, max: 120, placeholder: 'Enter your age', help: 'Your current age in years' },
      { name: 'Hypertension', type: 'select', options: [
        { value: 'true', label: 'Yes' },
        { value: 'false', label: 'No' }
      ], help: 'Do you have hypertension?' },
      { name: 'Heart Disease', type: 'select', options: [
        { value: 'true', label: 'Yes' },
        { value: 'false', label: 'No' }
      ], help: 'Do you have any heart conditions?' },
      { name: 'Smoking History', type: 'select', options: [
        { value: 'never', label: 'Never Smoked' },
        { value: 'current', label: 'Current Smoker' },
        { value: 'former', label: 'Former Smoker' },
        { value: 'not current', label: 'Not Current Smoker' }
      ], help: 'Your smoking history' },
      { name: 'BMI', type: 'number', min: 10, max: 50, step: 0.1, placeholder: 'Enter your BMI', help: 'Body Mass Index', unit: 'kg/m²' },
      { name: 'HbA1c Level', type: 'number', min: 3, max: 9, step: 0.1, placeholder: 'Enter HbA1c level', help: 'Glycated hemoglobin level', unit: '%' },
      { name: 'Blood Glucose Level', type: 'number', min: 70, max: 300, placeholder: 'Enter glucose level', help: 'Random blood glucose level', unit: 'mg/dL' }
    ]
  },
  { 
    id: 'svm', 
    name: 'Support Vector Machine', 
    icon: Activity, 
    description: 'Analyze chest X-rays to detect signs of pneumonia',
    inputs: [{ 
      type: 'image',
      help: 'Upload a clear chest X-ray image for pneumonia analysis',
      accept: ['image/jpeg', 'image/png'],
      maxSize: 10 * 1024 * 1024 // 10MB
    }]
  },
  { 
    id: 'decision-tree', 
    name: 'Decision Tree', 
    icon: TreeDeciduous, 
    description: 'Evaluate maternal health risks during pregnancy',
    inputs: [
      { name: 'Age', type: 'number', min: 13, max: 70, placeholder: 'Enter age', help: 'Mother\'s age in years' },
      { name: 'Systolic BP', type: 'number', min: 70, max: 180, placeholder: 'Enter systolic BP', help: 'Upper blood pressure number', unit: 'mmHg' },
      { name: 'Diastolic BP', type: 'number', min: 40, max: 120, placeholder: 'Enter diastolic BP', help: 'Lower blood pressure number', unit: 'mmHg' },
      { name: 'Blood Sugar', type: 'number', min: 30, max: 300, placeholder: 'Enter blood sugar', help: 'Blood glucose level', unit: 'mg/dL' },
      { name: 'Body Temperature', type: 'number', min: 35, max: 42, step: 0.1, placeholder: 'Enter temperature', help: 'Body temperature', unit: '°C' },
      { name: 'Heart Rate', type: 'number', min: 40, max: 200, placeholder: 'Enter heart rate', help: 'Heart beats per minute', unit: 'bpm' }
    ]
  },
  { 
    id: 'ann', 
    name: 'Artificial Neural Network', 
    icon: Brain, 
    description: 'Detect and locate brain tumors in MRI scans with precision',
    inputs: [{ 
      type: 'image',
      help: 'Upload a clear MRI scan for tumor detection',
      accept: ['image/jpeg', 'image/png'],
      maxSize: 10 * 1024 * 1024 // 10MB
    }]
  }
]

// Update the PredictionResponse type to include the new fields
type PredictionResponse = {
  success: boolean;
  prediction: number;
  message: string;
  confidence?: number;
  raw_prediction?: string;
  risk_level?: string;
  tumor_region?: { x: number, y: number, width: number, height: number }; // Add this line
  risk_factors?: string[];
  probabilities?: {
    low_risk: number;
    high_risk: number;
  };
  details?: {
    systolic_bp: number;
    cholesterol: number;
    max_heart_rate: number;
  };
}

export default function AlgorithmTest() {
  const { id } = useParams<{ id: string }>()
  const [algorithm, setAlgorithm] = useState<typeof algorithms[0] | undefined>()
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [result, setResult] = useState<string | JSX.Element | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)

  useEffect(() => {
    const foundAlgorithm = algorithms.find(algo => algo.id === id)
    setAlgorithm(foundAlgorithm)
    if (foundAlgorithm) {
      const initialInputs = foundAlgorithm.inputs.reduce((acc, input) => {
        if (typeof input === 'string') return { ...acc, [input]: '' };
        if ('name' in input) return { ...acc, [input.name]: '' };
        return acc;
      }, {})
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

  const API_URL = import.meta.env.VITE_API_URL;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    
    if (algorithm?.id === 'ann' && imageFile) {
      try {
        const formData = new FormData()
        formData.append('image', imageFile)
        
        const response = await fetch(`${API_URL}/api/ann/predict`, {
          method: 'POST',
          body: formData
        })
  
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
  
        const data: PredictionResponse = await response.json()
        setResult(
          <div className="space-y-2">
            <p className="font-semibold">Detected: {data.message}</p>
            <p className="text-sm text-gray-600">
              Confidence: {data.confidence ? `${data.confidence.toFixed(1)}%` : 'N/A'}
            </p>
            {imagePreview && data.tumor_region && data.message !== "No Tumor" && (
              <div className="relative inline-block w-full" style={{ maxHeight: '800px', overflow: 'hidden' }}>
                <img
                  src={imagePreview}
                  alt="MRI Scan"
                  className="w-full object-contain"
                  style={{ maxHeight: '800px' }}
                />
                <div
                  className="absolute border-2 border-red-500 animate-pulse"
                  style={{
                    left: `${data.tumor_region.x}%`,
                    top: `${data.tumor_region.y}%`,
                    width: `${data.tumor_region.width}%`,
                    height: `${data.tumor_region.height}%`,
                  }}
                />
              </div>
            )}
          </div>
        )
      } catch (error) {
        console.error('Upload error:', error)
        setResult(`Error: ${error instanceof Error ? error.message : 'Error processing image'}`)
      } finally {
        setIsLoading(false)
      }
    } else if (algorithm?.id === 'linear-regression') {
      try {
        const transformedInputs = {
          age: Number(inputs['Age']),
          male: inputs['Gender'] === 'Male' ? 1 : 0,  // Convert Gender to binary
          currentSmoker: inputs['Current Smoker'] === 'true' ? 1 : 0,
          cigsPerDay: Number(inputs['Cigarettes Per Day']),
          BPMeds: inputs['BP Medications'] === 'true' ? 1 : 0,
          diabetes: inputs['Diabetes'] === 'true' ? 1 : 0,
          totChol: Number(inputs['Total Cholesterol']),
          sysBP: Number(inputs['Systolic BP']),
          diaBP: Number(inputs['Diastolic BP']),
          BMI: Number(inputs['BMI']),
          heartRate: Number(inputs['Heart Rate']),
          glucose: Number(inputs['Glucose'])
        }
  
        const response = await fetch(`${API_URL}/api/lr/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ features: transformedInputs })
        })
  
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
  
        const data: PredictionResponse = await response.json()
        if (data.success) {
          setResult(
            <div className="space-y-2">
              <p className="font-semibold">{data.risk_level}</p>
              <p>{data.message}</p>
              <p className="text-sm text-gray-600">Risk Score: {data.prediction.toFixed(1)}%</p>
            </div>
          )
        } else {
          throw new Error('Prediction failed')
        }
      } catch (error) {
        console.error('Prediction error:', error)
        setResult(`Error: ${error instanceof Error ? error.message : 'Error processing prediction'}`)
      } finally {
        setIsLoading(false)
      }
    } else if (algorithm?.id === 'svm' && imageFile) {
      try {
        const formData = new FormData()
        formData.append('image', imageFile)

        console.log('Sending file:', imageFile.name, imageFile.type)
        
        const response = await fetch(`${API_URL}/api/svm/predict`, {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data: PredictionResponse = await response.json()
        console.log('Response:', data)
        
        setResult(`Classification: ${data.message} ${data.confidence !== undefined ? `(Confidence: ${data.confidence.toFixed(1)}%)` : ''}`)
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

        const response = await fetch(`${API_URL}/api/knn/predict`, {
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
          const riskLevel = data.prediction === 'Diabetic' ? 'High Risk' : 'Low Risk'
          setResult(
            <div className="space-y-4">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <p className="text-xl font-bold text-gray-900">
                  {riskLevel} of Diabetes
                </p>
                <p className="text-gray-600 mt-1">{data.message}</p>
                {data.confidence && (
                  <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className={`h-2.5 rounded-full ${
                          data.prediction === 'Diabetic' ? 'bg-red-600' : 'bg-green-500'
                        }`}
                        style={{ width: `${data.confidence}%` }}
                      />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      Confidence: {data.confidence.toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>
              {data.scatter_plot_data && (
                <div className="bg-white p-4 rounded-lg border border-gray-200">
                  <Scatter 
                    data={{
                      datasets: [{
                        label: 'Patient Data',
                        data: data.scatter_plot_data,
                        backgroundColor: data.prediction === 'Diabetic' ? 'rgba(255, 99, 132, 0.8)' : 'rgba(75, 192, 192, 0.8)',
                        pointRadius: 8
                      }]
                    }}
                    options={{
                      scales: {
                        x: { title: { display: true, text: 'Age' } },
                        y: { title: { display: true, text: 'BMI' } }
                      }
                    }}
                  />
                </div>
              )}
            </div>
          )
        } else {
          setResult('Error: Could not generate prediction')
        }
      } catch (error) {
        console.error('Prediction error:', error)
        setResult(`Error: ${error instanceof Error ? error.message : 'Error processing prediction'}`)
      } finally {
        setIsLoading(false)
      }
    } else if (algorithm?.id === 'decision-tree') {
      try {
        const transformedInputs = {
          age: Number(inputs['Age']),
          systolicBP: Number(inputs['Systolic BP']),
          diastolicBP: Number(inputs['Diastolic BP']),
          bs: Number(inputs['Blood Sugar']),
          bodyTemp: Number(inputs['Body Temperature']),
          heartRate: Number(inputs['Heart Rate'])
        }

        const response = await fetch(`${API_URL}/api/dt/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ features: transformedInputs })
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data: PredictionResponse = await response.json()
        if (data.success) {
          setResult(
            <div className="space-y-2">
              <p className="font-semibold">Risk Level: {data.risk_level}</p>
              <p className="text-sm text-gray-600">{data.message}</p>
              <p className="text-sm text-gray-600">
                Confidence: {data.confidence ? data.confidence.toFixed(1) : 'N/A'}%
              </p>
            </div>
          )
        } else {
          throw new Error('Prediction failed')
        }
      } catch (error) {
        console.error('Prediction error:', error)
        setResult(`Error: ${error instanceof Error ? error.message : 'Error processing prediction'}`)
      } finally {
        setIsLoading(false)
      }
    } else if (algorithm?.id === 'naive-bayes') {
      try {
        const transformedInputs = {
          age: Number(inputs['age']),
          sex: Number(inputs['sex']),
          cp: Number(inputs['cp']),
          trestbps: Number(inputs['trestbps']),
          chol: Number(inputs['chol']),
          fbs: Number(inputs['fbs']),
          restecg: Number(inputs['restecg']),
          thalach: Number(inputs['thalach']),
          exang: Number(inputs['exang']),
          oldpeak: Number(inputs['oldpeak']),
          slope: Number(inputs['slope']),
          ca: Number(inputs['ca']),
          thal: Number(inputs['thal'])
        }
    
        const response = await fetch(`${API_URL}/api/nb/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ features: transformedInputs })
        })
    
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
    
        const data: PredictionResponse = await response.json()
        if (data.success) {
          setResult(
            <div className="space-y-4">
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <p className="text-xl font-bold text-gray-900">
                  {data.risk_level} Risk Level
                </p>
                <p className="text-gray-600 mt-1">{data.message}</p>
                {data.probabilities && (
                  <div className="mt-4 grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-500">Low Risk Probability</p>
                      <p className="text-lg font-medium">{(data.probabilities.low_risk * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">High Risk Probability</p>
                      <p className="text-lg font-medium">{(data.probabilities.high_risk * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                )}
              </div>
              {/* Rest of the existing result display */}
            </div>
          )
        } else {
          throw new Error('Prediction failed')
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
              {(algorithm.id === 'svm' || algorithm.id === 'ann') ? (
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
                        className="max-h-64 mx-auto rounded-lg shadow-md object-contain"  // Changed back to max-h-64
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
                  {algorithm.inputs.map((input) => {
                    if ('type' in input && input.type === 'image') {
                      return null; // Skip image inputs as they're handled separately
                    }
                    
                    const inputName = typeof input === 'string' ? input : 'name' in input ? input.name : '';
                    const inputLabel = typeof input === 'string' ? input : 'label' in input ? input.label : 'name' in input ? input.name : '';
                    
                    return (
                      <div key={inputName} className="space-y-2">
                        <Label htmlFor={inputName} className="text-gray-700">
                          {inputLabel}
                        </Label>
                        {typeof input !== 'string' && 'type' in input && input.type === 'select' && 'options' in input ? (
                          <select
                            id={inputName}
                            value={inputs[inputName]}
                            onChange={(e) => handleInputChange(inputName, e.target.value)}
                            required
                            className="bg-gray-50 hover:bg-gray-100 focus:bg-white w-full p-2 border border-gray-300 rounded-md"
                          >
                            <option value="">Select {inputLabel}</option>
                            {input.options?.map((option: { value: string; label: string }) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        ) : typeof input !== 'string' ? (
                          <Input
                            id={inputName}
                            type={'type' in input ? input.type : 'text'}
                            min={'min' in input ? input.min : undefined}
                            max={'max' in input ? input.max : undefined}
                            step={'step' in input ? input.step : undefined}
                            value={inputs[inputName]}
                            onChange={(e) => handleInputChange(inputName, e.target.value)}
                            required
                            className="bg-gray-50 hover:bg-gray-100 focus:bg-white"
                            placeholder={`Enter ${inputLabel.toLowerCase()}`}
                          />
                        ) : (
                          <Input
                            id={input}
                            value={inputs[input]}
                            onChange={(e) => handleInputChange(input, e.target.value)}
                            required
                            className="bg-gray-50 hover:bg-gray-100 focus:bg-white"
                            placeholder={`Enter value`}
                          />
                        )}
                      </div>
                    );
                  })}
                </>
              )}
              <Button 
                type="submit" 
                className={`w-full transition-all duration-200 ${
                  isLoading ? 'bg-blue-500' : 'bg-blue-600 hover:bg-blue-700'
                }`}
                disabled={isLoading || ((algorithm.id === 'svm' || algorithm.id === 'ann') && !imageFile)}
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
                <div className="text-gray-800 font-medium">{result}</div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}