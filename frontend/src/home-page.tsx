import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from "./components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Textarea } from "./components/ui/textarea"
import { Brain, Activity, Stethoscope, LineChart, Network, TreeDeciduous } from 'lucide-react'

export default function HomePage() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')

  const algorithms = [
    { id: 'linear-regression', name: 'Linear Regression', icon: LineChart, description: 'Predicting patient recovery times' },
    { id: 'naive-bayes', name: 'Naive Bayes', icon: Brain, description: 'Classifying patient symptoms for diagnosis' },
    { id: 'knn', name: 'K-Nearest Neighbors', icon: Network, description: 'Identifying similar patient profiles' },
    { id: 'svm', name: 'Support Vector Machine', icon: Activity, description: 'Classifying medical images' },
    { id: 'decision-tree', name: 'Decision Tree', icon: TreeDeciduous, description: 'Guiding treatment decisions' },
    { id: 'ann', name: 'Artificial Neural Network', icon: Brain, description: 'Predicting disease outbreaks' },
  ]

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    console.log('Form submitted:', { name, email, message })
    // Here you would typically send this data to your backend
    setName('')
    setEmail('')
    setMessage('')
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-100 to-green-100">
      <header className="bg-white shadow-md sticky top-0 z-10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Stethoscope className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-2xl font-bold text-gray-800">HealthAI Solutions</span>
            </div>
            <nav>
              <ul className="flex space-x-4">
                <li><a href="#home" className="text-gray-600 hover:text-blue-600">Home</a></li>
                <li><a href="#algorithms" className="text-gray-600 hover:text-blue-600">Algorithms</a></li>
                <li><a href="#contact" className="text-gray-600 hover:text-blue-600">Contact</a></li>
              </ul>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-12 max-w-7xl">
        <section id="home" className="mb-24 text-center max-w-3xl mx-auto">
          <h1 className="mb-4 text-4xl font-bold text-gray-800">Transforming Healthcare with AI</h1>
          <p className="mb-8 text-xl text-gray-600">HealthAI Solutions leverages machine learning to address specific challenges in healthcare, improving patient outcomes and operational efficiency.</p>
          <Button size="lg">Learn More</Button>
        </section>

        <section id="algorithms" className="mb-24">
          <h2 className="mb-12 text-3xl font-bold text-center text-gray-800">Our Machine Learning Algorithms</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {algorithms.map((algo) => (
              <Link to={`/algorithm/${algo.id}`} key={algo.id}>
                <Card className="cursor-pointer transition-all duration-300 hover:shadow-xl hover:scale-105 bg-white/95 backdrop-blur-sm h-full">
                  <CardHeader className="text-center pb-2">
                    <CardTitle className="flex flex-col items-center space-y-3">
                      <algo.icon className="h-10 w-10 text-blue-600" />
                      <span className="text-xl">{algo.name}</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-center text-sm">{algo.description}</CardDescription>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </section>

        <section id="contact" className="mb-24">
          <Card className="bg-white shadow-lg">
            <CardHeader>
              <CardTitle>Contact Us</CardTitle>
              <CardDescription>Have questions? Get in touch with our team.</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label htmlFor="name" className="block text-sm font-medium text-gray-700">Name</label>
                  <Input
                    id="name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700">Email</label>
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label htmlFor="message" className="block text-sm font-medium text-gray-700">Message</label>
                  <Textarea
                    id="message"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    required
                  />
                </div>
                <Button type="submit">Send Message</Button>
              </form>
            </CardContent>
          </Card>
        </section>
      </main>

      <footer className="bg-gray-800 text-white py-12">
        <div className="container mx-auto px-6 text-center">
          <p>&copy; 2024 HealthAI Solutions. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}