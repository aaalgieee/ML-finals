import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card"
import { Stethoscope, Heart, Award, Target } from 'lucide-react'
import { Link } from 'react-router-dom'

export default function AboutPage() {
  const features = [
    {
      icon: Heart,
      title: "Our Mission",
      description: "Revolutionizing patient care through cutting-edge machine learning technologies and enhancing treatment personalization."
    },
    {
      icon: Award,
      title: "Our Values",
      description: "Committed to excellence, innovation, and improving healthcare outcomes through advanced analytics and AI solutions."
    },
    {
      icon: Target,
      title: "Our Goals",
      description: "Developing predictive models, streamlining healthcare operations, and creating personalized treatment plans."
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-100 to-green-100">
      <header className="bg-white shadow-md sticky top-0 z-10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center">
              <Stethoscope className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-2xl font-bold text-gray-800">HealthAI Solutions</span>
            </Link>
            <nav>
              <ul className="flex space-x-4">
                <li><Link to="/" className="text-gray-600 hover:text-blue-600">Home</Link></li>
                <li><Link to="/about" className="text-gray-600 hover:text-blue-600">About</Link></li>
              </ul>
            </nav>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-12 max-w-7xl">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">About HealthAI Solutions</h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Leading the transformation in healthcare through innovative AI solutions
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <Card key={index} className="bg-white/95 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <CardHeader className="text-center pb-2">
                <feature.icon className="h-12 w-12 mx-auto text-blue-600 mb-4" />
                <CardTitle className="text-xl font-semibold text-gray-800">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 text-center">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Main Content */}
        <Card className="bg-white/95 backdrop-blur-sm hover:shadow-xl transition-all duration-300">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-center text-gray-800">Our Story</CardTitle>
          </CardHeader>
          <CardContent className="text-gray-600 leading-relaxed space-y-6">
            <p className="text-lg">
              At HealthAI Solutions, we are dedicated to revolutionizing patient care through cutting-edge machine learning technologies. 
              Our mission is to enhance treatment personalization, improve diagnostic accuracy, and drive better health outcomes for patients worldwide.
            </p>
            <p className="text-lg">
              Guided by our vision to be at the forefront of healthcare technology, we leverage data-driven insights to create innovative solutions 
              that empower healthcare providers and improve lives.
            </p>
            <p className="text-lg">
              Our operational goals include developing predictive models to aid patient recovery, streamlining healthcare operations with advanced analytics, 
              and crafting personalized treatment plans based on individual patient data. At HealthAI Solutions, we believe in transforming healthcare for a healthier future.
            </p>
          </CardContent>
        </Card>
      </main>

      <footer className="bg-gray-800 text-white py-12">
        <div className="container mx-auto px-6 text-center">
          <p>&copy; 2024 HealthAI Solutions. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}