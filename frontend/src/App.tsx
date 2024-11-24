import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import HomePage from './home-page'
import AlgorithmTest from './algorithm-test'
import AboutPage from './about-page'

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/algorithm/:id" element={<AlgorithmTest />} />
      </Routes>
    </Router>
  )
}