import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'
import HomePage from './home-page'
import AlgorithmTest from './algorithm-test'

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/algorithm/:id" element={<AlgorithmTest />} />
      </Routes>
    </Router>
  )
}