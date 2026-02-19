import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import QuizPage from './pages/QuizPage.jsx'
import FeedPage from './pages/FeedPage.jsx'

function RequireQuiz({ children }) {
  const quizDone = localStorage.getItem('quizCompleted') === 'true'
  if (!quizDone) return <Navigate to="/quiz" replace />
  return children
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/quiz" element={<QuizPage />} />
        <Route
          path="/feed"
          element={
            <RequireQuiz>
              <FeedPage />
            </RequireQuiz>
          }
        />
        <Route path="*" element={<Navigate to="/quiz" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
