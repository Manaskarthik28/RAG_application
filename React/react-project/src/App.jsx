import React, { useState } from 'react';
import './App.css'

const API = '/ask';

function App() {
  // 1. Descriptive name for the user's input
  const [question, setQuestion] = useState(''); 
  
  // 2. Descriptive name for the API result
  const [answer, setAnswer] = useState(null); 
  
  // 3. Standard boolean name for loading status
  const [isLoading, setIsLoading] = useState(false); 

  const submit = async (e) => {
    e.preventDefault();
    setIsLoading(true); // use setIsLoading
    setAnswer(null);    // use setAnswer

    const response = await fetch(API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question }), // use 'question'
    });

    const data = await response.json();
    setAnswer(data.answer.result); // use setAnswer
    setIsLoading(false);           // use setIsLoading
  };

  return (
    <div className="container">
      <h1>RAG Q&A</h1>
      <form onSubmit={submit}>
        <textarea
          value={question} // use 'question'
          onChange={(e) => setQuestion(e.target.value)} // use setQuestion
          rows="3"
          placeholder="Ask a question..."
          disabled={isLoading} // use isLoading 
        />
        <button type="submit" disabled={isLoading}>
          Ask Document
        </button>
      </form>

      {answer && ( // use 'answer'
        <div className="response-area">
          <p>🤖 Answer</p>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}

export default App;