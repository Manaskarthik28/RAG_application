import React, {useState} from "react";
import './App.css';

const API = '/ask';
function App(){
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const submit = async (e) =>{
    e.preventDefault()
    setIsLoading(true);
    setAnswer(null)
    const response = await fetch(API, {
      method:'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question:question})
    });
    const data = await response.json()
    setAnswer(data.answer.result);
    setIsLoading(false);
  }

  return(
    <div className="container">
      <h2>RAG Q&A</h2>
      <p>This smart bot can answer questions from your documents.</p>
      <p>Dont trust.Ask question to find out!</p>
      <form onSubmit={submit}>
        <textarea value={question} rows="3" 
        onChange={(e)=>setQuestion(e.target.value)}
        placeholder="Enter a Question here"
        disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Ask</button>
      </form>
      {answer && (
        <div className="response-area">
          <p>🤖 Answer</p>
          <textarea value={answer} rows="3" />
          </div>
      )}
    </div>
  )
}

export default App;
