import logo from './logo.svg';
import './App.css';

function App() {
  fetch('http://logical_circuits_client')
    .then( res => {
      console.log( res.json() )
    })
  return (
    <div className="App">
       Hello 
    </div>
  );
}

export default App;
