// // import logo from './logo.svg';
// // import './App.css';

// // function App() {
// //   return (
// //     <div className="App">
// //       <header className="App-header">
// //         <img src={logo} className="App-logo" alt="logo" />
// //         <p>
// //           Edit <code>src/App.js</code> and save to reload.
// //         </p>
// //         <a
// //           className="App-link"
// //           href="https://reactjs.org"
// //           target="_blank"
// //           rel="noopener noreferrer"
// //         >
// //           Learn React
// //         </a>
// //       </header>
// //     </div>
// //   );
// // }

// // export default App;
// import React, { useState } from "react";
// import axios from "axios";

// function App() {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [previewURL, setPreviewURL] = useState(null);
//   const [responseData, setResponseData] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleFileChange = (event) => {
//     const file = event.target.files[0];
//     setSelectedFile(file);
//     setResponseData(null);
//     if (file) {
//       setPreviewURL(URL.createObjectURL(file));
//     }
//   };

//   const handleSubmit = async () => {
//     if (!selectedFile) return;

//     const formData = new FormData();
//     formData.append("file", selectedFile);

//     setLoading(true);
//     try {
//       const res = await axios.post("http://localhost:5000/upload-image", formData);
//       setResponseData(res.data);
//     } catch (err) {
//       console.error("Upload failed:", err);
//       setResponseData({ error: "Something went wrong." });
//     }
//     setLoading(false);
//   };

//   return (
//     <div style={{ padding: "20px", fontFamily: "Arial" }}>
//       <h1>🦋 Butterfly Species Identifier</h1>

//       <input type="file" accept="image/*" onChange={handleFileChange} />
//       <br /><br />
//       <button onClick={handleSubmit} disabled={loading}>
//         {loading ? "Processing..." : "Upload Image"}
//       </button>

//       {previewURL && (
//         <div style={{ marginTop: "20px" }}>
//           <h3>Uploaded Image:</h3>
//           <img src={previewURL} alt="Preview" style={{ width: "300px", borderRadius: "10px" }} />
//         </div>
//       )}

//       {responseData && (
//         <div style={{ marginTop: "30px" }}>
//           {responseData.error ? (
//             <p style={{ color: "red" }}>{responseData.error}</p>
//           ) : (
//             <>
//               {responseData.heatmap && (
//                 <>
//                   <h3>Heatmap:</h3>
//                   <img
//                     src={responseData.heatmap}
//                     alt="Butterfly Heatmap"
//                     style={{ width: "300px", borderRadius: "10px" }}
//                   />
//                 </>
//               )}
//               <h3>Description:</h3>
//               <p>{responseData.description}</p>
//             </>
//           )}
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;
import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [caption, setCaption] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please select an image!");
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5000/upload-image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setPrediction(response.data.predicted_label);
      setConfidence((response.data.confidence));
      setCaption(response.data.description); 
      if (response.data.heatmap_url) {
        setHeatmapUrl(`http://localhost:5000${response.data.heatmap_url}`);
      } else {
        setHeatmapUrl(null);
      }
    } catch (error) {
      console.error("Error uploading image:", error);
      alert("Something went wrong! Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🦋 Butterfly Species Classifier</h1>

      <div style={styles.card}>
        <input type="file" onChange={handleFileChange} style={styles.input} />
        <button onClick={handleUpload} style={styles.button} disabled={loading}>
          {loading ? 'Predicting...' : 'Upload and Predict'}
        </button>

        {prediction && (
          <div style={styles.resultContainer}>
            <h2>🎯 Prediction</h2>
            <p><strong>Species:</strong> {prediction}</p>
            <p><strong>Probability:</strong> {confidence}</p>
            {heatmapUrl && (
              <>
                <h3>🔍 Heatmap</h3>
                <img src={heatmapUrl} alt="Heatmap" style={styles.image} />
              </>
            )}
            {caption && (
              <div style={styles.captionBox}>
                <h3>📝 Description</h3>
                <p style={{ fontSize: '1.1rem' }}>{caption}</p>
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '40px',
    fontFamily: 'Poppins, sans-serif',
  },
  title: {
    fontSize: '2.8rem',
    color: '#333',
    marginBottom: '30px',
  },
  card: {
    backgroundColor: '#fff',
    padding: '30px 40px',
    borderRadius: '16px',
    boxShadow: '0 8px 20px rgba(0, 0, 0, 0.15)',
    textAlign: 'center',
    maxWidth: '600px',
    width: '100%',
  },
  input: {
    marginBottom: '20px',
  },
  button: {
    backgroundColor: '#0077ff',
    color: '#fff',
    padding: '10px 20px',
    fontSize: '1.1rem',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    marginBottom: '20px',
  },
  resultContainer: {
    marginTop: '30px',
  },
  image: {
    marginTop: '15px',
    width: '90%',
    borderRadius: '12px',
    // boxShadow: '0 4px 10px rgba(0, 0, 0, 0.25)',
  }
};

export default App;

