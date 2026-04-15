import { useState } from 'react';
import axios from 'axios';
import { Upload, Activity, Image as ImageIcon, Zap, ShieldAlert, LayoutDashboard } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

import { MODEL_DETAILS } from './data/modelsInfo';
import ModelCard from './components/ModelCard';
import ModelDetailTab from './components/ModelDetailTab'; // Zmieniony import

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);
  
  // Zamiast modala, przechowujemy aktualną zakładkę. 'summary' = strona główna wyników.
  const [activeTab, setActiveTab] = useState('summary');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setPredictions(null);
      setError(null);
      setActiveTab('summary');
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setActiveTab('summary');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/v1/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPredictions(response.data.predictions);
    } catch (err) {
      setError('Błąd połączenia z serwerem. Upewnij się, że FastAPI działa.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-8 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Nagłówek */}
        <header className="flex items-center justify-between pb-6 border-b border-slate-800">
          <div className="flex items-center gap-4">
            <ShieldAlert className="w-10 h-10 text-indigo-500" />
            <div>
              <h1 className="text-3xl font-bold text-white">System Detekcji Deepfake</h1>
              <p className="text-slate-400">Analiza wielomodelowa (Ensemble) z zaawansowanym XAI.</p>
            </div>
          </div>
          {predictions && activeTab !== 'summary' && (
             <div className="bg-slate-900 border border-slate-800 px-4 py-2 rounded-xl text-sm font-semibold text-slate-300 flex items-center gap-2">
               <Activity className="w-4 h-4 text-emerald-400 animate-pulse"/> Sesja analityczna w toku
             </div>
          )}
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* LEWA KOLUMNA STAŁA (Upload obrazu zawsze widoczny) */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl sticky top-8">
              <h2 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
                <Upload className="w-5 h-5 text-indigo-400" /> Obraz wejściowy
              </h2>
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-700 border-dashed rounded-xl cursor-pointer bg-slate-950 hover:bg-slate-900/50 transition-colors">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <ImageIcon className="w-8 h-8 mb-2 text-slate-500" />
                  <p className="mb-2 text-xs text-slate-400 font-semibold text-center px-2">Zmień obraz</p>
                </div>
                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>

              {preview && (
                <div className="mt-4">
                  <img src={preview} alt="Podgląd" className="w-full rounded-xl border border-slate-700 shadow-lg object-cover" />
                  <button 
                    onClick={handleAnalyze}
                    disabled={loading}
                    className="w-full mt-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 px-4 rounded-xl transition-all disabled:opacity-50 flex justify-center items-center gap-2"
                  >
                    {loading ? <Activity className="w-5 h-5 animate-spin" /> : <Zap className="w-5 h-5" />}
                    {loading ? 'Analizowanie...' : 'Uruchom Algorytmy'}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* PRAWA KOLUMNA DYNAMICZNA (Zakładki) */}
          <div className="lg:col-span-3">
            {!predictions ? (
              <div className="h-full min-h-[500px] flex flex-col items-center justify-center text-slate-500 border-2 border-slate-800 border-dashed rounded-2xl bg-slate-900/20">
                <LayoutDashboard className="w-16 h-16 mb-4 opacity-20" />
                <p>Wgraj obraz i uruchom skanowanie, aby wygenerować raport analityczny.</p>
              </div>
            ) : (
              <>
                {/* RENDEROWANIE WŁAŚCIWEJ ZAKŁADKI */}
                {activeTab === 'summary' ? (
                  <div className="space-y-6 animate-in fade-in duration-300">
                    {/* LLM */}
                    {predictions.llm_verdict && (
                      <div className="bg-indigo-950/40 border border-indigo-500/30 rounded-2xl p-6 shadow-xl relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500"></div>
                        <h2 className="text-xl font-semibold mb-3 text-indigo-300 flex items-center gap-2">
                          <Zap className="w-5 h-5 text-indigo-400" /> Werdykt Eksperta (DeepSeek AI)
                        </h2>
                        <div className="text-slate-300 leading-relaxed text-sm space-y-2 [&>p>strong]:text-indigo-300 [&>p>strong]:font-bold [&>ul]:list-disc [&>ul]:ml-5">
                          <ReactMarkdown>{predictions.llm_verdict}</ReactMarkdown>
                        </div>
                      </div>
                    )}
                    
                    {/* Karty modeli (Kliknięcie zmienia activeTab) */}
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
                      <h2 className="text-xl font-semibold mb-6 text-white flex items-center gap-2">
                        <LayoutDashboard className="w-5 h-5 text-blue-400" /> Modele Analityczne (Kliknij po szczegóły)
                      </h2>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {Object.keys(MODEL_DETAILS).map((key) => {
                          if (predictions[`${key}_prob`] === undefined) return null;
                          return (
                            <ModelCard 
                              key={key}
                              modelKey={key}
                              modelData={MODEL_DETAILS[key]}
                              prob={predictions[`${key}_prob`] * 100}
                              onClick={setActiveTab} // Zmienia na np. 'clip'
                            />
                          );
                        })}
                      </div>
                    </div>
                  </div>
                ) : (
                  // RENDEROWANIE ZAKŁADKI SZCZEGÓŁOWEJ DANEGO MODELU
                  <ModelDetailTab 
                    modelKey={activeTab}
                    modelData={MODEL_DETAILS[activeTab]}
                    predictions={predictions}
                    onBack={() => setActiveTab('summary')}
                  />
                )}
              </>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}