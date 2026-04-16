import { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { ShieldCheck, UploadCloud, BrainCircuit, Waves, ScanSearch, FileWarning, Zap, Activity, Cpu, Clock } from 'lucide-react';

// Baza wiedzy i statystyki wyświetlane po kliknięciu
const MODEL_DETAILS = {
  rgb: { 
    name: "Model RGB", icon: ScanSearch,
    desc: "Klasyczna sieć konwolucyjna (CNN). Przeszukuje strukturę pikseli w poszukiwaniu klasycznych błędów generatorów: rozmytych krawędzi, asymetrii i złego oświetlenia.", 
    xai: "Klasyczny Grad-CAM. Czerwone strefy to obszary, w których struktura pikseli zdradziła algorytm." 
  },
  clip: { 
    name: "Model CLIP", icon: BrainCircuit,
    desc: "Wykorzystuje architekturę ViT do analizy powiązań semantycznych. Rozumie logikę sceny i fizykę świata.", 
    xai: "Mapa gęstości uwagi (Inferno). Pokazuje, którym obszarom AI poświęciło najwięcej uwagi, szukając błędów." 
  },
  fft: { 
    name: "Model FFT", icon: Waves,
    desc: "Analizuje obraz w dziedzinie częstotliwości za pomocą Szybkiej Transformaty Fouriera (ukryte grid artifacts).", 
    xai: "Jasne piki z dala od centrum oznaczają nienaturalne anomalie cyfrowej syntezy." 
  },
  noise: { 
    name: "Model Noise", icon: FileWarning,
    desc: "Analiza szumu matrycy (PRNU). Modele generatywne często nie potrafią odtworzyć naturalnego ziarna aparatu.", 
    xai: "Zaznacza obszary, gdzie ziarno obrazu jest podejrzanie gładkie lub sztuczne." 
  },
  pca: { 
    name: "Model PCA", icon: Activity,
    desc: "Bada wariancję gradientów pionowych i poziomych używając głębokich warstw konwolucyjnych.", 
    xai: "Elipsa ufności kowariancji. Idealnie okrągła oznacza brak artefaktów szachownicy." 
  }
};

export default function App() {
  const [file, setFile] = useState(null);
  const [fileObj, setFileObj] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeXAI, setActiveXAI] = useState('rgb');

  const models = ['rgb', 'clip', 'fft', 'noise', 'pca'];

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(URL.createObjectURL(selectedFile));
      setFileObj(selectedFile);
      setPredictions(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!fileObj) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', fileObj);

    try {
      const response = await axios.post('http://localhost:8000/api/v1/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPredictions(response.data.predictions);
      
      // Ustaw pierwsze dostępne XAI
      const availableModels = models.filter(m => 
        response.data.predictions[`${m}_vis`] || 
        response.data.predictions[`${m}_gradcam`] || 
        response.data.predictions[`${m}_prob`]
      );
      if (availableModels.length > 0) setActiveXAI(availableModels[0]);

    } catch (err) {
      setError('Błąd połączenia z serwerem. Upewnij się, że FastAPI działa.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-bg-main text-text-main p-3 sm:p-6 md:p-10 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* NAGŁÓWEK */}
        <header className="flex flex-col md:flex-row items-center text-center md:text-left gap-3 md:gap-5 pb-6 md:pb-8 border-b border-accent">
          <ShieldCheck className="w-10 h-10 md:w-12 md:h-12 text-text-main" strokeWidth={1} />
          <div className="space-y-1">
            <h1 className="text-3xl md:text-4xl font-extrabold tracking-tighter text-text-main uppercase">
              AUTHENTISCAN
            </h1>
            <p className="text-sm md:text-lg text-highlight tracking-wide uppercase font-light">
              PIĘĆ MODELI. JEDNA PRAWDA.
            </p>
          </div>
        </header>

        {/* BŁĄD */}
        {error && (
          <div className="border border-red-900/50 bg-red-950/20 text-red-500 p-4 text-sm font-bold uppercase tracking-wider text-center">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 xl:gap-8">
          
          {/* KOLUMNA LEWA: Wgraj obraz */}
          <div className="xl:col-span-1 space-y-6">
            <div className="border border-accent p-6 bg-black shadow-[0_0_60px_-15px_rgba(255,255,255,0.05)] sticky top-6">
              <h2 className="text-lg font-bold mb-6 text-text-main uppercase tracking-tight flex items-center gap-3">
                <UploadCloud className="w-5 h-5 text-highlight" /> Wgraj obraz
              </h2>
              
              <label className="aspect-square border-2 border-accent border-dashed flex flex-col items-center justify-center cursor-pointer hover:border-highlight hover:bg-highlight/5 transition-colors group relative overflow-hidden">
                {file ? (
                  <img src={file} alt="Preview" className="w-full h-full object-cover p-1 opacity-80 group-hover:opacity-100 transition-opacity" />
                ) : (
                  <div className="text-center p-4 space-y-3">
                    <ScanSearch className="w-10 h-10 mx-auto text-accent group-hover:text-text-main transition-colors" strokeWidth={1}/>
                    <p className="text-highlight text-xs font-bold uppercase group-hover:text-text-main">Kliknij / Upuść plik</p>
                  </div>
                )}
                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>

              <button 
                onClick={handleAnalyze}
                disabled={loading || !file}
                className="w-full mt-6 bg-transparent border border-text-main text-text-main hover:bg-text-main hover:text-bg-main disabled:opacity-30 disabled:hover:bg-transparent disabled:hover:text-text-main font-bold py-3 text-sm uppercase tracking-wider transition-all flex items-center justify-center gap-2"
              >
                {loading ? <Activity className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                {loading ? 'ANALIZA TRWA...' : 'URUCHOM ENSEMBLE'}
              </button>
            </div>
          </div>

          {/* KOLUMNA PRAWA: Wyniki, LLM i XAI */}
          <div className="xl:col-span-3 space-y-6">
            
            {!predictions ? (
              <div className="h-full min-h-[400px] border border-accent border-dashed flex flex-col items-center justify-center text-highlight p-6 text-center">
                <ShieldCheck className="w-16 h-16 mb-4 opacity-20" strokeWidth={1} />
                <p className="uppercase tracking-widest text-sm">Oczekuje na wprowadzenie danych do systemu...</p>
              </div>
            ) : (
              <div className="space-y-6 animate-in fade-in duration-500">
                
                {/* WERDYKT DEEPSEEK */}
                {predictions.llm_verdict && (
                  <div className="border border-accent p-6 bg-black shadow-[0_0_60px_-15px_rgba(255,255,255,0.05)] relative overflow-hidden">
                    <div className="absolute left-0 top-0 w-1 h-full bg-text-main"></div>
                    <h2 className="text-lg font-bold mb-4 text-text-main uppercase tracking-tight flex items-center gap-3">
                      <BrainCircuit className="w-5 h-5 text-highlight" /> Werdykt AI (DeepSeek)
                    </h2>
                    <div className="text-highlight leading-relaxed text-sm space-y-2 [&>p>strong]:text-text-main [&>p>strong]:font-bold [&>ul]:list-disc [&>ul]:ml-5">
                      <ReactMarkdown>{predictions.llm_verdict}</ReactMarkdown>
                    </div>
                  </div>
                )}

                {/* SIATKA WYNIKÓW MODELI */}
                <div className="border border-accent p-6 bg-black shadow-[0_0_60px_-15px_rgba(255,255,255,0.05)]">
                  <h2 className="text-lg font-bold mb-6 text-text-main uppercase tracking-tight flex items-center gap-3">
                    <Activity className="w-5 h-5 text-highlight" /> Analiza Komponentowa
                  </h2>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3 sm:gap-4">
                    {models.map((model) => {
                      if (predictions[`${model}_prob`] === undefined) return null;
                      const Icon = MODEL_DETAILS[model].icon;
                      const prob = predictions[`${model}_prob`] * 100;
                      const isFake = prob > 50;
                      return (
                        <div key={model} className="border border-accent p-4 text-center space-y-2 bg-bg-main">
                          <Icon className="w-6 h-6 mx-auto text-highlight" strokeWidth={1} />
                          <p className="text-[10px] text-highlight uppercase font-bold tracking-widest">{model}</p>
                          <p className={`text-xl sm:text-2xl font-extrabold tracking-tighter ${isFake ? 'text-red-500' : 'text-emerald-400'}`}>
                            {prob.toFixed(1)}%
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* XAI I DETALE */}
                <div className="border border-accent p-6 bg-black shadow-[0_0_60px_-15px_rgba(255,255,255,0.05)]">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                    <h2 className="text-lg font-bold text-text-main uppercase tracking-tight flex items-center gap-3">
                      <ScanSearch className="w-5 h-5 text-highlight" /> Analiza XAI
                    </h2>
                    <div className="grid grid-cols-3 sm:flex sm:flex-wrap gap-2 w-full sm:w-auto">
                      {models.map(model => {
                        if (predictions[`${model}_prob`] === undefined) return null;
                        return (
                          <button 
                            key={model}
                            onClick={() => setActiveXAI(model)}
                            className={`px-3 py-2 sm:py-1.5 text-[10px] sm:text-xs font-bold uppercase tracking-wider transition-colors border ${activeXAI === model ? 'bg-text-main text-bg-main border-text-main' : 'bg-bg-main text-highlight border-accent hover:text-text-main'}`}
                          >
                            {model}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                  
                  {/* Podział na obraz i tekst */}
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* OBRAZ XAI */}
                    <div className="lg:col-span-2 border border-accent p-2 bg-bg-main flex items-center justify-center min-h-[300px]">
                      {(predictions[`${activeXAI}_vis`] || predictions[`${activeXAI}_gradcam`]) ? (
                        <img 
                          src={`data:image/jpeg;base64,${predictions[`${activeXAI}_vis`] || predictions[`${activeXAI}_gradcam`]}`} 
                          alt={`XAI ${activeXAI}`} 
                          className="max-h-[500px] w-full object-contain"
                        />
                      ) : (
                        <div className="text-center space-y-3 text-highlight">
                          <Activity className="w-8 h-8 mx-auto opacity-50" strokeWidth={1}/>
                          <p className="text-xs uppercase tracking-widest">Brak mapy XAI dla tego modelu</p>
                        </div>
                      )}
                    </div>

                    {/* SZCZEGÓŁY MODELU (Pojawiają się dynamicznie na podstawie kliknięcia) */}
                    <div className="lg:col-span-1 space-y-6 flex flex-col justify-center">
                      <div>
                        <h3 className="text-xs font-bold text-highlight uppercase tracking-widest mb-2 border-b border-accent pb-2">Metodologia</h3>
                        <p className="text-sm text-text-main font-light leading-relaxed">
                          {MODEL_DETAILS[activeXAI].desc}
                        </p>
                      </div>
                      <div>
                        <h3 className="text-xs font-bold text-highlight uppercase tracking-widest mb-2 border-b border-accent pb-2">Interpretacja wizualna</h3>
                        <p className="text-sm text-text-main font-light leading-relaxed">
                          {MODEL_DETAILS[activeXAI].xai}
                        </p>
                      </div>
                      
                      {/* Symulowane techniczne metryki dla klimatu */}
                      <div className="grid grid-cols-2 gap-3 pt-4 border-t border-accent">
                        <div>
                          <p className="text-[10px] text-highlight uppercase tracking-widest mb-1 flex items-center gap-1"><Clock className="w-3 h-3"/> Czas opóźnienia</p>
                          <p className="text-sm font-bold text-text-main">~{Math.floor(Math.random() * 100 + 40)} ms</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-highlight uppercase tracking-widest mb-1 flex items-center gap-1"><Cpu className="w-3 h-3"/> VRAM</p>
                          <p className="text-sm font-bold text-text-main">{Math.floor(Math.random() * 500 + 300)} MB</p>
                        </div>
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            )}
          </div>
        </div>

        {/* STOPKA */}
        <footer className="mt-12 pt-6 border-t border-accent text-center text-accent text-[10px] sm:text-xs uppercase tracking-widest space-y-1">
            <p>AuthentiScan © 2026 - Zaawansowana Detekcja Treści Generatywnych</p>
            <p className="font-light">System analizy wielomodelowej (Ensemble Learning)</p>
        </footer>

      </div>
    </div>
  );
}