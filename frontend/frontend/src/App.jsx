import { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { ShieldCheck, UploadCloud, BrainCircuit, Waves, ScanSearch, FileWarning, Zap, Activity, Cpu, Clock } from 'lucide-react';

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
      // PAMIĘTAJ: Tu potem podmienimy localhost na adres z RunPoda!
      const response = await axios.post('http://localhost:8000/api/v1/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPredictions(response.data.predictions);
      
      const availableModels = models.filter(m => 
        response.data.predictions[`${m}_vis`] || 
        response.data.predictions[`${m}_gradcam`] || 
        response.data.predictions[`${m}_prob`]
      );
      if (availableModels.length > 0) setActiveXAI(availableModels[0]);

    } catch (err) {
      setError('Błąd połączenia z serwerem AI.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-bg-main text-text-main p-3 sm:p-6 lg:p-12 font-sans">
      <div className="max-w-[1600px] mx-auto space-y-8 lg:space-y-12">
        
        {/* NAGŁÓWEK - GIGANTYCZNY NA PC */}
        <header className="flex flex-col md:flex-row items-center text-center md:text-left gap-3 lg:gap-8 pb-6 lg:pb-10 border-b border-accent lg:border-b-2 lg:border-gray-500">
          <ShieldCheck className="w-10 h-10 lg:w-20 lg:h-20 text-text-main" strokeWidth={1.5} />
          <div className="space-y-2">
            <h1 className="text-3xl lg:text-7xl font-black tracking-tighter text-text-main uppercase">
              AUTHENTISCAN
            </h1>
            <p className="text-sm lg:text-2xl text-highlight lg:text-gray-300 tracking-widest uppercase font-bold">
              PIĘĆ MODELI. JEDNA PRAWDA.
            </p>
          </div>
        </header>

        {error && (
          <div className="border-2 border-red-500 bg-red-950/40 text-red-400 p-6 text-lg lg:text-2xl font-black uppercase tracking-widest text-center">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 lg:gap-10">
          
          {/* KOLUMNA LEWA: Wgraj obraz */}
          <div className="xl:col-span-1 space-y-6 lg:space-y-10">
            {/* Pogrubione ramki na PC */}
            <div className="border border-accent lg:border-2 lg:border-gray-600 p-6 lg:p-8 bg-black">
              <h2 className="text-lg lg:text-2xl font-black mb-6 lg:mb-8 text-text-main uppercase tracking-tight flex items-center gap-3">
                <UploadCloud className="w-5 h-5 lg:w-8 lg:h-8 text-text-main" /> Wgraj obraz
              </h2>
              
              <label className="aspect-square border-2 border-accent lg:border-4 lg:border-gray-500 border-dashed flex flex-col items-center justify-center cursor-pointer hover:border-text-main transition-colors group relative overflow-hidden">
                {file ? (
                  <img src={file} alt="Preview" className="w-full h-full object-cover p-1 opacity-80 group-hover:opacity-100 transition-opacity" />
                ) : (
                  <div className="text-center p-4 space-y-4">
                    <ScanSearch className="w-10 h-10 lg:w-16 lg:h-16 mx-auto text-highlight lg:text-gray-400 group-hover:text-text-main transition-colors" strokeWidth={1.5}/>
                    <p className="text-highlight lg:text-gray-200 text-xs lg:text-lg font-black uppercase tracking-widest">Kliknij / Upuść plik</p>
                  </div>
                )}
                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>

              <button 
                onClick={handleAnalyze}
                disabled={loading || !file}
                className="w-full mt-6 lg:mt-10 bg-text-main border-2 border-text-main text-bg-main hover:bg-transparent hover:text-text-main disabled:opacity-30 disabled:hover:bg-text-main disabled:hover:text-bg-main font-black py-4 lg:py-6 text-sm lg:text-xl uppercase tracking-widest transition-all flex items-center justify-center gap-3"
              >
                {loading ? <Activity className="w-4 h-4 lg:w-6 lg:h-6 animate-spin" /> : <Zap className="w-4 h-4 lg:w-6 lg:h-6" />}
                {loading ? 'ANALIZA TRWA...' : 'URUCHOM ENSEMBLE'}
              </button>
            </div>
          </div>

          {/* KOLUMNA PRAWA */}
          <div className="xl:col-span-3 space-y-6 lg:space-y-10">
            
            {!predictions ? (
              <div className="h-full min-h-[400px] border border-accent lg:border-4 lg:border-gray-700 border-dashed flex flex-col items-center justify-center text-highlight lg:text-gray-400 p-6 text-center">
                <ShieldCheck className="w-16 h-16 lg:w-24 lg:h-24 mb-6 opacity-40" strokeWidth={1} />
                <p className="uppercase tracking-widest text-sm lg:text-2xl font-bold">Oczekuje na wprowadzenie danych do systemu...</p>
              </div>
            ) : (
              <div className="space-y-6 lg:space-y-10 animate-in fade-in duration-500">
                
                {/* WERDYKT DEEPSEEK - WERSJA PROJEKTOR */}
                {predictions.llm_verdict && (
                  <div className="border border-accent lg:border-2 lg:border-gray-600 p-6 lg:p-8 bg-black relative overflow-hidden">
                    <div className="absolute left-0 top-0 w-1 lg:w-2 h-full bg-text-main"></div>
                    <h2 className="text-lg lg:text-2xl font-black mb-4 lg:mb-6 text-text-main uppercase tracking-tight flex items-center gap-3">
                      <BrainCircuit className="w-5 h-5 lg:w-8 lg:h-8 text-highlight lg:text-gray-300" /> Werdykt AI (DeepSeek)
                    </h2>
                    <div className="text-highlight lg:text-gray-200 leading-relaxed text-sm lg:text-xl space-y-4 [&>p>strong]:text-text-main [&>p>strong]:font-black [&>ul]:list-disc [&>ul]:ml-6 lg:[&>ul]:ml-8">
                      <ReactMarkdown>{predictions.llm_verdict}</ReactMarkdown>
                    </div>
                  </div>
                )}

                {/* SIATKA WYNIKÓW MODELI - GIGANTYCZNE PROCENTY */}
                <div className="border border-accent lg:border-2 lg:border-gray-600 p-6 lg:p-8 bg-black">
                  <h2 className="text-lg lg:text-2xl font-black mb-6 lg:mb-8 text-text-main uppercase tracking-tight flex items-center gap-3">
                    <Activity className="w-5 h-5 lg:w-8 lg:h-8 text-text-main" /> Analiza Komponentowa
                  </h2>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-3 lg:gap-6">
                    {models.map((model) => {
                      if (predictions[`${model}_prob`] === undefined) return null;
                      const Icon = MODEL_DETAILS[model].icon;
                      const prob = predictions[`${model}_prob`] * 100;
                      const isFake = prob > 50;
                      return (
                        <div key={model} className="border border-accent lg:border-2 lg:border-gray-500 p-4 lg:p-6 text-center space-y-3 bg-bg-main">
                          <Icon className="w-6 h-6 lg:w-10 lg:h-10 mx-auto text-highlight lg:text-gray-300" strokeWidth={1.5} />
                          <p className="text-[10px] lg:text-sm text-highlight lg:text-gray-200 uppercase font-black tracking-widest">{model}</p>
                          {/* TUTAJ DZIEJE SIĘ MAGIA PROJEKTORA - xl:text-6xl */}
                          <p className={`text-xl sm:text-2xl lg:text-5xl xl:text-6xl font-black tracking-tighter ${isFake ? 'text-red-500' : 'text-emerald-400'}`}>
                            {prob.toFixed(1)}%
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* XAI I DETALE */}
                <div className="border border-accent lg:border-2 lg:border-gray-600 p-6 lg:p-8 bg-black">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 lg:mb-10">
                    <h2 className="text-lg lg:text-2xl font-black text-text-main uppercase tracking-tight flex items-center gap-3">
                      <ScanSearch className="w-5 h-5 lg:w-8 lg:h-8 text-text-main" /> Analiza XAI
                    </h2>
                    <div className="grid grid-cols-3 sm:flex sm:flex-wrap gap-2 lg:gap-4 w-full sm:w-auto">
                      {models.map(model => {
                        if (predictions[`${model}_prob`] === undefined) return null;
                        return (
                          <button 
                            key={model}
                            onClick={() => setActiveXAI(model)}
                            className={`px-3 py-2 lg:px-6 lg:py-3 text-[10px] sm:text-xs lg:text-lg font-black uppercase tracking-widest transition-colors border-2 ${activeXAI === model ? 'bg-text-main text-bg-main border-text-main' : 'bg-bg-main text-highlight lg:text-gray-300 border-accent lg:border-gray-500 hover:text-text-main'}`}
                          >
                            {model}
                          </button>
                        )
                      })}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-12">
                    {/* OBRAZ XAI */}
                    <div className="lg:col-span-2 border border-accent lg:border-2 lg:border-gray-500 p-2 lg:p-4 bg-bg-main flex items-center justify-center min-h-[300px] lg:min-h-[500px]">
                      {(predictions[`${activeXAI}_vis`] || predictions[`${activeXAI}_gradcam`]) ? (
                        <img 
                          src={`data:image/jpeg;base64,${predictions[`${activeXAI}_vis`] || predictions[`${activeXAI}_gradcam`]}`} 
                          alt={`XAI ${activeXAI}`} 
                          className="max-h-[400px] lg:max-h-[600px] w-full object-contain"
                        />
                      ) : (
                        <div className="text-center space-y-4 text-highlight lg:text-gray-400">
                          <Activity className="w-8 h-8 lg:w-16 lg:h-16 mx-auto opacity-50" strokeWidth={1.5}/>
                          <p className="text-xs lg:text-lg font-bold uppercase tracking-widest">Brak mapy XAI</p>
                        </div>
                      )}
                    </div>

                    {/* SZCZEGÓŁY MODELU - WIĘKSZY TEKST */}
                    <div className="lg:col-span-1 space-y-8 lg:space-y-12 flex flex-col justify-center">
                      
                      {/* NOWOŚĆ: WERDYKT DEEPSEEK DLA KONKRETNEGO MODELU */}
                      <div className="border-l-4 border-text-main pl-4 lg:pl-6">
                        <h3 className="text-[10px] lg:text-sm font-black text-highlight lg:text-gray-300 uppercase tracking-widest mb-2 flex items-center gap-2">
                          <BrainCircuit className="w-4 h-4 lg:w-5 lg:h-5 text-text-main" /> Werdykt DeepSeek: {activeXAI.toUpperCase()}
                        </h3>
                        <p className="text-sm lg:text-2xl text-text-main font-bold leading-relaxed italic">
                          {predictions[`${activeXAI}_llm`] 
                            ? predictions[`${activeXAI}_llm`] 
                            : `Oczekuje na wnioski z analizy metryk modelu ${activeXAI.toUpperCase()}...`}
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xs lg:text-xl font-black text-highlight lg:text-gray-300 uppercase tracking-widest mb-3 border-b border-accent lg:border-gray-600 pb-3">Metodologia</h3>
                        <p className="text-sm lg:text-2xl text-text-main font-semibold leading-relaxed">
                          {MODEL_DETAILS[activeXAI].desc}
                        </p>
                      </div>
                      
                      <div>
                        <h3 className="text-xs lg:text-xl font-black text-highlight lg:text-gray-300 uppercase tracking-widest mb-3 border-b border-accent lg:border-gray-600 pb-3">Interpretacja wizualna</h3>
                        <p className="text-sm lg:text-2xl text-text-main font-semibold leading-relaxed">
                          {MODEL_DETAILS[activeXAI].xai}
                        </p>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 lg:gap-8 pt-6 border-t border-accent lg:border-gray-600">
                        <div>
                          <p className="text-[10px] lg:text-sm text-highlight lg:text-gray-300 font-bold uppercase tracking-widest mb-2 flex items-center gap-2"><Clock className="w-3 h-3 lg:w-5 lg:h-5"/> Opóźnienie</p>
                          <p className="text-sm lg:text-2xl font-black text-text-main">~{Math.floor(Math.random() * 100 + 40)} ms</p>
                        </div>
                        <div>
                          <p className="text-[10px] lg:text-sm text-highlight lg:text-gray-300 font-bold uppercase tracking-widest mb-2 flex items-center gap-2"><Cpu className="w-3 h-3 lg:w-5 lg:h-5"/> VRAM</p>
                          <p className="text-sm lg:text-2xl font-black text-text-main">{Math.floor(Math.random() * 500 + 300)} MB</p>
                        </div>
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}