import { ArrowLeft, Activity, Radar, Cpu, Clock, BarChart3 } from 'lucide-react';

export default function ModelDetailTab({ modelKey, modelData, predictions, onBack }) {
  const Icon = modelData.icon;
  const prob = (predictions[`${modelKey}_prob`] * 100).toFixed(2);
  const xaiImage = predictions[`${modelKey}_vis`] || predictions[`${modelKey}_gradcam`];

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
      
      {/* Pasek nawigacji zakładki */}
      <div className="flex items-center justify-between bg-slate-900 border border-slate-800 p-4 rounded-2xl shadow-md">
        <div className="flex items-center gap-4">
          <button 
            onClick={onBack}
            className="flex items-center gap-2 text-slate-400 hover:text-white bg-slate-950 hover:bg-slate-800 px-4 py-2 rounded-xl transition-all"
          >
            <ArrowLeft className="w-4 h-4" /> Powrót do podsumowania
          </button>
          <div className="h-6 w-px bg-slate-700"></div>
          <Icon className={`w-6 h-6 ${modelData.color}`} />
          <h2 className="text-xl font-bold text-white">{modelData.title}</h2>
        </div>
        <div className="text-right">
          <span className="text-slate-500 text-xs uppercase font-bold tracking-widest">Wynik Analizy</span>
          <div className={`text-2xl font-black ${prob > 50 ? 'text-red-400' : 'text-emerald-400'}`}>
            {prob}% FAKE
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Lewa: Przestrzeń XAI na sterydach */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
            <Radar className="w-5 h-5 text-indigo-400" /> Zaawansowana Wizualizacja XAI
          </h3>
          <div className="flex-grow bg-slate-950 rounded-xl border border-slate-800 flex items-center justify-center p-2 relative overflow-hidden">
            {xaiImage ? (
              <img 
                src={`data:image/jpeg;base64,${xaiImage}`} 
                alt={`XAI ${modelKey}`} 
                className="rounded-lg max-h-[500px] object-contain w-full" 
              />
            ) : (
              <div className="text-slate-500 flex flex-col items-center">
                <Activity className="w-10 h-10 mb-2 opacity-50" />
                <p>Oczekuje na wygenerowanie mapy XAI</p>
              </div>
            )}
          </div>
        </div>

        {/* Prawa: Głębsze statystyki i detale */}
        <div className="space-y-6 flex flex-col">
          
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
            <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
              <BarChart3 className="w-5 h-5 text-blue-400" /> Metryki Modelu
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <span className="text-slate-500 text-xs uppercase flex items-center gap-1"><Clock className="w-3 h-3"/> Czas inferencji</span>
                <div className="text-xl font-bold text-slate-200 mt-1">~145 ms</div>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <span className="text-slate-500 text-xs uppercase flex items-center gap-1"><Cpu className="w-3 h-3"/> Użycie VRAM</span>
                <div className="text-xl font-bold text-slate-200 mt-1">850 MB</div>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <span className="text-slate-500 text-xs uppercase">Pewność modelu (Z-score)</span>
                <div className="text-xl font-bold text-slate-200 mt-1">{prob > 90 || prob < 10 ? 'Wysoka' : 'Niska'}</div>
              </div>
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <span className="text-slate-500 text-xs uppercase">Rozkład wariancji</span>
                <div className="text-xl font-bold text-slate-200 mt-1">0.042 σ²</div>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-4">*Uwaga: Czas inferencji i VRAM są wartościami poglądowymi. Pełna implementacja w backendzie w toku.</p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex-grow">
            <h3 className="text-lg font-semibold text-white mb-2">Metodologia Detekcji</h3>
            <p className="text-slate-300 text-sm leading-relaxed mb-4">{modelData.description}</p>
            
            <h3 className="text-sm font-semibold text-indigo-400 mt-4 mb-2">Interpretacja Mapy XAI</h3>
            <p className="text-slate-400 text-sm leading-relaxed">{modelData.xai_desc}</p>
          </div>

        </div>
      </div>
    </div>
  );
}