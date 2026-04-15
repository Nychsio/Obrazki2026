import { X, Activity, Radar } from 'lucide-react';

export default function ModelModal({ activeModal, modelData, predictions, onClose }) {
  if (!activeModal || !modelData) return null;

  const Icon = modelData.icon;
  const xaiImage = predictions[`${activeModal}_vis`] || predictions[`${activeModal}_gradcam`];
  const prob = (predictions[`${activeModal}_prob`] * 100).toFixed(2);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-4xl overflow-hidden flex flex-col md:flex-row">
        
        {/* Lewa: Obraz XAI */}
        <div className="w-full md:w-1/2 bg-slate-950 p-6 flex flex-col items-center justify-center border-r border-slate-800 relative">
          <span className="absolute top-4 left-4 text-xs font-bold tracking-widest text-slate-500 uppercase">Wizualizacja XAI</span>
          {xaiImage ? (
            <img 
              src={`data:image/jpeg;base64,${xaiImage}`} 
              alt={`XAI ${activeModal}`} 
              className="rounded-lg shadow-lg w-full max-w-sm object-cover border border-slate-700" 
            />
          ) : (
            <div className="text-slate-500 flex flex-col items-center">
              <Activity className="w-12 h-12 mb-2 opacity-50" />
              <p>Brak wizualizacji dla tego modelu</p>
            </div>
          )}
        </div>

        {/* Prawa: Opis i Statystyki */}
        <div className="w-full md:w-1/2 p-8 relative flex flex-col">
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 text-slate-400 hover:text-white bg-slate-800 hover:bg-slate-700 p-2 rounded-full transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
          
          <div className="flex items-center gap-3 mb-6 mt-2">
            <Icon className={`w-8 h-8 ${modelData.color}`} />
            <h2 className="text-2xl font-bold text-white">{modelData.title}</h2>
          </div>
          
          <div className="space-y-6 flex-grow">
            <div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Wynik modelu</h3>
              <div className="text-4xl font-black text-white">
                {prob}% <span className="text-lg text-slate-500 font-normal">szans, że to AI</span>
              </div>
            </div>

            <div className="h-px w-full bg-slate-800" />

            <div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-2">Jak to działa?</h3>
              <p className="text-slate-300 leading-relaxed text-sm">
                {modelData.description}
              </p>
            </div>

            <div className="bg-slate-950/50 border border-slate-800 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-indigo-400 mb-2 flex items-center gap-2">
                <Radar className="w-4 h-4" /> Jak czytać tę mapę?
              </h3>
              <p className="text-slate-300 text-sm">
                {modelData.xai_desc}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}