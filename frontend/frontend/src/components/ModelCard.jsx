export default function ModelCard({ modelKey, modelData, prob, onClick }) {
  const Icon = modelData.icon;
  const isFake = prob > 50;

  return (
    <button 
      onClick={() => onClick(modelKey)}
      className="bg-slate-950 hover:bg-slate-800 p-5 rounded-xl border border-slate-700 hover:border-indigo-500 transition-all flex flex-col items-center justify-center text-center group cursor-pointer"
    >
      <Icon className={`w-8 h-8 mb-2 ${modelData.color} group-hover:scale-110 transition-transform`} />
      <span className="text-slate-300 font-bold mb-1">{modelData.title.split(' ')[1]}</span>
      <span className={`text-2xl font-bold ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
        {prob.toFixed(1)}%
      </span>
      <span className="text-slate-500 text-xs mt-1">Prawdopodobieństwo AI</span>
      <div className="mt-3 text-xs bg-indigo-900/50 text-indigo-300 px-3 py-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
        Pokaż XAI ➔
      </div>
    </button>
  );
}