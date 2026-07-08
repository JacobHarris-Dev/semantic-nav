const VIEWS = [
  { id: "reader", label: "Reader" },
  { id: "playground", label: "API Playground" },
];

export default function AppNav({ activeView, onViewChange }) {
  return (
    <nav className="app-nav" aria-label="Main views">
      <div className="app-nav-brand">Semantic Nav</div>
      <div className="app-nav-tabs" role="tablist">
        {VIEWS.map((view) => (
          <button
            key={view.id}
            type="button"
            role="tab"
            aria-selected={activeView === view.id}
            className={`app-nav-tab${activeView === view.id ? " active" : ""}`}
            onClick={() => onViewChange(view.id)}
          >
            {view.label}
          </button>
        ))}
      </div>
    </nav>
  );
}

export { VIEWS };
