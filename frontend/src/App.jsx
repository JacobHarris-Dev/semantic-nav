import { useState } from "react";
import AppNav from "./components/layout/AppNav";
import ApiPlayground from "./views/ApiPlayground";
import ReaderView from "./views/ReaderView";
import "./index.css";

export default function App() {
  const [activeView, setActiveView] = useState("reader");

  return (
    <div className="app-frame">
      <AppNav activeView={activeView} onViewChange={setActiveView} />
      <main className="app-main">
        {activeView === "reader" ? <ReaderView /> : <ApiPlayground />}
      </main>
    </div>
  );
}
