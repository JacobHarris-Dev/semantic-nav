import { useEffect, useState } from "react";
import { subscribeApiDebugLog } from "../api/debugLog";

export function useApiDebugLog() {
  const [entries, setEntries] = useState([]);

  useEffect(() => subscribeApiDebugLog(setEntries), []);

  return entries;
}
