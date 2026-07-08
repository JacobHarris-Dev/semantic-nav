import { useCallback, useState } from "react";

export function useApiCall() {
  const [loading, setLoading] = useState(false);
  const [lastRequest, setLastRequest] = useState(null);
  const [lastResponse, setLastResponse] = useState(null);

  const run = useCallback(async (label, requestFn) => {
    setLoading(true);
    setLastRequest({ label, at: new Date().toISOString() });

    const result = await requestFn();
    setLastResponse({ label, ...result });
    setLoading(false);
    return result;
  }, []);

  return { loading, lastRequest, lastResponse, run };
}
