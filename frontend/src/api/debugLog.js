const MAX_ENTRIES = 40;

let entries = [];
const listeners = new Set();
let entryCounter = 0;

function notify() {
  listeners.forEach((listener) => listener(entries));
}

export function getApiDebugEntries() {
  return entries;
}

export function subscribeApiDebugLog(listener) {
  listeners.add(listener);
  listener(entries);
  return () => listeners.delete(listener);
}

export function recordApiDebugEntry(entry) {
  entryCounter += 1;
  entries = [
    {
      id: entryCounter,
      at: new Date().toISOString(),
      ...entry,
    },
    ...entries,
  ].slice(0, MAX_ENTRIES);
  notify();
}

export function annotateLatestDebugEntry(label) {
  if (!entries.length) {
    return;
  }

  entries = [{ ...entries[0], label }, ...entries.slice(1)];
  notify();
}

export function clearApiDebugLog() {
  entries = [];
  notify();
}
