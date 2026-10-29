// Per-browser conveniences: who is approving prices, and the store's API key if the server requires one.

function read(key: string): string {
  try {
    return window.localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function write(key: string, value: string): void {
  try {
    if (value) window.localStorage.setItem(key, value);
    else window.localStorage.removeItem(key);
  } catch {
    /* storage unavailable: the value lasts for this page only */
  }
}

let name = read("pricing.user");
let apiKey = read("pricing.apiKey");

export const getUser = () => name;
export const setUser = (v: string) => {
  name = v.trim();
  write("pricing.user", name);
};
export const getApiKey = () => apiKey;
export const setApiKey = (v: string) => {
  apiKey = v.trim();
  write("pricing.apiKey", apiKey);
};
