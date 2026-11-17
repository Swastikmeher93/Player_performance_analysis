// Copies the Python pipeline outputs (../../pressure_outputs) into a JSON module
// that is bundled with the API route, so it works on serverless deploys where
// files outside the app directory are not available at runtime.
import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const appDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = resolve(appDir, "../../pressure_outputs");
const target = resolve(appDir, "app/api/rankings/rankings-data.json");

const files = {
  batting: "batting_rankings.csv",
  bowling: "bowling_rankings.csv",
  audit: "data_integration_audit.csv",
  diagnostics: "context_model_diagnostics.txt",
};

const data = {};
for (const [key, file] of Object.entries(files)) {
  try {
    data[key] = await readFile(resolve(outputDirectory, file), "utf8");
  } catch {
    console.warn(`[bundle-rankings] Missing ${file}; run python3 src/player_rankings.py`);
    data[key] = null;
  }
}

await writeFile(target, JSON.stringify(data));
console.log(`[bundle-rankings] Wrote ${target}`);
