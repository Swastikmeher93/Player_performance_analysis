import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

type CsvValue = number | string;
type CsvRow = Record<string, CsvValue>;

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    const next = text[index + 1];

    if (character === '"') {
      if (quoted && next === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && next === "\n") index += 1;
      row.push(field);
      if (row.some((cell) => cell.length > 0)) rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

function csvObjects(text: string): CsvRow[] {
  const [header, ...rows] = parseCsv(text);
  if (!header) return [];

  return rows.map((row) =>
    Object.fromEntries(
      header.map((key, index) => {
        const value = row[index]?.trim() ?? "";
        if (value === "") return [key, ""];
        const numeric = Number(value);
        return [key, Number.isFinite(numeric) ? numeric : value];
      }),
    ),
  );
}

export async function GET() {
  try {
    const outputDirectory = resolve(process.cwd(), "../../pressure_outputs");
    const [battingCsv, bowlingCsv, auditCsv, diagnostics] = await Promise.all([
      readFile(resolve(outputDirectory, "batting_rankings.csv"), "utf8"),
      readFile(resolve(outputDirectory, "bowling_rankings.csv"), "utf8"),
      readFile(resolve(outputDirectory, "data_integration_audit.csv"), "utf8"),
      readFile(resolve(outputDirectory, "context_model_diagnostics.txt"), "utf8"),
    ]);

    return Response.json(
      {
        batters: csvObjects(battingCsv),
        bowlers: csvObjects(bowlingCsv),
        audit: csvObjects(auditCsv),
        diagnostics: diagnostics.trim().split("\n").filter(Boolean),
        updatedAt: new Date().toISOString(),
      },
      { headers: { "Cache-Control": "no-store, max-age=0" } },
    );
  } catch {
    return Response.json(
      {
        error:
          "Ranking files are unavailable. Run python3 src/player_rankings.py from the project root, then reload this dashboard.",
      },
      { status: 503, headers: { "Cache-Control": "no-store, max-age=0" } },
    );
  }
}
