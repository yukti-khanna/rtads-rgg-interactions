import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = path.resolve(process.argv[2] ?? "final_figure_package");
const payload = JSON.parse(await fs.readFile(path.join(root, "tables_payload.json"), "utf8"));
const workbook = Workbook.create();

function colName(n) {
  let s = "";
  while (n > 0) { n--; s = String.fromCharCode(65 + (n % 26)) + s; n = Math.floor(n / 26); }
  return s;
}

function columnWidth(name) {
  const key = name.toLowerCase();
  if (key.includes("sequence")) return 34;
  if (["pair_id", "pair_display_name", "description", "definition", "entry_rule", "reason", "status", "source", "notes", "method"].some(x => key.includes(x))) return 28;
  if (key.includes("comparison") || key.includes("evaluation") || key.includes("record_type")) return 22;
  return 14;
}

for (const item of payload) {
  const sheet = workbook.worksheets.add(item.sheet);
  sheet.showGridLines = false;
  sheet.tabColor = "#1F4E78";
  let row = 1;
  let widest = 1;

  sheet.getRange(`A${row}`).values = [[`Supplementary Table S11 — ${item.sheet}`]];
  sheet.getRange(`A${row}`).format = {font: {name: "Arial", size: 14, bold: true, color: "#17324D"}};
  row += 1;
  sheet.getRange(`A${row}`).values = [[item.description ?? ""]];
  sheet.getRange(`A${row}`).format = {font: {name: "Arial", size: 9, italic: true, color: "#555555"}, wrapText: true};
  row += 2;

  for (const section of item.sections) {
    const columns = section.columns ?? [];
    if (!columns.length) continue;
    widest = Math.max(widest, columns.length);
    const last = colName(columns.length);
    sheet.mergeCells(`A${row}:${last}${row}`);
    sheet.getRange(`A${row}`).values = [[section.title]];
    sheet.getRange(`A${row}:${last}${row}`).format = {
      fill: "#D9EAF7",
      font: {name: "Arial", size: 10, bold: true, color: "#17324D"},
      verticalAlignment: "center"
    };
    row += 1;
    if (section.note) {
      sheet.mergeCells(`A${row}:${last}${row}`);
      sheet.getRange(`A${row}`).values = [[section.note]];
      sheet.getRange(`A${row}:${last}${row}`).format = {font: {name: "Arial", size: 9, italic: true, color: "#555555"}, wrapText: true};
      row += 1;
    }
    const headerRow = row;
    sheet.getRange(`A${row}:${last}${row}`).values = [columns];
    sheet.getRange(`A${row}:${last}${row}`).format = {
      fill: "#1F4E78",
      font: {name: "Arial", size: 9, bold: true, color: "#FFFFFF"},
      wrapText: true,
      verticalAlignment: "center",
      borders: {preset: "all", style: "thin", color: "#D9E2F3"}
    };
    row += 1;
    if (section.rows?.length) {
      const endRow = row + section.rows.length - 1;
      sheet.getRange(`A${row}:${last}${endRow}`).values = section.rows;
      sheet.getRange(`A${row}:${last}${endRow}`).format = {
        font: {name: "Arial", size: 9, color: "#000000"},
        verticalAlignment: "center",
        wrapText: true,
        borders: {preset: "all", style: "thin", color: "#E7EAF0"}
      };
      for (let c = 0; c < columns.length; c++) {
        const key = columns[c].toLowerCase();
        const letter = colName(c + 1);
        if (key.includes("fraction") && !key.includes("percent")) sheet.getRange(`${letter}${row}:${letter}${endRow}`).format.numberFormat = "0.000";
        else if (key.includes("percent")) sheet.getRange(`${letter}${row}:${letter}${endRow}`).format.numberFormat = "0.0";
        else if (key.includes("kd_um") || key.includes("kd_error")) sheet.getRange(`${letter}${row}:${letter}${endRow}`).format.numberFormat = "0.000";
        else if (["auroc", "estimate", "ci95_low", "ci95_high", "balanced_accuracy", "sensitivity", "specificity", "threshold"].includes(key)) sheet.getRange(`${letter}${row}:${letter}${endRow}`).format.numberFormat = "0.000";
      }
      row = endRow + 2;
    } else {
      sheet.getRange(`A${row}`).values = [["No eligible rows in the current master dataset."]];
      sheet.getRange(`A${row}`).format = {font: {name: "Arial", size: 9, italic: true, color: "#777777"}};
      row += 2;
    }
    sheet.getRange(`A${headerRow}:${last}${headerRow}`).format.rowHeight = 30;
    for (let c = 0; c < columns.length; c++) sheet.getRange(`${colName(c + 1)}:${colName(c + 1)}`).format.columnWidth = columnWidth(columns[c]);
  }
  sheet.freezePanes.freezeRows(4);
  sheet.getRange(`A1:${colName(widest)}${Math.max(1,row-1)}`).format.autofitRows();
}

workbook.recalculate();
const errors = await workbook.inspect({kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: {useRegex: true, maxResults: 100}, summary: "final formula error scan"});
if (errors?.matches?.length) throw new Error(`Formula errors found: ${JSON.stringify(errors.matches)}`);
const overview = await workbook.inspect({kind: "workbook,sheet,region", maxChars: 20000, tableMaxRows: 6, tableMaxCols: 8});
await fs.writeFile(path.join(root, "Supplementary_Table_S11.xlsx.inspect.ndjson"), overview.ndjson ?? JSON.stringify(overview, null, 2));
const preview = await workbook.render({sheetName: "NMR outcomes and Kd", range: "A1:V18", scale: 1, format: "png"});
await fs.writeFile(path.join(root, "Supplementary_Table_S11_preview.png"), new Uint8Array(await preview.arrayBuffer()));
for (const name of ['pH simulation comparison','pH predictor comparison']) {
 const preview = await workbook.render({sheetName:name,range:'A4:G12',scale:1,format:'png'});
 await fs.writeFile(path.join(root,name.replaceAll(' ','_')+'_preview.png'),new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(root, "Supplementary_Table_S11.xlsx"));
