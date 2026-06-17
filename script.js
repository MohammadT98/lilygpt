// LilyBench demo — renders the generation gallery, understanding cards and
// result tables from the three JSON files emitted by build.py. Vanilla ES6.

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const k of kids.flat()) node.append(k?.nodeType ? k : document.createTextNode(k ?? ""));
  return node;
};
const fetchJSON = async (path) => (await fetch(path)).json();

// Mutopia composer codes ("CouperinF") read better split ("Couperin F").
const prettify = (s) => String(s).replace(/([a-z])([A-Z])/g, "$1 $2");

// ----------------------------------------------------------------- generation
const modelBadges = (m) => {
  const badges = [
    el("span", { className: "badge ok" }, "compiles ✓"),
    el("span", { className: "badge" }, `${m.notes} notes`),
  ];
  if (m.scale_drift) badges.push(el("span", { className: "badge warn" }, "scale drift"));
  return el("div", { className: "badges" }, badges);
};

const modelPanel = (idx, m) => {
  const score = el("div", { className: "score" });
  score.append(el("object", { type: "image/svg+xml", data: m.score, className: "svg" }));

  const audio = m.audio
    ? el("audio", { controls: true, preload: "none", src: m.audio })
    : el("p", { className: "noaudio" }, "no audio");

  const code = el("details", {},
    el("summary", {}, "LilyPond source"),
    el("pre", {}, el("code", {}, m.ly_code)));

  return el("article", { className: "panel" },
    el("header", { className: "panel-head" },
      el("h4", {}, m.name), modelBadges(m)),
    score, audio, code);
};

const renderPrompt = (gallery, prompt) => {
  gallery.replaceChildren();
  const meta = prompt.metadata;
  const block = [
    ["composer", meta.composer], ["period", meta.period], ["form", meta.form],
    ["ensemble", meta.ensemble], ["part", meta.part],
  ].filter(([, v]) => v).map(([k, v]) => `%% ${k}: ${v}`).join("\n");

  gallery.append(
    el("div", { className: "promptcard" },
      el("div", { className: "metablock" },
        el("pre", {}, `%% === METADATA ===\n${block}\n%% === END METADATA ===`)),
      el("p", { className: "userprompt" }, prompt.user_prompt)),
    el("div", { className: "grid4" }, prompt.models.map((m) => modelPanel(prompt.idx, m))));
};

const buildGeneration = async () => {
  const prompts = await fetchJSON("data/samples.json");
  const picker = $("#prompt-picker");
  const gallery = $("#generation-gallery");
  const select = (i) => {
    [...picker.children].forEach((b, j) => b.classList.toggle("active", i === j));
    renderPrompt(gallery, prompts[i]);
  };
  prompts.forEach((p, i) => {
    const label = `${p.metadata.composer} · ${p.metadata.form}`;
    picker.append(el("button", { className: "pill", onclick: () => select(i) }, label));
  });
  select(0);
};

// --------------------------------------------------------------- understanding
const answerChip = (task, p) => {
  const label = task === "composer_recognition" ? prettify(p.answer) : p.answer;
  return el("div", { className: `pred ${p.correct ? "right" : "wrong"}` },
    el("span", { className: "pred-model" }, p.name),
    el("span", { className: "pred-ans" }, label),
    el("span", { className: "pred-mark" }, p.correct ? "✓" : "✗"));
};

const optionChips = (item) => {
  if (!item.options) {
    return el("div", { className: "gold-only" }, "Gold: ", el("strong", {}, item.gold ?? "—"));
  }
  const pretty = (o) => (item.task === "composer_recognition" ? prettify(o) : o);
  return el("div", { className: "options" },
    item.options.map((o) => el("span",
      { className: "opt" + (o === item.gold ? " gold" : "") }, pretty(o))));
};

const taskCard = (item) => {
  const correct = item.predictions.filter((p) => p.correct).length;
  const verdict = correct === 4 ? "all" : correct === 0 ? "none" : "some";
  return el("article", { className: `ucard verdict-${verdict}` },
    el("header", { className: "ucard-head" },
      el("span", { className: "cat" }, item.category),
      el("h4", {}, item.task),
      el("span", { className: "scoretag" }, `${correct}/4 correct`)),
    el("p", { className: "instruction" }, item.instruction),
    el("details", {},
      el("summary", {}, "LilyPond input (excerpt)"),
      el("pre", {}, el("code", {}, item.input_excerpt))),
    optionChips(item),
    el("div", { className: "preds" }, item.predictions.map((p) => answerChip(item.task, p))));
};

const buildUnderstanding = async () => {
  const items = await fetchJSON("data/understanding.json");
  const list = $("#understanding-list");
  let current = null;
  for (const item of items) {
    if (item.category !== current) {
      current = item.category;
      list.append(el("h3", { className: "cathead" }, `${current} reasoning`));
    }
    list.append(taskCard(item));
  }
};

// -------------------------------------------------------------------- tables
const bestIndices = (values, dir) => {
  const best = dir === "down" ? Math.min(...values) : Math.max(...values);
  return new Set(values.map((v, i) => (v === best ? i : -1)).filter((i) => i >= 0));
};

const genTable = (g) => {
  const head = el("tr", {}, el("th", {}, "Regime"), el("th", {}, "Model"),
    g.columns.map((c) => el("th", {}, c.label, el("span", { className: "dir" }, c.dir === "down" ? " ↓" : " ↑"))));
  const rows = [];
  for (const regime of g.regimes) {
    const best = {};
    g.columns.forEach((c) => (best[c.key] = bestIndices(regime.rows.map((r) => r[c.key]), c.dir)));
    regime.rows.forEach((row, ri) => {
      const cells = [];
      if (ri === 0) cells.push(el("td", { className: "regime", rowSpan: regime.rows.length }, regime.name));
      cells.push(el("td", {}, row.model));
      g.columns.forEach((c) => cells.push(
        el("td", { className: best[c.key].has(ri) ? "best" : "" },
          c.key === "comp" ? row[c.key].toFixed(1) : row[c.key].toFixed(c.key.startsWith("fmd") ? 3 : 2))));
      rows.push(el("tr", { className: ri === 0 ? "regime-start" : "" }, cells));
    });
  }
  return el("table", { className: "resulttable" },
    el("thead", {}, head), el("tbody", {}, rows));
};

const undTable = (u) => {
  const head = el("tr", {}, el("th", {}, "Category"), el("th", {}, "Task"),
    u.models.map((m) => el("th", {}, m)), el("th", {}, "Metric"));
  const rows = [];
  let current = null;
  for (const r of u.rows) {
    const best = bestIndices(r.scores, "up");
    const cells = [];
    cells.push(el("td", { className: "cat-cell" }, r.category !== current ? r.category : ""));
    current = r.category;
    cells.push(el("td", { className: "task-cell" }, r.task));
    r.scores.forEach((s, i) => cells.push(el("td", { className: best.has(i) ? "best" : "" }, s.toFixed(3))));
    cells.push(el("td", { className: "metric-cell" }, r.metric));
    rows.push(el("tr", {}, cells));
  }
  for (const agg of u.aggregate) {
    const best = bestIndices(agg.scores, "up");
    rows.push(el("tr", { className: "agg" },
      el("td", { colSpan: 2 }, agg.label),
      agg.scores.map((s, i) => el("td", { className: best.has(i) ? "best" : "" }, s.toFixed(3))),
      el("td", {})));
  }
  return el("table", { className: "resulttable" }, el("thead", {}, head), el("tbody", {}, rows));
};

const buildTables = async () => {
  const r = await fetchJSON("data/results.json");
  $("#gen-table").append(genTable(r.generation),
    el("p", { className: "note" }, r.generation.caption));
  $("#und-table").append(undTable(r.understanding),
    el("p", { className: "note" }, r.understanding.caption));
};

// -------------------------------------------------------------------- boot
const main = async () => {
  try {
    await Promise.all([buildGeneration(), buildUnderstanding(), buildTables()]);
  } catch (err) {
    console.error(err);
    document.body.append(el("p", { className: "note" }, `Failed to load demo data: ${err}`));
  }
};
main();
