const state = {
  page: 1,
  totalPages: 1,
};

function value(id) {
  return document.getElementById(id).value;
}

function queryParams() {
  const p = new URLSearchParams();
  const scenario = value("scenario");
  const event = value("event");
  const signal_source = value("signal_source");
  const final_success = value("final_success");
  const run_id_query = value("run_id_query").trim();
  const sort_by = value("sort_by");
  const page_size = value("page_size");

  if (scenario) p.set("scenario", scenario);
  if (event) p.set("event", event);
  if (signal_source) p.set("signal_source", signal_source);
  if (final_success) p.set("final_success", final_success);
  if (run_id_query) p.set("run_id_query", run_id_query);
  if (sort_by) p.set("sort_by", sort_by);
  p.set("page", String(state.page));
  p.set("page_size", page_size);

  return p;
}

function addMeta(dl, key, val) {
  const dt = document.createElement("dt");
  dt.textContent = key;
  const dd = document.createElement("dd");
  dd.textContent = String(val);
  dl.appendChild(dt);
  dl.appendChild(dd);
}

function renderCards(items) {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  const tpl = document.getElementById("card-template");
  for (const item of items) {
    const node = tpl.content.cloneNode(true);
    const title = node.querySelector(".card-title");
    const img = node.querySelector(".frame");
    const missing = node.querySelector(".missing");
    const meta = node.querySelector(".meta");
    const raw = node.querySelector(".raw-json");

    title.textContent = item.header_text;
    if (item.image_exists) {
      img.src = item.image_url;
    } else {
      img.style.display = "none";
      missing.style.display = "block";
    }

    addMeta(meta, "run_id", item.run_id);
    addMeta(meta, "route_id", item.route_id);
    addMeta(meta, "frame", item.frame);
    addMeta(meta, "timestamp", item.t);
    addMeta(meta, "status", item.status);
    addMeta(meta, "num_infractions", item.num_infractions);
    addMeta(meta, "event_confidence", item.confidence);
    addMeta(meta, "run_confidence", item.rule_confidence);
    addMeta(meta, "signal_source", item.signal_source);
    addMeta(meta, "final_success", item.final_success);
    addMeta(meta, "image_exists", item.image_exists);
    addMeta(meta, "label_text", item.label_text || "-");
    addMeta(meta, "diagnostics", JSON.stringify(item.diagnostics));

    raw.textContent = JSON.stringify(
      {
        run: item.raw_run,
        event: item.raw_event,
      },
      null,
      2
    );

    grid.appendChild(node);
  }
}

async function loadEvents() {
  const params = queryParams();
  const resp = await fetch(`/api/events?${params.toString()}`);
  const data = await resp.json();

  renderCards(data.items);
  state.totalPages = data.pagination.total_pages;
  state.page = data.pagination.page;

  document.getElementById("page_info").textContent = `Page ${data.pagination.page}/${data.pagination.total_pages}`;
  document.getElementById("count_info").textContent = `Items ${data.items.length} / ${data.pagination.total_count}`;
}

function resetFilters() {
  for (const id of ["scenario", "event", "signal_source", "final_success", "run_id_query", "sort_by", "page_size"]) {
    const el = document.getElementById(id);
    if (!el) continue;
    if (id === "sort_by") el.value = "scenario_run";
    else if (id === "page_size") el.value = "60";
    else el.value = "";
  }
  state.page = 1;
  loadEvents();
}

document.getElementById("apply").addEventListener("click", () => {
  state.page = 1;
  loadEvents();
});

document.getElementById("reset").addEventListener("click", resetFilters);

document.getElementById("prev").addEventListener("click", () => {
  if (state.page > 1) {
    state.page -= 1;
    loadEvents();
  }
});

document.getElementById("next").addEventListener("click", () => {
  if (state.page < state.totalPages) {
    state.page += 1;
    loadEvents();
  }
});

loadEvents();
