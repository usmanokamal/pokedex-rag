const TYPES = [
  "Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting", "Fire", "Flying",
  "Ghost", "Grass", "Ground", "Ice", "Normal", "Poison", "Psychic", "Rock",
  "Steel", "Water"
];

const typeContainer = document.getElementById("type-chips");
const queryInput = document.getElementById("query");
const form = document.getElementById("chat-form");
const buildBtn = document.getElementById("build-query");
const metaPill = document.getElementById("meta-pill");
const answerEl = document.getElementById("answer");
const resultsEl = document.getElementById("results");

const minHp = document.getElementById("min-hp");
const minAttack = document.getElementById("min-attack");
const minDefense = document.getElementById("min-defense");
const generation = document.getElementById("generation");
const legendaryOnly = document.getElementById("legendary-only");
const nonLegendaryOnly = document.getElementById("non-legendary-only");

const activeTypes = new Set();

function renderTypeChips() {
  TYPES.forEach(type => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip";
    btn.textContent = type;
    btn.addEventListener("click", () => {
      if (activeTypes.has(type)) {
        activeTypes.delete(type);
        btn.classList.remove("active");
      } else {
        activeTypes.add(type);
        btn.classList.add("active");
      }
    });
    typeContainer.appendChild(btn);
  });
}

function composeQuery() {
  const parts = [];

  if (activeTypes.size) {
    parts.push(`${Array.from(activeTypes).join(" ")} type`);
  }

  if (minHp.value) {
    parts.push(`HP >= ${minHp.value}`);
  }
  if (minAttack.value) {
    parts.push(`Attack >= ${minAttack.value}`);
  }
  if (minDefense.value) {
    parts.push(`Defense >= ${minDefense.value}`);
  }
  if (generation.value) {
    parts.push(`Gen ${generation.value}`);
  }

  if (legendaryOnly.checked && !nonLegendaryOnly.checked) {
    parts.push("legendary");
  }
  if (nonLegendaryOnly.checked && !legendaryOnly.checked) {
    parts.push("non legendary");
  }

  return parts.join(" ");
}

function setMeta(meta, usedLlm) {
  const strategy = meta?.strategy ? `Mode: ${meta.strategy}` : "Mode: vector";
  const llm = usedLlm ? "Gemini" : "Deterministic";
  metaPill.textContent = `${strategy} · ${llm}`;
}

function renderResults(results) {
  resultsEl.innerHTML = "";
  if (!results || results.length === 0) {
    return;
  }

  results.forEach(item => {
    const card = document.createElement("div");
    card.className = "result-card";

    const title = document.createElement("h3");
    title.textContent = `${item.name} #${item.id}`;

    const tags = document.createElement("div");
    tags.className = "tag-row";
    const type2 = item.type_2 && item.type_2.trim() ? item.type_2 : "None";
    [item.type_1, type2].forEach(t => {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = t;
      tags.appendChild(tag);
    });
    if (item.legendary) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "Legendary";
      tags.appendChild(tag);
    }

    const stats = document.createElement("div");
    stats.className = "stat-row";
    stats.innerHTML = `
      <span>HP: ${item.hp}</span>
      <span>Attack: ${item.attack}</span>
      <span>Defense: ${item.defense}</span>
      <span>Sp. Atk: ${item.sp_atk}</span>
      <span>Sp. Def: ${item.sp_def}</span>
      <span>Speed: ${item.speed}</span>
      <span>Total: ${item.total}</span>
      <span>Gen: ${item.generation}</span>
    `;

    card.appendChild(title);
    card.appendChild(tags);
    card.appendChild(stats);
    resultsEl.appendChild(card);
  });
}

async function submitQuery(query) {
  answerEl.textContent = "Searching…";
  resultsEl.innerHTML = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 8 })
    });

    if (!response.ok) {
      throw new Error("Request failed");
    }

    const data = await response.json();
    answerEl.textContent = data.answer || "No answer returned.";
    setMeta(data.meta, data.used_llm);
    renderResults(data.results);
  } catch (err) {
    answerEl.textContent = "Something went wrong. Check the server logs.";
    metaPill.textContent = "Error";
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    queryInput.focus();
    return;
  }
  submitQuery(query);
});

buildBtn.addEventListener("click", () => {
  const query = composeQuery();
  queryInput.value = query;
  queryInput.focus();
});

legendaryOnly.addEventListener("change", () => {
  if (legendaryOnly.checked) {
    nonLegendaryOnly.checked = false;
  }
});

nonLegendaryOnly.addEventListener("change", () => {
  if (nonLegendaryOnly.checked) {
    legendaryOnly.checked = false;
  }
});

renderTypeChips();
