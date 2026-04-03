/* ============================================================
   Golazo! — script.js
   Goals data → FastAPI (/players/{nation}) from goals.csv
   Visual metadata → NATION_META  |  Supplementary → PLAYER_META
   ============================================================ */

// ── NATION METADATA  (colours, flag, card hint) ──────────────
// "hint" is just a display cache for the landing card — goals
// always come live from the API when a nation is clicked.
const NATION_META = {
  Argentina: {
    flagCode: 'ar',
    gradient: 'linear-gradient(135deg, #6db8e8 0%, #1a6fb5 55%, #0a3a7a 100%)',
    primary:  '#75AADB',
    glow:     'rgba(100,170,230,0.45)',
    hint:     { name: 'Lionel Messi', goals: 116 },
  },
  Belgium: {
    flagCode: 'be',
    gradient: 'linear-gradient(135deg, #1a1a1a 0%, #b71c1c 50%, #e8a000 100%)',
    primary:  '#FAE042',
    glow:     'rgba(237,41,57,0.4)',
    hint:     { name: 'Romelu Lukaku', goals: 89 },
  },
  Brazil: {
    flagCode: 'br',
    gradient: 'linear-gradient(135deg, #00c853 0%, #007a33 50%, #004d1f 100%)',
    primary:  '#FFDF00',
    glow:     'rgba(255,223,0,0.4)',
    hint:     { name: 'Neymar', goals: 79 },
  },
  England: {
    flagCode: 'gb-eng',
    gradient: 'linear-gradient(135deg, #7f0000 0%, #c0071a 45%, #1a1a2e 100%)',
    primary:  '#CF081F',
    glow:     'rgba(207,8,31,0.45)',
    hint:     { name: 'Harry Kane', goals: 78 },
  },
  France: {
    flagCode: 'fr',
    gradient: 'linear-gradient(135deg, #001f80 0%, #002ecf 50%, #a00 100%)',
    primary:  '#4A90E2',
    glow:     'rgba(0,35,149,0.5)',
    hint:     { name: 'Olivier Giroud', goals: 57 },
  },
  Germany: {
    flagCode: 'de',
    gradient: 'linear-gradient(135deg, #1c1c1c 0%, #8b0000 50%, #c9a000 100%)',
    primary:  '#FFCE00',
    glow:     'rgba(255,206,0,0.4)',
    hint:     { name: 'Miroslav Klose', goals: 71 },
  },
  Italy: {
    flagCode: 'it',
    gradient: 'linear-gradient(135deg, #001a80 0%, #0040cc 55%, #004d00 100%)',
    primary:  '#4A80E8',
    glow:     'rgba(0,51,153,0.45)',
    hint:     { name: 'Gigi Riva', goals: 35 },
  },
  Netherlands: {
    flagCode: 'nl',
    gradient: 'linear-gradient(135deg, #c94400 0%, #ff6600 45%, #001a7a 100%)',
    primary:  '#FF6600',
    glow:     'rgba(255,102,0,0.45)',
    hint:     { name: 'Memphis Depay', goals: 55 },
  },
  Portugal: {
    flagCode: 'pt',
    gradient: 'linear-gradient(135deg, #004d00 0%, #cc0000 55%, #7a0000 100%)',
    primary:  '#FF4444',
    glow:     'rgba(220,0,0,0.45)',
    hint:     { name: 'Cristiano Ronaldo', goals: 143 },
  },
  Spain: {
    flagCode: 'es',
    gradient: 'linear-gradient(135deg, #8a0f15 0%, #c0181f 50%, #7a0a10 100%)',
    primary:  '#F5C518',
    glow:     'rgba(241,191,0,0.45)',
    hint:     { name: 'David Villa', goals: 59 },
  },
  Uruguay: {
    flagCode: 'uy',
    gradient: 'linear-gradient(135deg, #3a9fd4 0%, #1060bb 50%, #082050 100%)',
    primary:  '#5EB6E4',
    glow:     'rgba(94,182,228,0.45)',
    hint:     { name: 'Luis Suarez', goals: 69 },
  },
};

// ── PLAYER SUPPLEMENTARY DATA ─────────────────────────────────
// Keys MUST match the "Player" column in goals.csv exactly.
// caps / debut are informational; img points to /images/ folder.
const PLAYER_META = {
  'Lionel Messi':        { caps: 180, debut: 2005, img: 'images/messi.png' },
  'Gabriel Batistuta':   { caps:  78, debut: 1991, img: 'images/batistuta.png' },
  'Sergio Aguero':       { caps: 101, debut: 2006, img: 'images/aguero.png' },

  'Romelu Lukaku':       { caps: 114, debut: 2010, img: 'images/lukaku.png' },
  'Kevin De Bruyne':     { caps: 103, debut: 2010, img: 'images/kdb.png' },
  'Eden Hazard':         { caps: 126, debut: 2008, img: 'images/hazard.png' },

  'Neymar':              { caps: 125, debut: 2010, img: 'images/neymar.png' },
  'Pele':                { caps:  92, debut: 1957, img: 'images/pele.png' },
  'Ronaldo':             { caps:  98, debut: 1994, img: 'images/r9.png' },

  'Harry Kane':          { caps: 100, debut: 2015, img: 'images/kane.png' },
  'Wayne Rooney':        { caps: 120, debut: 2003, img: 'images/rooney.png' },
  'Bobby Charlton':      { caps: 106, debut: 1958, img: 'images/bobby.png' },

  'Olivier Giroud':      { caps: 132, debut: 2011, img: 'images/giroud.png' },
  'Kylian Mbappe':       { caps:  86, debut: 2017, img: 'images/mbappe.png' },
  'Thierry Henry':       { caps: 123, debut: 1994, img: 'images/henry.png' },

  'Miroslav Klose':      { caps: 137, debut: 2001, img: 'images/klose.png' },
  'Gerd Muller':         { caps:  62, debut: 1966, img: 'images/gerd.png' },
  'Lukas Podolski':      { caps: 130, debut: 2004, img: 'images/podolski.png' },

  'Gigi Riva':           { caps:  42, debut: 1965, img: 'images/riva.png' },
  'Giuseppe Meazza':     { caps:  53, debut: 1930, img: 'images/meazza.png' },
  'Silvio Piola':        { caps:  34, debut: 1935, img: 'images/piola.png' },

  'Memphis Depay':       { caps:  99, debut: 2013, img: 'images/depay.png' },
  'Robin van Persie':    { caps: 102, debut: 2005, img: 'images/rvp.png' },
  'Klaas-Jan Huntelaar': { caps:  76, debut: 2006, img: 'images/klaas.png' },

  'Cristiano Ronaldo':   { caps: 214, debut: 2003, img: 'images/cristiano.png' },
  'Pauleta':             { caps:  88, debut: 1996, img: 'images/pauleta.png' },
  'Eusebio':             { caps:  64, debut: 1961, img: 'images/eusebio.png' },

  'David Villa':         { caps:  98, debut: 2005, img: 'images/david.png' },
  'Raul':                { caps: 102, debut: 1996, img: 'images/raul.png' },
  'Fernando Torres':     { caps: 110, debut: 2003, img: 'images/torres.png' },

  'Luis Suarez':         { caps: 135, debut: 2007, img: 'images/suarez.png' },
  'Edinson Cavani':      { caps: 134, debut: 2008, img: 'images/cavani.png' },
  'Diego Forlan':        { caps: 112, debut: 2002, img: 'images/forlan.png' },
};

// ── STATE ─────────────────────────────────────────────────────
let chartInstance = null;

// ── INIT ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderNationsGrid();
});

// ── NATIONS GRID (instant — uses local NATION_META, no API) ──
function renderNationsGrid() {
  const grid = document.getElementById('nationsGrid');
  grid.innerHTML = '';

  Object.entries(NATION_META).forEach(([name, meta]) => {
    const card = document.createElement('div');
    card.className = 'nation-card';
    card.style.setProperty('--nc-gradient', meta.gradient);
    card.style.setProperty('--nc-glow', meta.glow);
    card.setAttribute('id', `nc-${name}`);
    card.innerHTML = `
      <div class="nc-bg"></div>
      <div class="nc-overlay"></div>
      <img class="nc-flag-img" src="https://flagcdn.com/w80/${meta.flagCode}.png" alt="${name} flag" />
      <div class="nc-name">${name}</div>
      <div class="nc-top">⚽ ${meta.hint.name} · ${meta.hint.goals} goals</div>
    `;
    card.addEventListener('click', () => selectNation(name, meta));
    grid.appendChild(card);
  });
}

// ── SELECT NATION (fetches live from API) ─────────────────────
async function selectNation(name, meta) {
  // Swap sections
  document.getElementById('landing').style.display    = 'none';
  document.getElementById('nationView').style.display  = 'block';
  document.getElementById('backBtn').style.display     = 'inline-flex';

  // Populate hero immediately (no wait)
  const hero = document.getElementById('nationHero');
  hero.style.background = meta.gradient;
  document.getElementById('heroFlag').innerHTML =
    `<img src="https://flagcdn.com/w160/${meta.flagCode}.png" alt="${name}"
          style="height:90px;border-radius:6px;box-shadow:0 4px 20px rgba(0,0,0,0.5)" />`;
  document.getElementById('heroName').textContent = name;
  document.getElementById('heroSub').textContent  = 'Loading…';
  document.getElementById('playersGrid').innerHTML =
    `<div class="loading-msg">Fetching from goals.csv…</div>`;

  window.scrollTo({ top: 0, behavior: 'smooth' });

  // ── Fetch from API (/players/{nation} reads goals.csv) ─────
  try {
    const res = await fetch(`/players/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { players } = await res.json();

    // Merge CSV data with local supplementary data
    const enriched = players.map(p => {
      const sup = PLAYER_META[p.Player] ?? {};
      return {
        name:  p.Player,
        goals: p.Goals,
        caps:  sup.caps  ?? '–',
        debut: sup.debut ?? '–',
        img:   sup.img   ?? '',
      };
    });

    document.getElementById('heroSub').textContent =
      `Top ${enriched.length} all-time · Record: ${enriched[0].name} (${enriched[0].goals} goals)`;

    renderPlayerCards(meta, enriched);
    renderChart(meta, enriched);

  } catch (err) {
    document.getElementById('playersGrid').innerHTML =
      `<p style="color:rgba(255,255,255,0.5);padding:2rem;grid-column:1/-1">
        ⚠️ Could not load data. Make sure the server is running:<br>
        <code style="opacity:.7">uvicorn app:app --reload</code>
      </p>`;
    console.error('API error:', err);
  }
}

// ── BACK ──────────────────────────────────────────────────────
function showLanding() {
  document.getElementById('landing').style.display    = 'block';
  document.getElementById('nationView').style.display = 'none';
  document.getElementById('backBtn').style.display    = 'none';
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
}

// ── PLAYER CARDS ──────────────────────────────────────────────
function renderPlayerCards(meta, players) {
  const grid = document.getElementById('playersGrid');
  grid.innerHTML = '';
  const maxGoals = players[0].goals;

  players.forEach((p, i) => {
    const initials = p.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
    const barPct   = ((p.goals / maxGoals) * 100).toFixed(1);

    const card = document.createElement('div');
    card.className = 'player-card';
    card.innerHTML = `
      <div class="pc-img-wrap" style="--pc-bg:${meta.gradient}">
        <div class="pc-rank">#${i + 1}</div>
        <img  class="pc-img"
              src="${p.img}"
              alt="${p.name}"
              onerror="this.style.display='none';this.nextElementSibling.style.display='flex'" />
        <div  class="pc-avatar" style="background:${meta.gradient};display:none">
          <span class="pc-initials">${initials}</span>
          <span class="pc-ball-bg">⚽</span>
        </div>
      </div>
      <div class="pc-body">
        <div class="pc-name">${p.name}</div>
        <div class="pc-goals-row">
          <span class="pc-goals-num" style="color:${meta.primary}">${p.goals}</span>
          <span class="pc-goals-lbl">international goals</span>
        </div>
        <div class="pc-caps">${p.caps} caps &nbsp;·&nbsp; Debut ${p.debut}</div>
        <div class="pc-bar">
          <div class="pc-bar-fill" data-w="${barPct}%"></div>
        </div>
      </div>
    `;

    grid.appendChild(card);

    // Animate progress bar
    requestAnimationFrame(() => {
      setTimeout(() => {
        const fill = card.querySelector('.pc-bar-fill');
        if (fill) { fill.style.background = meta.primary; fill.style.width = fill.dataset.w; }
      }, 80 + i * 140);
    });
  });
}

// ── CHART ─────────────────────────────────────────────────────
function renderChart(meta, players) {
  if (chartInstance) chartInstance.destroy();

  const ctx    = document.getElementById('goalChart').getContext('2d');
  const labels = players.map(p => p.name);
  const values = players.map(p => p.goals);
  const bgs    = ['CC', '99', '66'].map(a => meta.primary + a);

  chartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'International Goals',
        data: values,
        backgroundColor: bgs,
        borderColor: meta.primary,
        borderWidth: 2,
        borderRadius: 10,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1200, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(5,5,20,0.85)',
          titleColor: '#fff',
          bodyColor: 'rgba(255,255,255,0.65)',
          padding: 14,
          cornerRadius: 10,
          callbacks: { label: ctx => `  ${ctx.raw} goals` },
        },
      },
      scales: {
        x: {
          grid:  { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: 'rgba(255,255,255,0.65)', font: { family: 'Outfit', size: 13, weight: '600' } },
        },
        y: {
          beginAtZero: true,
          grid:  { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: 'rgba(255,255,255,0.55)', font: { family: 'Outfit', size: 12 } },
        },
      },
    },
  });
}