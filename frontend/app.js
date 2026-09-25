const API_BASE = "http://127.0.0.1:8000";

const $ = id => document.getElementById(id);
const state = {
  tcirDemo: null,
  tcirTemporal: null,
  tcirSingle: null,
  insat: null,
  synthetic: null,
  track: null,
};

function text(id, value) {
  const v = value == null || value === "" ? "—" : String(value);
  document.querySelectorAll(`[id="${id}"]`).forEach(el => { el.textContent = v; });
}
function status(message, good=false) {
  text("statusText", message);
  const dot = $("systemDot");
  if (dot) dot.style.background = good ? "var(--green)" : "var(--cyan)";
}
function setBusy(button, label){
  if(!button) return ()=>{};
  const original=button.dataset.originalLabel || button.textContent;
  button.dataset.originalLabel=original; button.disabled=true; button.classList.add("btn-loading"); button.textContent=label;
  return ()=>{button.disabled=false;button.classList.remove("btn-loading");button.textContent=original;};
}
function flashData(selector){
  document.querySelectorAll(selector).forEach(el=>{el.classList.remove("data-in"); void el.offsetWidth; el.classList.add("data-in");});
}
async function api(path, options={}, timeout=45000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(`${API_BASE}${path}`, {...options, signal:controller.signal});
    const raw = await response.text();
    let data = {};
    try { data = raw ? JSON.parse(raw) : {}; } catch { data = {error: raw}; }
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    if (data.status === "error") throw new Error(data.error || "Backend error");
    return data;
  } finally { clearTimeout(timer); }
}

function updateClock(){
  const now = new Intl.DateTimeFormat("en-IN", {
    hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false, timeZone:"Asia/Kolkata"
  }).format(new Date());
  text("clock", `${now} IST`);
}

function openWindow(name){
  const layer = $("windowLayer");
  document.querySelectorAll("[data-window-panel]").forEach(w => w.classList.remove("visible"));
  const panel = $(`window-${name}`);
  if (!panel) return;
  layer.classList.add("open");
  panel.classList.add("visible");
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.window === name));
  if(name === "sources") loadSources();
  if(name === "reports") loadReports();
  if(name === "satellite") loadSources();
  if(name === "forecast") { loadTrack(); setTimeout(()=>{ if(trackMap) trackMap.invalidateSize(); },250); }
  if(name === "comparison") {}
}
function closeWindows(){
  $("windowLayer")?.classList.remove("open");
  document.querySelectorAll("[data-window-panel]").forEach(w => w.classList.remove("visible"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.window === "overview"));
}

function setupWindows(){
  document.querySelectorAll("[data-window]").forEach(btn => {
    btn.addEventListener("click", () => openWindow(btn.dataset.window));
  });
  document.querySelectorAll(".close-window").forEach(btn => btn.addEventListener("click", closeWindows));
  $("windowLayer").addEventListener("click", e => { if(e.target === $("windowLayer")) closeWindows(); });
  document.addEventListener("keydown", e => { if(e.key === "Escape") closeWindows(); });
}

function color(v){
  let x = Math.max(0, Math.min(1, Number(v) || 0));
  const stops = [
    [0,[8,8,12]],[.35,[80,80,80]],[.55,[220,220,220]],
    [.65,[255,255,0]],[.75,[255,140,0]],[.85,[255,0,0]],[1,[160,0,200]]
  ];
  for(let i=0;i<stops.length-1;i++){
    const [ta,ca]=stops[i], [tb,cb]=stops[i+1];
    if(x>=ta && x<=tb){
      const f=(x-ta)/(tb-ta);
      return ca.map((c,j)=>Math.round(c+f*(cb[j]-c)));
    }
  }
  return stops.at(-1)[1];
}
function drawArray(canvas, array, mode="thermal"){
  if(!canvas || !array) return;
  const ctx=canvas.getContext("2d");
  const h=array.length, w=array[0]?.length || 0;
  if(!w) return;
  const values = array.flat();
  let min=Math.min(...values), max=Math.max(...values);
  if(max === min){ min=0; max=1; }
  const img=ctx.createImageData(w,h);
  let p=0;
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const n=(array[y][x]-min)/(max-min);
    const c=mode==="gray" ? [n*255,n*255,n*255] : color(n);
    img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=255;
  }
  const tmp=document.createElement("canvas"); tmp.width=w; tmp.height=h;
  tmp.getContext("2d").putImageData(img,0,0);
  ctx.imageSmoothingEnabled=true;
  ctx.drawImage(tmp,0,0,canvas.width,canvas.height);
}

function drawFrame(frames){
  const frame=frames?.at(-1);
  if(!frame) return;
  drawArray($("frameCanvas"), frame.map(r=>r.map(px=>px[0])));
  const names=[["irCanvas",0,"thermal"],["wvCanvas",1,"thermal"],["visCanvas",2,"gray"],["pmwCanvas",3,"thermal"]];
  for(const [id,ch,mode] of names){
    const c=$(id);
    if(c) drawArray(c, frame.map(r=>r.map(px=>px[ch])), mode);
  }
}
function renderFrameGrid(frames){
  const grid=$("frameGrid");
  if(!grid) return;
  grid.innerHTML="";
  frames.forEach((frame,i)=>{
    const cell=document.createElement("div"); cell.className="frame-cell";
    const c=document.createElement("canvas"); c.width=180; c.height=180;
    cell.appendChild(c);
    const label=document.createElement("span"); label.textContent=`FRAME ${i+1} · t−${(frames.length-1-i)*3}h`;
    cell.appendChild(label); grid.appendChild(cell);
    drawArray(c,frame.map(r=>r.map(px=>px[0])));
  });
}
function setGauge(wind){
  const gauge=$(".gauge");
  if(!gauge) return;
  const pct=Math.max(0,Math.min(100,(Number(wind)||0)/130*100));
  gauge.style.background=`conic-gradient(var(--amber) ${pct*3.6}deg,#e4ebea 0deg)`;
}

let trackMap = null;
let trackLayers = [];

function fmtCoord(value, positive, negative) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${Math.abs(n).toFixed(2)}° ${n >= 0 ? positive : negative}`;
}

function clearTrackLayers() {
  if (!trackMap) return;
  trackLayers.forEach(layer => { try { trackMap.removeLayer(layer); } catch {} });
  trackLayers = [];
}

function forecastIcon(label) {
  return L.divIcon({ className:"vn-forecast-icon", html:`<span>${label}</span>`, iconSize:[42,28], iconAnchor:[21,14] });
}
function currentIcon() {
  return L.divIcon({ className:"vn-current-icon", html:'<span class="vn-current-ring"><b></b></span>', iconSize:[30,30], iconAnchor:[15,15] });
}
function ensureTrackMap() {
  const el = $("trackMap");
  if (!el || typeof L === "undefined") return null;
  if (!trackMap) {
    trackMap = L.map(el,{zoomControl:true,attributionControl:true,scrollWheelZoom:true,zoomSnap:.25});
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:18,attribution:'&copy; OpenStreetMap contributors'}).addTo(trackMap);
    trackMap.setView([22.5,88.5],5);
  }
  return trackMap;
}
function drawGeoMap(hist, proj, data) {
  const map=ensureTrackMap(), statusEl=$("mapRenderStatus");
  if(!map){if(statusEl)statusEl.textContent="MAP LIBRARY UNAVAILABLE";return;}
  clearTrackLayers();
  const all=[...hist,...proj].filter(p=>Number.isFinite(Number(p.lat))&&Number.isFinite(Number(p.lon)));
  if(!all.length){if(statusEl)statusEl.textContent="NO TRACK COORDINATES";return;}
  const observed=hist.map(p=>[Number(p.lat),Number(p.lon)]).filter(p=>p.every(Number.isFinite));
  const forecast=proj.map(p=>[Number(p.lat),Number(p.lon)]).filter(p=>p.every(Number.isFinite));
  if(observed.length>=2){const l=L.polyline(observed,{color:"#12a9b8",weight:4,opacity:.95,lineCap:"round",lineJoin:"round"}).addTo(map);trackLayers.push(l);}
  observed.forEach((latlon,i)=>{const m=L.circleMarker(latlon,{radius:i===observed.length-1?7:4,color:i===observed.length-1?"#102733":"#fff",weight:i===observed.length-1?3:2,fillColor:"#12a9b8",fillOpacity:1}).addTo(map);trackLayers.push(m);});
  if(forecast.length){const start=observed.length?observed[observed.length-1]:forecast[0];const l=L.polyline([start,...forecast],{color:"#7656d7",weight:4,opacity:.95,dashArray:"9 8",lineCap:"round",lineJoin:"round"}).addTo(map);trackLayers.push(l);proj.forEach(p=>{const lat=Number(p.lat),lon=Number(p.lon);if(!Number.isFinite(lat)||!Number.isFinite(lon))return;const m=L.marker([lat,lon],{icon:forecastIcon(`+${p.hours_ahead}H`),zIndexOffset:400}).addTo(map);trackLayers.push(m);});}
  if(data.current){const lat=Number(data.current.lat),lon=Number(data.current.lon);if(Number.isFinite(lat)&&Number.isFinite(lon)){const m=L.marker([lat,lon],{icon:currentIcon(),zIndexOffset:800}).addTo(map);m.bindTooltip(`${data.storm_name||data.cyclone_id||"CURRENT"} · CURRENT`,{permanent:true,direction:"right",offset:[12,-12],className:"vn-current-tooltip"});trackLayers.push(m);}}
  const bounds=L.latLngBounds(all.map(p=>[Number(p.lat),Number(p.lon)]));
  if(bounds.isValid())map.fitBounds(bounds.pad(.18),{animate:true,duration:.55,maxZoom:8});
  setTimeout(()=>map.invalidateSize(),120);
  if(statusEl){statusEl.textContent="GEO LAYER READY · LIVE TRACK";statusEl.classList.add("ready");}
  $("trackMap")?.classList.add("map-ready");
}
function renderTrack(data){
  state.track=data;const current=data.current,motion=data.motion||{};
  text("trackPosition",current?`${fmtCoord(current.lat,"N","S")} / ${fmtCoord(current.lon,"E","W")}`:"NOT AVAILABLE");
  text("trackSpeed",motion.available?`${Number(motion.speed_kt).toFixed(1)} kt`:"—");
  text("trackDirection",motion.available?`${motion.direction||"—"} / ${Number(motion.bearing_deg).toFixed(1)}°`:"—");
  text("trackStatus",data.history?.length?"OBSERVED TRACK READY":"TRACK DATA UNAVAILABLE");
  text("trackProjectionNote",data.history?.length?`IMD best-track history · ${data.history.length} observations · learned GRU forecast at +3h / +6h / +9h.`:(data.availability?.reason||"No track data."));
  text("trackHistoryCount",data.history?.length||0);text("trackProjectionCount",data.projection?.length||0);
  const rows=$("projectionRows");
  if(rows)rows.innerHTML=(data.projection||[]).length?data.projection.map(p=>`<div class="projection-row"><span>+${p.hours_ahead}H</span><b>${fmtCoord(p.lat,"N","S")}</b><b>${fmtCoord(p.lon,"E","W")}</b></div>`).join(""):`<div class="empty-note">${data.track_model?.available===false?(data.track_model?.reason||"AI forecast unavailable"):"No forecast points available."}</div>`;
  drawGeoMap(data.history||[],data.projection||[],data);
}
async function loadTrack(){
  text("trackStatus","ACQUIRING TRACK…");const ms=$("mapRenderStatus");if(ms){ms.classList.remove("ready");ms.textContent="ACQUIRING GEOSPATIAL LAYER…";}
  try{const data=await api("/track",{},15000);if(data.status==="error")throw new Error(data.error||"Track service error");renderTrack(data);}catch(err){text("trackStatus","UNAVAILABLE");text("trackProjectionNote",err.message);if(ms)ms.textContent="TRACK SERVICE UNAVAILABLE";}
}

function renderCategoryBars(category){
  const box=$("categoryBars");
  if(!box) return;
  box.innerHTML="";
  const row=document.createElement("div"); row.className="bar-row";
  row.innerHTML=`<span>${category}</span><div class="bar-bg"><div class="bar-fill" style="width:100%"></div></div><b>DERIVED</b>`;
  box.appendChild(row);
}

function renderTCIRPrediction(pred){
  state.tcirTemporal=pred;
  const i=pred.intensity || {};
  const category=pred.classification?.predicted_category || "—";
  text("intensityValue", Number.isFinite(Number(i.wind_kt)) ? Number(i.wind_kt).toFixed(1) : "—");
  text("cycloneWind", Number.isFinite(Number(i.wind_kt)) ? `${Number(i.wind_kt).toFixed(1)} kt` : "—");
  text("predCategory",category); text("predCategorySecondary",category);
  text("pressureValue", Number.isFinite(Number(i.pressure_hpa)) ? `${Number(i.pressure_hpa).toFixed(1)} hPa` : "—");
  text("sizeValue", Number.isFinite(Number(i.size_nmi)) ? `${Number(i.size_nmi).toFixed(1)} nmi` : "—");
  text("liveWind", `${Number(i.wind_kt).toFixed(2)} kt`);
  text("livePressure", `${Number(i.pressure_hpa).toFixed(2)} hPa`);
  text("liveSize", `${Number(i.size_nmi).toFixed(2)} nmi`);
  text("liveCategory",category);
  text("forecastWindow",`${pred.temporal_context?.history_hours ?? 9} H`);
  text("framesReceived",pred.temporal_context?.frames ?? 4);
  text("temporalMode",`${pred.temporal_context?.frames ?? 4}-FRAME / ${pred.temporal_context?.interval_hours ?? 3}H`);
  text("sequenceTag","TCIR READY");
  text("presence","KNOWN"); text("presenceState","TCIR TEST SAMPLE");
  $("presenceBar").style.width="100%";
  renderCategoryBars(category);
  setGauge(Number(i.wind_kt));
  text("intensityNote","TCIR temporal CNN + GRU output. Category is derived from predicted wind."); text("activeCyclones","01"); text("activeCycloneCaption","REAL TCIR TEST SAMPLE");
  text("tcirState","READY");
  text("singleState","READY");
}

function renderTCIRDemo(demo){
  state.tcirDemo=demo;
  const id=demo.cyclone_id || demo.source?.cyclone_id || "TCIR TEST SAMPLE";
  const timestamp=demo.target_timestamp || demo.source?.target_timestamp || "—";
  text("cycloneName",id); text("cycloneId",id); text("frameSource",`TCIR · ${timestamp}`); text("activeCyclones","01"); text("activeCycloneCaption","REAL TCIR TEST SAMPLE");
  text("gtWind",`${Number(demo.ground_truth?.wind_kt).toFixed(1)} kt`);
  text("gtPressure",`${Number(demo.ground_truth?.pressure_hpa).toFixed(1)} hPa`);
  text("gtSize",`${Number(demo.ground_truth?.size_nmi).toFixed(1)} nmi`);
  text("cycloneCategory",demo.ground_truth?.category || "Reference sample");
  renderFrameGrid(demo.frames);
  drawFrame(demo.frames);
  text("sequenceMeta",`Cyclone ${id} · target ${timestamp} · HDF5 indices: ${(demo.temporal_context?.h5_indices||[]).join(", ")} · chronological order: oldest → newest`);
}

async function runTCIR(){
  const release=setBusy($("liveBtn")," LOADING TCIR…"); const releaseWindow=setBusy($("rerunTcir")," PROCESSING…");
  document.querySelectorAll(".metric-card").forEach(x=>x.classList.add("is-loading"));
  status("Loading real TCIR sequence…");
  text("sequenceTag","LOADING");
  try{
    const demo=await api("/tcir_demo",{},30000);
    renderTCIRDemo(demo);
    loadTrack();
    status("TCIR frames loaded. Running temporal model…");
    const pred=await api("/predict_tcir_sequence",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({frames:demo.frames})
    },60000);
    renderTCIRPrediction(pred);
    const single=await api("/predict_tcir_single",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({frame:demo.frames.at(-1)})
    },60000);
    state.tcirSingle=single;
    text("compareTemporal",`${pred.intensity.wind_kt.toFixed(1)} kt`);
    text("compareSingle",`${single.intensity.wind_kt.toFixed(1)} kt`);
    status("TCIR temporal analysis complete.",true);
    text("footerModel","TCIR TEMPORAL / READY");
    document.querySelectorAll(".metric-card").forEach(x=>x.classList.remove("is-loading")); flashData(".metric-card,.analysis-panel,.result-card"); release(); releaseWindow(); openWindow("live");
  }catch(err){
    document.querySelectorAll(".metric-card").forEach(x=>x.classList.remove("is-loading")); release(); releaseWindow();
    console.error(err); text("sequenceTag","ERROR");
    try {
      const d = await api("/debug_paths",{},10000);
      const h = d.tcir_h5, m = d.tcir_manifest;
      status(`TCIR failed · H5 ${h?.exists ? "OK" : "MISSING"} · manifest ${m?.exists ? "OK" : "MISSING"}`);
    } catch { status(`TCIR failed: ${err.message}`); }
  }
}

async function runINSAT(){
  const release=setBusy($("insatBtn")," LOADING INSAT…"); const releaseWindow=setBusy($("runInsatWindow")," RENDERING…");
  document.querySelectorAll("#insatSourcePanel .insat-live-wrap,.metric-card.blue").forEach(x=>x.classList.add("is-loading"));
  status("Loading local INSAT-3DR/3DS processed observation…");
  try{
    const data=await api("/insat_demo",{},60000);
    if(data.status === "error") throw new Error(data.error || "INSAT sample unavailable");
    state.insat=data;
    const tir=$("insatTirPreview"), wv=$("insatWvPreview");
    if(tir) tir.src = data.preview?.tir1 || `${API_BASE}/insat_preview/tir1?ts=${Date.now()}`;
    if(wv) wv.src = data.preview?.wv || `${API_BASE}/insat_preview/wv?ts=${Date.now()}`;
    text("insatCategory",data.classification?.predicted_category || "INSAT SAMPLE");
    text("insatWind",data.intensity?.wind_kt != null ? `${Number(data.intensity.wind_kt).toFixed(1)} kt` : "—");
    text("insatModel",data.model ? "AI INFERENCE" : "READY");
    text("compareInsat",data.classification?.predicted_category || "—");
    text("insatState","LOCAL + AI");
    text("intensityNote",`Processed INSAT sample · ${data.sample?.tir1 || "TIR1"} · classifier inference active.`);
    status("INSAT processed sample + classifier loaded.",true); document.querySelectorAll("#insatSourcePanel .insat-live-wrap,.metric-card.blue").forEach(x=>x.classList.remove("is-loading")); flashData(".metric-card.blue,#insatSourcePanel .insat-live-wrap"); release(); releaseWindow(); openWindow("satellite");
  }catch(err){
    document.querySelectorAll("#insatSourcePanel .insat-live-wrap,.metric-card.blue").forEach(x=>x.classList.remove("is-loading")); release(); releaseWindow();
    console.error(err);
    text("insatModel","ERROR");
    text("insatState","UNAVAILABLE");
    try {
      const d = await api("/debug_paths",{},10000);
      const s = d.insat_sample || {};
      status(`INSAT failed · TIR ${s.tir1 ? "OK" : "MISSING"} · WV ${s.wv ? "OK" : "MISSING"}`);
    } catch { status(`INSAT failed: ${err.message}`); }
  }
}

async function runSynthetic(){
  const release=setBusy($("runSynthetic")," VALIDATING…");
  status("Running synthetic validation path…");
  try{
    const demo=await api("/demo_sequence",{},30000);
    if(demo.status === "error") throw new Error(demo.error || "Synthetic validation sample unavailable");
    state.synthetic=demo;
    const pred=await api("/predict",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({frames:demo.frames})
    },30000);
    text("syntheticPresence",`${(pred.identification?.cyclone_present_probability*100).toFixed(1)}%`);
    text("syntheticTrend",pred.prediction?.trend || "—");
    text("syntheticTrack",(pred.prediction?.predicted_next_step_track_delta||[]).map(v=>Number(v).toFixed(2)).join(" / "));
    text("syntheticState",pred.mode === "trained_model" ? "MODEL VALIDATION" : "PIPELINE FALLBACK");
    text("syntheticCategory",pred.classification?.predicted_category || "VALIDATION");
    status(pred.mode === "trained_model" ? "Synthetic model validation complete." : "Synthetic validation fallback complete — not real-world performance.",true); release(); flashData("#syntheticSourcePanel .source-result-grid");
  }catch(err){release(); status(`Synthetic path failed: ${err.message}`);}
}

async function loadSources(){
  const box=$("sourceMatrix"); if(!box) return;
  try{
    const data=await api("/data_sources",{},10000);
    box.innerHTML="";
    Object.entries(data.sources||{}).forEach(([name,s])=>{
      const row=document.createElement("div"); row.className="source-row";
      let cls=s.status==="READY"?"status-ready":s.status==="VALIDATION ONLY"?"status-validation":"status-missing";
      const detail=Object.entries(s).filter(([k])=>!["status"].includes(k)).map(([k,v])=>`${k}: ${typeof v==="object"?JSON.stringify(v):v}`).join(" · ");
      row.innerHTML=`<strong>${name}</strong><span class="${cls}">${s.status}</span><span>${s.real_sample===true?"REAL SAMPLE":s.manifest===true?"MANIFEST READY":""}</span><span>${detail}</span>`;
      box.appendChild(row);
    });
  }catch(err){box.innerHTML=`<div class="empty-note">Source matrix unavailable: ${err.message}</div>`;}
}

async function loadReports(){
  const summary=$("reportSummary"), tbody=$("reportRows");
  try{
    const data=await api("/metrics",{},10000);
    summary.innerHTML="";
    const chip=(label,val)=>{const d=document.createElement("div");d.className="summary-chip";d.innerHTML=`<span>${label}</span><b>${val}</b>`;summary.appendChild(d)};
    chip("OVERALL ROWS",data.overall?.length||0); chip("INTENSITY BINS",data.intensity_bins?.length||0);
    tbody.innerHTML="";
    (data.overall||[]).forEach(r=>{
      const tr=document.createElement("tr");
      tr.innerHTML=`<td>${r.target??"—"}</td><td>${fmt(r.MAE)}</td><td>${fmt(r.RMSE)}</td><td>${fmt(r.R2)}</td><td>${r.unit??"—"}</td>`;
      tbody.appendChild(tr);
    });
    if(!data.overall?.length) tbody.innerHTML=`<tr><td colspan="5">No evaluation CSV found on backend.</td></tr>`;
  }catch(err){summary.innerHTML=`<div class="empty-note">Reports unavailable: ${err.message}</div>`;}
}
function fmt(v){return v==null||Number.isNaN(Number(v))?"—":Number(v).toFixed(3)}

async function runComparison(){
  if(!state.tcirDemo) await runTCIR();
  if(!state.insat) { try{state.insat=await api("/insat_demo",{},60000)}catch{} }
  if(!state.tcirSingle && state.tcirDemo) {
    try{state.tcirSingle=await api("/predict_tcir_single",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({frame:state.tcirDemo.frames.at(-1)})},60000)}catch{}
  }
  if(state.tcirTemporal) text("compareTemporal",`${state.tcirTemporal.intensity.wind_kt.toFixed(1)} kt`);
  if(state.tcirSingle) text("compareSingle",`${state.tcirSingle.intensity.wind_kt.toFixed(1)} kt`);
  if(state.insat) text("compareInsat",state.insat.classification?.predicted_category || "—");
  openWindow("comparison");
}

async function health(){
  try{
    const data=await api("/health",{},7000);
    text("systemState","NODE ONLINE");
    $("systemDot").style.background="var(--green)";
    text("tcirState",data.models?.tcir_temporal?"READY":"MISSING");
    text("singleState",data.models?.tcir_intensity?"READY":"MISSING");
    text("insatState",data.models?.insat?"MODEL + LIVE":"LIVE SOURCE");
    text("syntheticState",data.models?.synthetic?"VALIDATION ONLY":"MISSING");
    text("footerModel",`LOCAL NODE / ${data.device||"CPU"}`);
    status("Backend online · model registry loaded.",true);
  }catch(err){
    text("systemState","NODE OFFLINE"); $("systemDot").style.background="var(--red)";
    status("Backend offline. Start FastAPI on port 8000.");
  }
}

function setupTabs(){
  document.querySelectorAll(".source-tab").forEach(btn=>{
    btn.addEventListener("click",()=>{
      document.querySelectorAll(".source-tab").forEach(x=>x.classList.remove("active"));
      btn.classList.add("active");
      ["insat","tcir","synthetic"].forEach(name=>$(`${name}SourcePanel`)?.classList.toggle("hidden",name!==btn.dataset.sourceTab));
      if(btn.dataset.sourceTab === "insat" && !state.insat) runINSAT();
      if(btn.dataset.sourceTab === "synthetic" && !state.synthetic) runSynthetic();
    });
  });
}

document.addEventListener("DOMContentLoaded",()=>{
  const loginScreen=$("loginScreen"), shell=$("consoleShell"), form=$("loginForm");
  const enter=()=>{ loginScreen?.classList.add("login-exit"); setTimeout(()=>{loginScreen?.remove(); if(shell){shell.setAttribute("aria-hidden","false");shell.classList.add("console-enter");}},650); updateClock(); setInterval(updateClock,1000); health(); loadSources(); loadReports(); loadTrack(); };
  form?.addEventListener("submit",e=>{e.preventDefault(); const u=$("loginUser")?.value.trim(), p=$("loginPass")?.value.trim(); if(!u||!p)return; form.classList.add("login-verified"); setTimeout(enter,420); });
  setupWindows(); setupTabs();
  $("liveBtn")?.addEventListener("click",runTCIR);
  $("insatBtn")?.addEventListener("click",runINSAT);
  $("sourceBtn")?.addEventListener("click",()=>openWindow("sources"));
  $("rerunTcir")?.addEventListener("click",runTCIR);
  $("runInsatWindow")?.addEventListener("click",runINSAT);
  $("runSynthetic")?.addEventListener("click",runSynthetic);
  $("runComparison")?.addEventListener("click",runComparison);
});