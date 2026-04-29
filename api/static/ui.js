// ═══════════════════════════════════════════════════════════════════════════
// Face ID — Dashboard JS
// ═══════════════════════════════════════════════════════════════════════════

// ── State ──────────────────────────────────────────────────────────────────
var _token    = null;
var _username = null;
var _role        = null;
var _displayName = null;
var _curPage  = "feeds";

function qs(id) { return document.getElementById(id); }

// ── Auth screen tab ─────────────────────────────────────────────────────────
function authTab(t) {
  qs("tab-login").classList.toggle("active",    t === "login");
  qs("tab-register").classList.toggle("active", t === "register");
  qs("form-login").classList.toggle("show",     t === "login");
  qs("form-register").classList.toggle("show",  t === "register");
  qs("loginErr").textContent = "";
  qs("regErr").textContent   = "";
}

// ── Login ──────────────────────────────────────────────────────────────────
async function doLogin() {
  var u = qs("loginUser").value.trim();
  var p = qs("loginPass").value;
  qs("loginErr").textContent = "";
  if (!u || !p) { qs("loginErr").textContent = "Fill in both fields."; return; }

  var res, data;
  try {
    var body = new URLSearchParams({ username: u, password: p });
    res  = await fetch("/token", { method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: body });
    data = await res.json();
  } catch(e) { qs("loginErr").textContent = "Network error: " + e.message; return; }

  if (!res.ok) { qs("loginErr").textContent = data.detail || "Login failed"; return; }

  _token = data.access_token;
  try {
    var pl = JSON.parse(atob(_token.split(".")[1]));
    _username = pl.sub; _role = pl.role;
    _displayName = pl.display_name || pl.sub;
  } catch { _username = u; _role = "user"; _displayName = u; }
  enterDashboard();
}

// ── Register ───────────────────────────────────────────────────────────────
async function doRegister() {
  var u  = qs("regUser").value.trim();
  var p  = qs("regPass").value;
  var p2 = qs("regPass2").value;
  qs("regErr").textContent = "";

  if (u.length < 3)   { qs("regErr").textContent = "Username min 3 chars.";    return; }
  if (p.length < 6)   { qs("regErr").textContent = "Password min 6 chars.";    return; }
  if (p !== p2)        { qs("regErr").textContent = "Passwords do not match.";  return; }

  var res, data;
  try {
    res  = await fetch("/users/register", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p, password2: p2 })
    });
    data = await res.json();
  } catch(e) { qs("regErr").textContent = "Network error: " + e.message; return; }

  if (!res.ok) { qs("regErr").textContent = data.detail || ("Error " + res.status); return; }

  // Auto-login after register
  qs("loginUser").value = u;
  qs("loginPass").value = p;
  authTab("login");
  await doLogin();
}

// ── Logout ─────────────────────────────────────────────────────────────────
function doLogout() {
  pages.reg.stopCam();
  pages.logs.stopAuto();
  _token = _username = _role = _displayName = null;
  qs("auth-screen").style.display = "flex";
  qs("app").style.display = "none";
  qs("loginPass").value = "";
}

// ── Enter dashboard ────────────────────────────────────────────────────────
function enterDashboard() {
  qs("auth-screen").style.display = "none";
  qs("app").style.display = "flex";

  var _shownName = _displayName || _username;
  qs("sidebar-user").textContent = _shownName + " (" + _role + ")";
  qs("topbar-user").textContent  = _shownName;
  var _mu = qs("menuUsername"); if (_mu) _mu.textContent = _shownName;
  var _mr = qs("menuRole");
  if (_mr) {
    var _rc = {admin:"#f59e0b", moderator:"#818cf8", user:"#4ade80"};
    _mr.innerHTML = '<span style="color:' + (_rc[_role]||"#aaa") + ';font-weight:600">' + _role + '</span>';
  }

  // Role visibility
  var isStaff = (_role === "admin" || _role === "moderator");
  var isAdmin = (_role === "admin");
  document.querySelectorAll(".staff-only").forEach(function(el) {
    el.style.display = isStaff ? "" : "none";
  });
  document.querySelectorAll(".admin-only").forEach(function(el) {
    el.style.display = isAdmin ? "" : "none";
  });

  // Users land on their personal log; staff land on live feeds
  nav(_role === "user" ? "logs" : "feeds");
}

// ── Navigation ─────────────────────────────────────────────────────────────
var PAGE_LABELS = {
  feeds: "Live Feeds", logs: "Detection Log", register: "Register Face",
  cameras: "Cameras", password: "Change Password",
  users: "User Management", facedb: "Face Database", audit: "Audit Log"
};

// Pages that auto-load on each visit vs only first visit
var _pagesLoaded = {};

function nav(page) {
  document.querySelectorAll(".page").forEach(function(el) { el.style.display = "none"; });
  document.querySelectorAll(".nav-item").forEach(function(el) {
    el.classList.toggle("active", el.dataset.page === page);
  });
  var el = qs("page-" + page);
  if (el) el.style.display = "";
  qs("topbar-title").textContent = PAGE_LABELS[page] || page;
  _curPage = page;

  // Load data for page
  if (page === "feeds")   pages.feeds.refresh();
  if (page === "logs")    pages.logs.init();
  if (page === "reg")     pages.reg.init();
  if (page === "cameras") { pages.cams.load(); }
  if (page === "users")   pages.users.load();
  if (page === "facedb")  pages.facedb.load();
  if (page === "audit")   pages.audit.load();
  // Stop metrics polling when leaving cameras page
  if (page !== "cameras" && pages.cams && pages.cams.stopMetricsAuto) pages.cams.stopMetricsAuto();
}

function toggleSidebar() {
  qs("sidebar").classList.toggle("open");
}

function toggleUserMenu() {
  var m = qs("userMenu"); if (!m) return;
  m.style.display = (m.style.display === "none") ? "" : "none";
}
document.addEventListener("click", function(e) {
  var btn = qs("userMenuBtn"), menu = qs("userMenu");
  if (!btn || !menu) return;
  if (!btn.contains(e.target) && !menu.contains(e.target))
    menu.style.display = "none";
});

// ── Core API helper ────────────────────────────────────────────────────────
async function api(url, opts) {
  if (!_token) { alert("Session expired — please log in again."); return null; }
  var res, text, data;
  try {
    res  = await fetch(url, Object.assign({}, opts, {
      headers: Object.assign({ "Authorization": "Bearer " + _token },
                              (opts && opts.headers) || {})
    }));
    text = await res.text();
    try { data = JSON.parse(text); } catch { data = null; }
  } catch(e) { alert("Network error: " + e.message); return null; }
  if (!res.ok) {
    alert("Error " + res.status + ":\n" + ((data && data.detail) || text.slice(0, 300)));
    return null;
  }
  return data;
}

// Small helper: set message element
function setMsg(id, text, isErr) {
  var el = qs(id);
  if (!el) return;
  el.textContent = text || "";
  el.className   = "msg" + (isErr ? " err" : (text ? " ok" : ""));
}

// Lightbox
function showPhoto(src) {
  qs("lightbox-img").src = src;
  qs("lightbox").classList.add("open");
}
function closeLightbox() {
  qs("lightbox").classList.remove("open");
}

// Escape HTML for safe inline insertion
function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")
                  .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ═══════════════════════════════════════════════════════════════════════════
// PAGE MODULES
// ═══════════════════════════════════════════════════════════════════════════
var pages = {};

// ── LIVE FEEDS ──────────────────────────────────────────────────────────────
pages.feeds = (function() {
  function refresh() {
    api("/cameras/").then(function(data) {
      if (!data) return;
      var cameras = (data.cameras || []).filter(function(c) { return c.streaming; });
      var grid = qs("feeds-grid");
      if (!grid) return;

      if (cameras.length === 0) {
        grid.innerHTML = '<p style="opacity:.45;grid-column:1/-1">' +
          'No cameras streaming — go to <a href="#" onclick="nav(\'cameras\');return false">Cameras</a> and click ▶ Start.</p>';
        return;
      }

      // Build set of existing tile ids
      var existing = {};
      grid.querySelectorAll("[data-cam-id]").forEach(function(el) {
        existing[el.dataset.camId] = el;
      });
      // Remove tiles for cameras no longer streaming
      Object.keys(existing).forEach(function(id) {
        if (!cameras.find(function(c) { return String(c.id) === id; }))
          existing[id].remove();
      });
      // Add tiles for new cameras
      cameras.forEach(function(cam) {
        var sid = String(cam.id);
        if (existing[sid]) {
          // Update recognition badge
          var b = existing[sid].querySelector(".rec-badge");
          if (b) {
            b.textContent = cam.run_recognition ? "🔍 On" : "⬜ Off";
            b.className   = "pill rec-badge " + (cam.run_recognition ? "pill-green" : "pill-muted");
          }
          return;
        }
        var tile = document.createElement("div");
        tile.className    = "feed-tile";
        tile.dataset.camId = sid;
        tile.innerHTML =
          '<div class="feed-tile-head">' +
            '<strong>' + esc(cam.name) + '</strong>' +
            '<span class="pill rec-badge ' + (cam.run_recognition ? "pill-green" : "pill-muted") + '">' +
              (cam.run_recognition ? "🔍 On" : "⬜ Off") + '</span>' +
          '</div>' +
          '<img src="/cameras/stream/' + cam.id + '" alt="Loading…"' +
          ' onerror="this.alt=\'Stream unavailable\'">';
        grid.appendChild(tile);
      });
    });
  }
  return { refresh: refresh };
})();

// ── DETECTION LOG ───────────────────────────────────────────────────────────
pages.logs = (function() {
  // Regular users see only their own detections — no unknown-faces tab.

  function init() {
    // Personalize the detections card title for regular users
    var title = document.querySelector("#page-logs .card-sub");
    if (title && _role === "user") {
      title.innerHTML = "My Access Log <span id='detCount' class='pill pill-blue' style='font-size:11px'></span>";
    }
    _loadChips();
    load();
  }
  var _timer      = null;
  var _activeCams = {};   // {camId: true} — null means "All"
  var _allCams    = [];   // [{id, name}, ...]

  // Build the ?camera_db_id= param from active chip selection
  function _camParam() {
    var ids = Object.keys(_activeCams).filter(function(k) { return _activeCams[k]; });
    if (ids.length === 0) return "";            // all cameras
    if (ids.length === 1) return "&camera_db_id=" + ids[0];
    return "&camera_db_id=" + ids[0];          // API only supports one at a time; first selection used
  }

  // Build query string from all active filters
  function _params(extra) {
    var limit = (qs("detLimit")||{}).value || "100";
    var name  = ((qs("detName")||{}).value || "").trim();
    var from  = (qs("detFrom")||{}).value || "";
    var to    = (qs("detTo")  ||{}).value || "";
    var s     = "limit=" + limit;
    if (name) s += "&name=" + encodeURIComponent(name);
    if (from) s += "&date_from=" + from;
    if (to)   s += "&date_to="   + to;
    s += _camParam();
    if (extra) s += extra;
    return s;
  }

  // Render camera chip strip
  function _renderChips() {
    var wrap = qs("camChips");
    if (!wrap) return;
    if (_allCams.length === 0) { wrap.innerHTML = '<span class="hint">No cameras configured</span>'; return; }

    var noFilter = Object.keys(_activeCams).every(function(k){ return !_activeCams[k]; });
    var html = '<button class="cam-chip' + (noFilter ? " active" : "") +
               '" onclick="pages.logs._chipAll()">' +
               '<span class="dot"></span>All</button>';
    _allCams.forEach(function(cam) {
      var on = !!_activeCams[String(cam.id)];
      html += '<button class="cam-chip' + (on ? " active" : "") +
              '" onclick="pages.logs._chipToggle(' + cam.id + ')">' +
              '<span class="dot"></span>' + esc(cam.name) + '</button>';
    });
    wrap.innerHTML = html;
  }

  function _chipAll() {
    _activeCams = {};
    _renderChips();
    load();
  }

  function _chipToggle(id) {
    var sid = String(id);
    _activeCams[sid] = !_activeCams[sid];
    // If everything deselected → treat as "all"
    if (Object.keys(_activeCams).every(function(k){ return !_activeCams[k]; })) _activeCams = {};
    _renderChips();
    load();
  }

  // Load cameras list then build chips
  function _loadCams() {
    if (_role === "user") return;
    api("/cameras/").then(function(data) {
      if (!data) return;
      _allCams = data.cameras || [];
      _renderChips();
    });
  }

  function load() {
    // ── recognised ──
    var logsUrl = (_role === "user") ? "/logs/my?" + _params() : "/logs/detections?" + _params();
    api(logsUrl).then(function(data) {
      if (!data) return;
      var rows = data.data || [];
      var cnt = qs("detCount");
      if (cnt) cnt.textContent = rows.length;

      qs("detTbody").innerHTML = rows.map(function(d) {
        var thumb = d.image_path
          ? '<img src="/detections/img?path=' + encodeURIComponent(d.image_path) +
            '" style="height:36px;border-radius:4px;cursor:zoom-in" onclick="showPhoto(this.src)">'
          : "—";
        return "<tr><td>" + esc(d.id||"") + "</td>" +
          "<td><strong>" + esc(d.name||"Unknown") + "</strong></td>" +
          "<td>" + esc(d.camera_name||d.camera_id||"—") + "</td>" +
          "<td style='white-space:nowrap'>" + esc(d.timestamp||"—") + "</td>" +
          "<td>" + (d.confidence != null ? Number(d.confidence).toFixed(3) : "—") + "</td>" +
          "<td>" + thumb + "</td></tr>";
      }).join("") || "<tr><td colspan='7' style='opacity:.4;text-align:center;padding:20px'>No entries</td></tr>";
    });

    // ── unknowns (staff only) ──
    if (_role === "user") return;
    api("/logs/unknown?" + _params()).then(function(data) {
      if (!data) return;
      var rows = data.data || [];
      var cnt = qs("unknownCount");
      if (cnt) cnt.textContent = rows.length;

      qs("unknownTbody").innerHTML = rows.map(function(d) {
        var thumb = d.image_path
          ? '<img src="/detections/img?path=' + encodeURIComponent(d.image_path) +
            '" style="height:36px;border-radius:4px;cursor:zoom-in" onclick="showPhoto(this.src)">'
          : "—";
        return "<tr><td>" + esc(d.id||"") + "</td>" +
          "<td>" + esc(d.camera_name||d.camera_id||"—") + "</td>" +
          "<td style='white-space:nowrap'>" + esc(d.timestamp||"—") + "</td>" +
          "<td>" + thumb + "</td></tr>";
      }).join("") || "<tr><td colspan='4' style='opacity:.4;text-align:center;padding:20px'>No entries</td></tr>";
    });
  }

  function clearFilters() {
    var n = qs("detName"); if (n) n.value = "";
    var f = qs("detFrom"); if (f) f.value = "";
    var t = qs("detTo");   if (t) t.value = "";
    _activeCams = {};
    _renderChips();
    load();
  }

  function exportXlsx() {
    var url = "/logs/export/xlsx?" + _params();
    // Add auth token as query param for download link workaround
    // (fetch + blob gives clean download without exposing token in URL history)
    if (!_token) { alert("Not logged in."); return; }
    fetch(url, { headers: { "Authorization": "Bearer " + _token } })
    .then(function(res) {
      if (!res.ok) { res.text().then(function(t){ alert("Export failed: " + t.slice(0,200)); }); return null; }
      return res.blob();
    })
    .then(function(blob) {
      if (!blob) return;
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "detections.xlsx";
      a.click();
      setTimeout(function(){ URL.revokeObjectURL(a.href); }, 3000);
    })
    .catch(function(e){ alert("Export error: " + e.message); });
  }

  function toggleAuto() {
    var on = (qs("detAutoRefresh")||{}).checked;
    if (_timer) { clearInterval(_timer); _timer = null; }
    if (on) _timer = setInterval(load, 5000);
  }

  function stopAuto() { if (_timer) { clearInterval(_timer); _timer = null; } }

  function init() { _loadCams(); load(); }

  return {
    load: load, init: init, clearFilters: clearFilters,
    exportXlsx: exportXlsx, toggleAuto: toggleAuto, stopAuto: stopAuto,
    _chipAll: _chipAll, _chipToggle: _chipToggle
  };
})();

// ── REGISTER FACE ───────────────────────────────────────────────────────────
pages.reg = (function() {
  var _ws = null, _stream = null, _video = null;
  var _rafId = null, _sendTimer = null;
  var _capturing = false, _samples = 0;
  var _TARGET = 150;
  var _dets = [];
  var _serverCamId = null;
  var _pickedUsername = "";   // unique vector-DB key (login username)
  var _pickedUserId   = null; // SQL user id

  function _mReg(t, e) { setMsg("regMsg", t, e); }
  function _mCap(t, e) { setMsg("capMsg", t, e); }

  function _progress(n) {
    _samples = n;
    var pct = Math.min(100, n / _TARGET * 100);
    var bar = qs("capBar"); if (bar) bar.style.width = pct + "%";
    var cnt = qs("capCount"); if (cnt) cnt.textContent = n + " / " + _TARGET + " anchors";
    var prog = qs("capProgress"); if (prog) prog.style.display = n > 0 ? "" : "none";
    var sv = qs("btnCapSave"); if (sv) sv.disabled = (n === 0);
  }

  function _draw() {
    var canvas = qs("regCanvas"), ctx = canvas && canvas.getContext("2d");
    if (!ctx || !_video) return;
    ctx.drawImage(_video, 0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2;
    _dets.forEach(function(d) {
      var b = d.bbox, cx = (b[0]+b[2])/2, cy = (b[1]+b[3])/2;
      var r = Math.max(b[2]-b[0], b[3]-b[1]) / 2;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.stroke();
    });
    _rafId = requestAnimationFrame(_draw);
  }

  function _sendFrame() {
    if (!_ws || _ws.readyState !== 1 || !_video) return;
    var c = qs("regCanvas"); if (!c) return;
    c.toBlob(function(blob) {
      if (!blob || !_ws || _ws.readyState !== 1) return;
      blob.arrayBuffer().then(function(buf) {
        if (_ws && _ws.readyState === 1) _ws.send(buf);
      });
    }, "image/jpeg", 0.7);
  }

  function startCam() {
    if (!window.isSecureContext &&
        location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
      _mReg("Camera requires HTTPS or localhost.", true); return;
    }
    _mReg("Opening camera…");
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function(stream) {
      _stream = stream;
      _video  = document.createElement("video");
      _video.playsInline = true; _video.muted = true;
      _video.srcObject = stream;
      return _video.play();
    })
    .then(function() {
      var proto = location.protocol === "https:" ? "wss" : "ws";
      _ws = new WebSocket(proto + "://" + location.host +
            "/clientcam/ws?token=" + encodeURIComponent(_token));
      _ws.binaryType = "arraybuffer";
      _ws.onopen = function() {
        _ws.send(JSON.stringify({ type: "settings", run_recognition: false }));
        _mReg("Camera ready. Enter a name then click ▶ Start capture.");
        qs("btnRegStart").disabled = true;
        qs("btnRegStop").disabled  = false;
        // Enable Start — user will type username in the field
        qs("btnCapStart").disabled = false;
        qs("btnCapReset").disabled = false;
        _rafId     = requestAnimationFrame(_draw);
        _sendTimer = setInterval(_sendFrame, 120);
      };
      _ws.onmessage = function(ev) {
        try {
          var m = JSON.parse(ev.data);
          if (m.type === "result") {
            _dets = m.detections || [];
          } else if (m.type === "register_frame") {
            // Server detected a face and accumulated an embedding
            _dets = m.bbox ? [{ bbox: m.bbox }] : [];
            _progress(m.samples || 0);
            if (!m.detected) _mCap("No face detected — move closer.");
            else _mCap("Capturing… " + (m.samples || 0) + " frames so far.");
          } else if (m.type === "register_saved") {
            _mCap("✅ Saved " + m.anchors_saved + " anchors for \"" + esc(m.name) + "\".");
            _progress(0);
            var n = qs("capName"); if (n) n.value = "";
            // Return to recognize mode visually
            qs("btnCapStart").disabled = false;
            qs("btnCapStop").disabled  = true;
            qs("btnCapSave").disabled  = true;
            _capturing = false;
          } else if (m.type === "ack" && m.samples !== undefined) {
            _progress(m.samples);
          }
        } catch(e) {}
      };
      _ws.onclose = function() { _mReg("Camera disconnected."); _capturing = false; };
      _ws.onerror = function() { _mReg("WebSocket error.", true); };
    })
    .catch(function(e) { _mReg("Camera error: " + e.message, true); });
  }

  function stopCam() {
    if (_rafId)     { cancelAnimationFrame(_rafId); _rafId = null; }
    if (_sendTimer) { clearInterval(_sendTimer);    _sendTimer = null; }
    if (_ws)        { _ws.close();  _ws = null; }
    if (_stream)    { _stream.getTracks().forEach(function(t){ t.stop(); }); _stream = null; }
    _video = null; _dets = []; _capturing = false;
    var s = qs("btnRegStart"); if (s) s.disabled = false;
    var p = qs("btnRegStop");  if (p) p.disabled = true;
    var c = qs("btnCapStart"); if (c) c.disabled = true;
    var d = qs("btnCapStop");  if (d) d.disabled = true;
    _mReg("Camera stopped.");
  }

  // ── Capture: client-cam mode uses WebSocket; server-cam mode uses REST ──────

  function startCapture() {
    var typed = ((qs("capName")||{}).value||"").trim();
    if (typed) _pickedUsername = typed;   // allow typing directly
    if (!_pickedUsername) { _mCap("Enter a username first.", true); return; }

    if (_source === "client") {
      if (!_ws || _ws.readyState !== 1) { _mCap("Start the camera first.", true); return; }
      _capturing = true;
      _ws.send(JSON.stringify({ type: "set_mode", mode: "register" }));
      qs("btnCapStart").disabled = true;
      qs("btnCapStop").disabled  = false;
      qs("btnCapReset").disabled = false;
      qs("btnCapSave").disabled  = true;
      _mCap("Capturing — look at the camera.");
    } else {
      var camId = (qs("regServerCam")||{}).value;
      if (!camId) { _mCap("Pick a server camera first.", true); return; }
      _serverCamId = camId;
      _mCap("Starting capture mode…");
      qs("btnCapStart").disabled = true;

      // Single call: anchors/start auto-starts the camera and enables detection
      api("/cameras/" + _serverCamId + "/anchors/start?interval_sec=0.15", { method: "POST" })
      .then(function(d) {
        if (!d) {
          // api() already showed the error; re-enable start button
          qs("btnCapStart").disabled = false;
          return;
        }
        _capturing = true;
        qs("btnCapStop").disabled  = false;
        qs("btnCapReset").disabled = false;
        qs("btnCapSave").disabled  = true;
        _mCap("🔴 Recording — stand in front of the camera, then ■ Stop.");
        _pollServer();
      });
    }
  }

  function _pollServer() {
    if (!_capturing || _source !== "server" || !_serverCamId) return;
    api("/cameras/" + _serverCamId + "/anchors/status").then(function(d) {
      if (!d) return;
      _progress(d.samples || 0);
      if (_capturing) setTimeout(_pollServer, 800);
    });
  }

  function stopCapture() {
    _capturing = false;
    if (_source === "client") {
      if (_ws && _ws.readyState === 1)
        _ws.send(JSON.stringify({ type: "set_mode", mode: "recognize" }));
      qs("btnCapStart").disabled = false;
      qs("btnCapStop").disabled  = true;
      if (_samples > 0) qs("btnCapSave").disabled = false;
      _mCap("Stopped. " + _samples + " anchors buffered. Now click 💾 Save.");
    } else {
      if (!_serverCamId) return;
      api("/cameras/" + _serverCamId + "/anchors/stop", { method: "POST" }).then(function() {
        // restore camera to normal recognition mode
        api("/cameras/" + _serverCamId + "/reg_mode?enabled=false", { method: "POST" });
        qs("btnCapStart").disabled = false;
        qs("btnCapStop").disabled  = true;
        if (_samples > 0) qs("btnCapSave").disabled = false;
        _mCap("Stopped. " + _samples + " anchors buffered. Now click 💾 Save.");
      });
    }
  }

  function resetCapture() {
    _capturing = false;
    if (_source === "client") {
      if (_ws && _ws.readyState === 1)
        _ws.send(JSON.stringify({ type: "register_reset" }));
      _progress(0);
      qs("btnCapStart").disabled = false;
      qs("btnCapStop").disabled  = true;
      qs("btnCapSave").disabled  = true;
      _mCap("Buffer cleared.");
    } else {
      if (!_serverCamId) { _progress(0); return; }
      api("/cameras/" + _serverCamId + "/anchors/reset", { method: "POST" }).then(function(d) {
        if (!d) return;
        _progress(0);
        qs("btnCapStart").disabled = false;
        qs("btnCapStop").disabled  = true;
        qs("btnCapSave").disabled  = true;
        _mCap("Buffer cleared.");
      });
    }
  }

  function saveCapture() {
    var typed = ((qs("capName")||{}).value||"").trim();
    if (typed) _pickedUsername = typed;
    if (!_pickedUsername) { _mCap("Enter a username.", true); return; }
    if (_samples === 0)   { _mCap("Nothing captured yet.", true); return; }
    if (_capturing) stopCapture();

    if (_source === "client") {
      if (!_ws || _ws.readyState !== 1) { _mCap("WebSocket closed — cannot save.", true); return; }
      _ws.send(JSON.stringify({ type: "register_save", name: _pickedUsername }));
      _mCap("Saving…");
    } else {
      if (!_serverCamId) { _mCap("No server camera selected.", true); return; }
      api("/cameras/" + _serverCamId + "/anchors/save?name=" + encodeURIComponent(_pickedUsername), { method: "POST" })
      .then(function(d) {
        if (!d) return;
        _mCap("✅ Saved " + d.anchors_saved + " anchors for \"" + esc(_pickedUsername) + "\".");
        _progress(0);
        _serverCamId = null;
      });
    }
  }

  // ── camera source switching ─────────────────────────────────────────────
  var _source = "client";

  function init() {
    _pickedUsername = ""; _pickedUserId = null;
    var inp = qs("capName"); if (inp) inp.value = "";
    qs("btnCapStart").disabled = true;
    qs("btnCapStop").disabled  = true;
    qs("btnCapReset").disabled = false;
    qs("btnCapSave").disabled  = true;
  }

  function switchSource(src) {
    _source = src;
    var isServer = src === "server";
    var cb = qs("regClientBtns"); if (cb) cb.style.display = isServer ? "none" : "";
    var cw = qs("regClientWrap"); if (cw) cw.style.display = isServer ? "none" : "";
    var sw = qs("regServerWrap"); if (sw) sw.style.display = isServer ? "" : "none";
    var sc = qs("regServerCam");  if (sc) sc.style.display = isServer ? "" : "none";
    if (isServer) {
      stopCam();
      _loadServerCams();
      qs("btnCapStart").disabled = true;   // enabled once camera is picked
      qs("btnCapStop").disabled  = true;
      qs("btnCapReset").disabled = false;
    } else {
      qs("btnCapStart").disabled = true;   // enabled after WebSocket opens
      qs("btnCapReset").disabled = false;
      var img = qs("regServerImg"); if (img) img.src = "";
    }
  }

  function _loadServerCams() {
    api("/cameras/").then(function(data) {
      if (!data) return;
      var sel = qs("regServerCam"); if (!sel) return;
      var cams = (data.cameras || []).filter(function(c){ return c.is_active; });
      sel.innerHTML = '<option value="">— pick camera —</option>' +
        cams.map(function(c){
          var statusHint = c.booting ? " ⏳" : c.streaming ? " 🟢" : " ⚫";
          return '<option value="' + c.id + '">' + esc(c.name) + statusHint + '</option>';
        }).join("");
      if (!cams.length)
        setMsg("regMsg", "No cameras configured — go to Cameras and add one.", true);
      else
        setMsg("regMsg", "Pick a camera. It will start automatically when you begin capture.");
    });
  }

  function onServerCamChange() {
    var sel = qs("regServerCam"); if (!sel) return;
    var img = qs("regServerImg"); if (!img) return;
    if (!sel.value) {
      img.src = "";
      qs("btnCapStart").disabled = true;
      return;
    }
    img.src = "/cameras/stream/" + sel.value;
    qs("btnCapStart").disabled = false;
    setMsg("regMsg", "Camera selected — enter a name and click ▶ Start capture.");
  }

  return { init: init, startCam: startCam, stopCam: stopCam,
           startCapture: startCapture, stopCapture: stopCapture,
           resetCapture: resetCapture, saveCapture: saveCapture,
           switchSource: switchSource, onServerCamChange: onServerCamChange };
})();

// ── CAMERAS ─────────────────────────────────────────────────────────────────
pages.cams = (function() {

  // Tracks which cameras are in registration mode {cam_id: true/false}
  var _regMode = {};

  function _renderRow(cam) {
    var sn = esc(cam.name);
    var inReg = !!_regMode[cam.id];
    var statusPill = cam.booting
      ? '<span class="pill" style="background:rgba(251,191,36,0.18);color:#fbbf24">⏳ Starting</span>'
      : cam.streaming
        ? '<span class="pill pill-green">🟢 Live</span>'
        : '<span class="pill pill-muted">⚫ Idle</span>';
    var streamBtn = cam.streaming
      ? '<button class="btn btn--sm btn--danger" onclick="pages.cams.stop(' + cam.id + ')">■ Stop</button>'
      : '<button class="btn btn--sm btn--primary" onclick="pages.cams.start(' + cam.id + ')">▶ Start</button>';
    var pingBtn = (cam.streaming && !cam.booting)
      ? '<button class="btn btn--sm btn--ghost" title="Check frame" onclick="pages.cams.pingCam(' + cam.id + ')">📡</button>'
      : "";
    var regToggle =
      '<label class="toggle-pill" title="Enable to use this camera for face registration">' +
        '<input type="checkbox" ' + (inReg ? "checked" : "") +
        ' onchange="pages.cams.setRegMode(' + cam.id + ',this.checked)">' +
        '<span class="toggle-track"><span class="toggle-knob"></span></span>' +
        '<span style="font-size:11px;margin-left:5px">' + (inReg ? "📷 Reg" : "Off") + '</span>' +
      '</label>';
    var rowBg = inReg
      ? ' style="background:rgba(249,115,22,0.07)"'
      : (cam.streaming ? ' style="background:rgba(59,130,246,0.06)"' : "");
    return "<tr" + rowBg + ">" +
      "<td>" + cam.id + "</td>" +
      "<td><strong>" + sn + "</strong></td>" +
      "<td><code style='font-size:11px'>" + esc(cam.source) + "</code></td>" +
      "<td>" + statusPill + "</td>" +
      "<td><label class='check'>" +
        "<input type='checkbox' " + (cam.run_recognition ? "checked" : "") +
        " onchange=\"pages.cams.setRec(" + cam.id + ",this.checked)\"/>" +
        " " + (cam.run_recognition ? "On" : "Off") +
      "</label></td>" +
      "<td>" + regToggle + "</td>" +
      "<td><div class='actions'>" +
        streamBtn + pingBtn +
        "<button class='btn btn--sm' onclick=\"pages.cams.sessions(" + cam.id + ",'" +
          cam.name.replace(/'/g, "\\'") + "')\">📋</button>" +
        "<button class='btn btn--sm btn--danger' onclick=\"pages.cams.del(" + cam.id + ",'" +
          cam.name.replace(/'/g, "\\'") + "')\">🗑</button>" +
      "</div></td></tr>";
  }

  var _qualCamData = {};

  function _fillQualCam(cameras) {
    var sel = qs("qualCam"); if (!sel) return;
    var live = cameras.filter(function(c){ return c.streaming && !c.booting; });
    _qualCamData = {};
    live.forEach(function(c){ _qualCamData[c.id] = c; });
    sel.innerHTML = live.length
      ? live.map(function(c){ return '<option value="' + c.id + '">' + esc(c.name) + '</option>'; }).join("")
      : '<option value="">No cameras streaming</option>';
    _syncThrSlider();
    var qsel = qs("qualCam");
    if (qsel && !qsel._thrBound) {
      qsel.addEventListener("change", _syncThrSlider);
      qsel._thrBound = true;
    }
  }

  function _syncThrSlider() {
    var sel = qs("qualCam"); if (!sel || !sel.value) return;
    var id  = sel.value;
    // Fetch live threshold from server — bypasses stale cam list data
    api("/cameras/" + id + "/threshold").then(function(d) {
      if (!d) return;
      var thr = parseFloat(d.similarity_threshold);
      if (isNaN(thr)) return;
      var slider = qs("qualThr"); if (slider) slider.value = Math.round(thr * 100);
      var lbl    = qs("thrVal");  if (lbl)    lbl.textContent = thr.toFixed(2);
    });
  }

  function load() {
    api("/cameras/").then(function(data) {
      if (!data) return;
      setMsg("camsMsg", "");
      var cameras = data.cameras || [];
      qs("camsTbody").innerHTML = cameras.length === 0
        ? "<tr><td colspan='7' style='opacity:.4;text-align:center;padding:20px'>No cameras — add one below</td></tr>"
        : cameras.map(_renderRow).join("");
      _fillQualCam(cameras);
      _fillMetricsCam(cameras);
      _syncThrSlider();
    });
  }

  function start(id) {
    setMsg("camsMsg", "Starting…");
    api("/cameras/" + id + "/start", { method: "POST" }).then(function(d) {
      if (d) { setMsg("camsMsg", "⏳ Starting camera — may take a few seconds."); load(); }
    });
  }

  function stop(id) {
    if (!confirm("Stop this stream?")) return;
    api("/cameras/" + id + "/stop", { method: "POST" }).then(function(d) {
      if (d) { setMsg("camsMsg", "✅ Stream stopped."); load(); }
    });
  }

  function setRec(id, enabled) {
    api("/cameras/" + id + "/recognition?enabled=" + enabled, { method: "POST" })
    .then(function(d) {
      if (!d) return;
      setMsg("camsMsg", "✅ Recognition " + (enabled ? "ON" : "OFF") + " for camera " + id + ".");
    });
  }

  function setRegMode(id, enabled) {
    _regMode[id] = enabled;
    api("/cameras/" + id + "/reg_mode?enabled=" + enabled, { method: "POST" })
    .then(function(d) {
      if (!d) { _regMode[id] = !enabled; load(); return; }
      var msg = enabled
        ? "📷 Camera " + id + " → Registration mode. Go to Register page to capture faces."
        : "✅ Camera " + id + " back to streaming mode.";
      setMsg("camsMsg", msg);
      load();
    });
  }

  function pingCam(id) {
    api("/cameras/" + id + "/ping").then(function(d) {
      if (!d) return;
      setMsg("camsMsg",
        d.alive   ? "✅ Camera " + id + " alive — frames arriving." :
        d.booting ? "⏳ Camera " + id + " still starting…" :
                    "❌ Camera " + id + " not producing frames.");
    });
  }

  function applyThreshold() {
    var id  = (qs("qualCam")||{}).value;
    var thr = Math.round(parseFloat((qs("qualThr")||{}).value || "50")) / 100;
    if (!id) { setMsg("qualMsg", "Pick a streaming camera first.", true); return; }
    api("/cameras/" + id + "/threshold?threshold=" + thr, { method: "POST" })
    .then(function(d) {
      if (!d) return;
      var live = parseFloat(d.similarity_threshold).toFixed(2);
      setMsg("qualMsg", "✅ Threshold set to " + live + " on cam #" + id);
      var lbl = qs("thrVal"); if (lbl) lbl.textContent = live;
    });
  }

  function applyQuality() {
    var id  = (qs("qualCam")||{}).value;
    var q   = parseInt((qs("qualSlider")   ||{}).value || "70");
    var w   = parseInt((qs("qualWidth")    ||{}).value || "0");
    var pw  = parseInt((qs("qualProcWidth")||{}).value || "0");
    var thr = Math.round(parseFloat((qs("qualThr")||{}).value || "50")) / 100;
    if (!id) { setMsg("qualMsg", "Pick a streaming camera first.", true); return; }
    api("/cameras/" + id + "/quality", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jpeg_quality: q, width: w, height: 0,
                             process_width: pw, similarity_threshold: thr })
    }).then(function(d) {
      if (!d) return;
      var live_thr = (d.similarity_threshold != null) ? parseFloat(d.similarity_threshold).toFixed(2) : thr.toFixed(2);
      var parts = ["✅ Q" + q, "threshold=" + live_thr];
      if (pw) parts.push("proc " + pw + "px");
      if (w)  parts.push("out " + w + "px");
      setMsg("qualMsg", parts.join(" · "));
      // Sync slider to confirmed server value
      var slider = qs("qualThr"); if (slider) slider.value = Math.round(parseFloat(live_thr) * 100);
      var lbl    = qs("thrVal");  if (lbl)    lbl.textContent = live_thr;
    });
  }

  function sessions(id, name) {
    api("/cameras/" + id + "/sessions").then(function(data) {
      if (!data) return;
      qs("sessionsCamName").textContent = "Sessions — " + name;
      qs("sessionsTbody").innerHTML = (data.sessions || []).map(function(s) {
        return "<tr><td>" + s.id + "</td><td>" + esc(s.started_at) + "</td><td>" +
          esc(s.ended_at) + "</td><td>" + esc(s.duration) + "</td>" +
          "<td>" + (s.online ? "🟢" : "⚫") + "</td></tr>";
      }).join("") || "<tr><td colspan='5' style='opacity:.4;text-align:center'>No sessions</td></tr>";
      qs("sessionsCard").style.display = "";
    });
  }

  function del(id, name) {
    if (!confirm('Delete camera "' + name + '" and all its sessions?')) return;
    api("/cameras/" + id, { method: "DELETE" }).then(function(d) {
      if (d) { setMsg("camsMsg", "✅ Deleted."); load(); }
    });
  }

  function create() {
    var name   = (qs("newCamName")||{}).value.trim();
    var source = (qs("newCamSrc") ||{}).value.trim();
    var recog  = !!(qs("newCamRec")||{}).checked;
    if (!name)   { setMsg("addCamMsg", "Name required.",   true); return; }
    if (!source) { setMsg("addCamMsg", "Source required.", true); return; }

    api("/cameras/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name, source: source, run_recognition: recog })
    }).then(function(d) {
      if (!d) return;
      setMsg("addCamMsg", "✅ Camera added: " + name);
      qs("newCamName").value = ""; qs("newCamSrc").value = "";
      load();
    });
  }

  var _metricsTimer   = null;
  var _metricsRunning = false;

  function startMetricsAuto() {
    if (_metricsTimer) return;
    _metricsRunning = true;
    _metricsTimer = setInterval(refreshMetrics, 1500);
    var btn = qs('btnMetricsAuto'); if (btn) btn.textContent = '⏸ Auto';
  }

  function stopMetricsAuto() {
    if (_metricsTimer) { clearInterval(_metricsTimer); _metricsTimer = null; }
    _metricsRunning = false;
    var btn = qs('btnMetricsAuto'); if (btn) btn.textContent = '▶ Auto';
  }

  function setThr(id) {
    var inp = qs('thr_' + id); if (!inp) return;
    var thr = parseFloat(inp.value);
    if (isNaN(thr)) return;
    thr = Math.max(0, Math.min(1, thr));
    api('/cameras/' + id + '/threshold?threshold=' + thr, { method: 'POST' })
    .then(function(d) {
      if (d) setMsg('camsMsg', '✅ Threshold → ' + parseFloat(d.similarity_threshold).toFixed(2) + ' (cam #' + id + ')');
    });
  }

  function _fillMetricsCam(cameras) {
    var sel = qs('metricsCam'); if (!sel) return;
    var live = cameras.filter(function(c){ return c.streaming && !c.booting; });
    sel.innerHTML = live.length
      ? live.map(function(c){ return '<option value="' + c.id + '">' + esc(c.name) + '</option>'; }).join('')
      : '<option value="">No cameras streaming</option>';
    var card = qs('metricsCard'); if (card) card.style.display = '';
    if (live.length) { refreshMetrics(); startMetricsAuto(); }
  }

  function refreshMetrics() {
    var sel = qs('metricsCam'); if (!sel || !sel.value) return;
    api('/cameras/' + sel.value + '/metrics').then(function(d) {
      if (!d) return;
      var body = qs('metricsBody'); if (!body) return;
      function _card(label, icon, data) {
        if (!data) return '';
        var fps  = data.fps  != null ? data.fps  + ' fps' : '—';
        var ms   = data.ms   != null ? data.ms   + ' ms'  : '—';
        return '<div style="background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:10px;padding:10px 14px">' +
          '<div style="font-size:11px;opacity:.5;margin-bottom:4px">' + icon + ' ' + label + '</div>' +
          '<div style="font-size:22px;font-weight:700;letter-spacing:-0.5px">' + fps + '</div>' +
          '<div style="font-size:12px;opacity:.55;margin-top:2px">' + ms + '</div>' +
          '</div>';
      }
      body.innerHTML =
        _card('Detection (YOLO)',   '🔍', d.detection)   +
        _card('Recognition (emb.)', '🎯', d.recognition) +
        _card('Full frame',         '🎞',  d.frame)       +
        '<div style="background:rgba(255,255,255,0.04);border:1px solid var(--border);border-radius:10px;padding:10px 14px">' +
          '<div style="font-size:11px;opacity:.5;margin-bottom:4px">⚡ Avg GPU batch</div>' +
          '<div style="font-size:22px;font-weight:700">' + (d.batch_size || '—') + '</div>' +
          '<div style="font-size:12px;opacity:.55;margin-top:2px">crops/pass</div>' +
        '</div>';
    });
  }

  function toggleMetricsAuto() {
    if (_metricsTimer) stopMetricsAuto(); else { refreshMetrics(); startMetricsAuto(); }
  }

  return { load: load, start: start, stop: stop, setRec: setRec, setRegMode: setRegMode,
           sessions: sessions, del: del, create: create,
           pingCam: pingCam, applyQuality: applyQuality, applyThreshold: applyThreshold, setThr: setThr,
           refreshMetrics: refreshMetrics, toggleMetricsAuto: toggleMetricsAuto,
           startMetricsAuto: startMetricsAuto, stopMetricsAuto: stopMetricsAuto,
           get _regMode() { return _regMode; } };
})();


// ── CHANGE PASSWORD ──────────────────────────────────────────────────────────
pages.pw = (function() {
  function save() {
    var old_p  = (qs("pwOld")  ||{}).value || "";
    var new_p  = (qs("pwNew")  ||{}).value || "";
    var new_p2 = (qs("pwNew2") ||{}).value || "";
    setMsg("pwMsg", "");

    if (!old_p)          { setMsg("pwMsg", "Enter your current password.", true); return; }
    if (new_p.length < 6){ setMsg("pwMsg", "New password min 6 characters.", true); return; }
    if (new_p !== new_p2){ setMsg("pwMsg", "New passwords do not match.", true); return; }

    api("/users/me/password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ old_password: old_p, new_password: new_p, new_password2: new_p2 })
    }).then(function(d) {
      if (!d) return;
      setMsg("pwMsg", "✅ Password changed successfully.");
      qs("pwOld").value = ""; qs("pwNew").value = ""; qs("pwNew2").value = "";
    });
  }
  return { save: save };
})();

// ── USERS ────────────────────────────────────────────────────────────────────
pages.users = (function() {
  function load() {
    api("/admin/users").then(function(data) {
      if (!data) return;
      setMsg("usersMsg", "");
      qs("usersTbody").innerHTML = (data.data || []).map(function(u) {
        var roleOpts = ["user","moderator","admin"].map(function(r) {
          return '<option value="' + r + '"' + (r===u.role?" selected":"") + '>' + r + '</option>';
        }).join("");
        var tag = '<span class="tag tag--' + u.role + '">' + u.role + '</span>';
        var dnVal = (u.display_name || u.username).replace(/'/g, "&#39;");
        return "<tr>" +
          "<td style='opacity:.5;font-size:12px'>#" + u.id + "</td>" +
          "<td style='font-size:12px;color:var(--muted)'>" + esc(u.username) + "</td>" +
          "<td><input class='input' style='padding:4px 8px;font-size:12px;width:120px' id='dname_" + u.id + "' value='" + dnVal + "'/> <button class='btn btn--sm' onclick=\"pages.users.setName(" + u.id + ")\">&#x270F;</button></td>" +
          "<td>" + tag + "</td>" +
          "<td>" + (u.is_active ? "✅" : "❌") + "</td>" +
          "<td><div class='actions'>" +
            "<select class='input' style='padding:4px 8px;font-size:12px;width:auto' id='role_" + u.id + "'>" +
              roleOpts + "</select>" +
            " <button class='btn btn--sm' onclick=\"pages.users.setRole(" + u.id + ")\">&#x1F4BE;</button>" +
            " <button class='btn btn--sm btn--danger' onclick=\"pages.users.del(" + u.id + ",'" +
              u.username.replace(/'/g, "\\\\'") + "')\">&#x1F5D1;</button>" +
          "</div></td></tr>";
      }).join("") || "<tr><td colspan='6' style='opacity:.4;text-align:center;padding:20px'>No users</td></tr>";
    });
  }

  function create() {
    var username = (qs("newUsr") ||{}).value.trim();
    var password = (qs("newPwd") ||{}).value;
    var role     = (qs("newRole")||{}).value || "user";
    if (!username) { setMsg("createUserMsg", "Username required.", true); return; }
    if (!password) { setMsg("createUserMsg", "Password required.", true); return; }

    api("/admin/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: username, password: password, role: role })
    }).then(function(d) {
      if (!d) return;
      setMsg("createUserMsg", "✅ Created: " + d.username + " (" + d.role + ")");
      qs("newUsr").value = ""; qs("newPwd").value = "";
      load();
    });
  }

  function setName(id) {
    var inp = qs("dname_" + id); if (!inp) return;
    var dn = inp.value.trim();
    if (!dn) { setMsg("usersMsg", "Display name cannot be empty.", true); return; }
    api("/admin/users/" + id + "/name", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: dn })
    }).then(function(d) {
      if (d) setMsg("usersMsg", "✅ Display name: \"" + esc(d.display_name) + "\"");
    });
  }

  function setRole(id) {
    var sel = qs("role_" + id); if (!sel) return;
    var role = sel.value;
    api("/admin/users/" + id + "/role", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role: role })
    }).then(function(d) {
      if (d) { setMsg("usersMsg", "✅ Role updated."); load(); }
    });
  }

  function del(id, username) {
    if (!confirm("Delete user \"" + username + "\"? This cannot be undone.")) return;
    api("/admin/users/" + id, { method: "DELETE" }).then(function(d) {
      if (d) { setMsg("usersMsg", "✅ Deleted: " + username); load(); }
    });
  }

  return { load: load, create: create, setRole: setRole, setName: setName, del: del };
})();

// ── FACE DATABASE ────────────────────────────────────────────────────────────
pages.facedb = (function() {
  function load() {
    api("/database/summary").then(function(data) {
      if (!data) return;
      setMsg("facedbMsg", "");
      qs("facedbTbody").innerHTML = (data.data || []).map(function(row) {
        var n = row.name.replace(/'/g,"\\'");
        var linkedCell = row.linked
          ? "<span style='color:#22c55e'>✔ " + esc(row.display_name || row.name) +
            "</span> <span style='opacity:.4;font-size:11px'>#" + row.user_id + "</span>"
          : "<span style='color:#ef4444;font-size:12px'>⚠ no user account</span>";
        return "<tr>" +
          "<td style='font-size:12px;opacity:.6'>" + esc(row.name) + "</td>" +
          "<td>" + linkedCell + "</td>" +
          "<td>" + row.count + " emb.</td>" +
          "<td><div class='actions'>" +
            "<button class='btn btn--sm btn--danger' onclick=\"pages.facedb.del('" + n + "')\">🗑 Delete</button>" +
          "</div></td></tr>";
      }).join("") || "<tr><td colspan='4' style='opacity:.4;text-align:center;padding:20px'>No faces registered</td></tr>";
    });
  }

  function rename(name) {
    var newName = prompt("Rename \"" + name + "\" to:", name);
    if (!newName || !newName.trim() || newName.trim() === name) return;
    api("/database/" + encodeURIComponent(name) + "/rename", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_name: newName.trim() })
    }).then(function(d) {
      if (d) { setMsg("facedbMsg", "✅ Renamed to: " + newName.trim()); load(); }
    });
  }

  function del(name) {
    if (!confirm("Delete ALL embeddings for \"" + name + "\"?")) return;
    api("/database/name/" + encodeURIComponent(name), { method: "DELETE" }).then(function(d) {
      if (d) { setMsg("facedbMsg", "✅ Deleted: " + name); load(); }
    });
  }

  return { load: load, rename: rename, del: del };
})();

// ── AUDIT LOG ────────────────────────────────────────────────────────────────
pages.audit = (function() {
  function load() {
    api("/admin/audit").then(function(data) {
      if (!data) return;
      qs("auditTbody").innerHTML = (data.data || []).map(function(d) {
        return "<tr>" +
          "<td>" + esc(d.id||"") + "</td>" +
          "<td><strong>" + esc(d.actor||"—") + "</strong></td>" +
          "<td>" + esc(d.action||"—") + "</td>" +
          "<td>" + esc((d.target_type||"") + (d.target_id != null ? " #"+d.target_id : "")) + "</td>" +
          "<td>" + esc(d.details||"—") + "</td>" +
          "<td style='white-space:nowrap'>" + esc(d.timestamp||"—") + "</td></tr>";
      }).join("") || "<tr><td colspan='6' style='opacity:.4;text-align:center;padding:20px'>No entries</td></tr>";
    });
  }
  return { load: load };
})();