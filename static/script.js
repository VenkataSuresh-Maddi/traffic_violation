/* ═══════════════════════════════════════════════════════════════════
   TRAFFIC VIOLATION DETECTION — Frontend Logic
   Confidence is fixed at 0.25 server-side; no slider needed.
   ═══════════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", function () {

    // ── Live Clock ────────────────────────────────────────────────────
    function tickClock() {
        var el = document.getElementById("live-clock");
        if (!el) return;
        var now = new Date();
        var pad = function(n) { return String(n).padStart(2, "0"); };
        el.textContent =
            now.getFullYear() + "-" + pad(now.getMonth()+1) + "-" + pad(now.getDate()) +
            "  " + pad(now.getHours()) + ":" + pad(now.getMinutes()) + ":" + pad(now.getSeconds());
    }
    setInterval(tickClock, 1000);
    tickClock();


    // ── Tabs ──────────────────────────────────────────────────────────
    var tabBtns   = document.querySelectorAll(".tab-btn");
    var tabPanels = document.querySelectorAll(".tab-panel");

    function activateTab(btn) {
        tabBtns.forEach(function(b) { b.classList.remove("active"); });
        tabPanels.forEach(function(p) { p.classList.remove("active"); });
        btn.classList.add("active");
        var target = document.getElementById(btn.dataset.tab);
        if (target) target.classList.add("active");
    }

    tabBtns.forEach(function(btn) {
        btn.addEventListener("click", function() { activateTab(btn); });
    });

    if (tabBtns.length > 0) activateTab(tabBtns[0]);


    // ── File Input Labels ─────────────────────────────────────────────
    function bindFileLabel(inputId, labelId) {
        var input = document.getElementById(inputId);
        var label = document.getElementById(labelId);
        if (!input || !label) return;
        input.addEventListener("change", function() {
            label.textContent = input.files[0] ? input.files[0].name : "No file selected";
        });
    }
    bindFileLabel("image-file-input", "image-filename");
    bindFileLabel("video-file-input", "video-filename");


    // ── Stats HTML builder ────────────────────────────────────────────
    function buildStatsHTML(stats, message) {
        var total      = (stats && stats.total)      || 0;
        var violations = (stats && stats.violations) || 0;
        var safe       = (stats && stats.safe)       || 0;
        var pct        = total > 0 ? Math.round((violations / total) * 100) : 0;
        var pctClass   = pct > 30 ? "high" : "low";
        var msgClass   = violations > 0 ? "warning" : "success";

        var msgHTML = "";
        if (message) {
            msgHTML = '<div class="alert-msg ' + msgClass + '">' + message + '</div>';
        }

        return msgHTML +
            '<div class="stats-wrap">' +
                '<div class="stats-heading">&#9658; Detection Summary</div>' +
                '<table class="stats-table">' +
                    '<thead><tr>' +
                        '<th>Total Vehicles</th>' +
                        '<th>Followed Rule</th>' +
                        '<th>Violated Rule</th>' +
                    '</tr></thead>' +
                    '<tbody><tr>' +
                        '<td class="cell-total">' + total + '</td>' +
                        '<td class="cell-safe">' + safe + '</td>' +
                        '<td class="cell-viol">' + violations +
                            ' <span class="viol-pct ' + pctClass + '">' + pct + '%</span>' +
                        '</td>' +
                    '</tr></tbody>' +
                '</table>' +
                '<div class="metric-row">' +
                    '<div class="metric-pill total"><div class="pill-num">' + total + '</div><div class="pill-label">Total Bikes</div></div>' +
                    '<div class="metric-pill safe"><div class="pill-num">' + safe + '</div><div class="pill-label">Wore Helmet</div></div>' +
                    '<div class="metric-pill danger"><div class="pill-num">' + violations + '</div><div class="pill-label">No Helmet</div></div>' +
                '</div>' +
            '</div>';
    }


    // ── Image Handler ─────────────────────────────────────────────────
    var imageForm = document.getElementById("image-form");
    if (imageForm) {
        imageForm.addEventListener("submit", function(e) {
            e.preventDefault();

            var resultsEl = document.getElementById("image-results");
            var submitBtn = document.getElementById("image-submit-btn");

            submitBtn.disabled = true;
            submitBtn.textContent = "ANALYSING...";
            resultsEl.innerHTML = '<div class="alert-msg info">Processing image, please wait...</div>';

            var fd = new FormData(this);
            fd.set("confidence", "0.25");

            fetch("/detect_image", { method: "POST", body: fd })
                .then(function(r) {
                    if (!r.ok) throw new Error("Server error " + r.status);
                    return r.json();
                })
                .then(function(d) {
                    resultsEl.innerHTML =
                        buildStatsHTML(d.stats || {}, d.message || "") +
                        '<div class="compare-grid">' +
                            '<div class="compare-item">' +
                                '<div class="compare-header"><h4>Original</h4>' +
                                '<button class="fs-btn" onclick="openFullscreen(this.parentElement.nextElementSibling)">&#x26F6; Fullscreen</button></div>' +
                                '<img src="' + d.original + '" alt="Original" onclick="openFullscreen(this)">' +
                            '</div>' +
                            '<div class="compare-item">' +
                                '<div class="compare-header"><h4>Processed</h4>' +
                                '<button class="fs-btn" onclick="openFullscreen(this.parentElement.nextElementSibling)">&#x26F6; Fullscreen</button></div>' +
                                '<img src="' + d.processed + '" alt="Processed" onclick="openFullscreen(this)">' +
                            '</div>' +
                        '</div>';
                })
                .catch(function(err) {
                    console.error("Image error:", err);
                    resultsEl.innerHTML = '<div class="alert-msg warning">Error: ' + (err.message || "Could not process image.") + '</div>';
                })
                .finally(function() {
                    submitBtn.disabled = false;
                    submitBtn.textContent = "ANALYSE IMAGE";
                });
        });
    }


    // ── Video Handler ─────────────────────────────────────────────────
    var videoForm = document.getElementById("video-form");
    if (videoForm) {
        videoForm.addEventListener("submit", function(e) {
            e.preventDefault();

            var resultsEl = document.getElementById("video-results");
            var submitBtn = document.getElementById("video-submit-btn");

            submitBtn.disabled = true;
            submitBtn.textContent = "PROCESSING...";

            resultsEl.innerHTML =
                '<div class="progress-wrap">' +
                    '<div class="progress-header">' +
                        '<span class="progress-label">PROCESSING VIDEO</span>' +
                        '<span class="progress-pct mono" id="prog-pct">0%</span>' +
                    '</div>' +
                    '<div class="progress-track"><div class="progress-fill" id="prog-fill"></div></div>' +
                '</div>' +
                '<div class="alert-msg info">This may take a while depending on video length...</div>';

            var fd = new FormData(this);
            fd.set("confidence", "0.25");

            var poll = setInterval(function() {
                fetch("/video_progress")
                    .then(function(r) { return r.json(); })
                    .then(function(d) {
                        var fill = document.getElementById("prog-fill");
                        var pct  = document.getElementById("prog-pct");
                        if (fill) fill.style.width = d.percent + "%";
                        if (pct)  pct.textContent  = d.percent + "%";
                    })
                    .catch(function() {});
            }, 600);

            fetch("/detect_video", { method: "POST", body: fd })
                .then(function(r) {
                    if (!r.ok) throw new Error("Server error " + r.status);
                    return r.json();
                })
                .then(function(d) {
                    clearInterval(poll);
                    var fill = document.getElementById("prog-fill");
                    var pct  = document.getElementById("prog-pct");
                    if (fill) fill.style.width = "100%";
                    if (pct)  pct.textContent  = "100%";

                    setTimeout(function() {
                        resultsEl.innerHTML =
                            buildStatsHTML(d.stats || {}, null) +
                            '<video class="video-player" controls autoplay muted>' +
                                '<source src="/play_video?t=' + Date.now() + '" type="video/mp4">' +
                            '</video>';
                    }, 500);
                })
                .catch(function(err) {
                    clearInterval(poll);
                    console.error("Video error:", err);
                    resultsEl.innerHTML = '<div class="alert-msg warning">Error: ' + (err.message || "Could not process video.") + '</div>';
                })
                .finally(function() {
                    submitBtn.disabled = false;
                    submitBtn.textContent = "PROCESS VIDEO";
                });
        });
    }


    // ── Webcam Handler ────────────────────────────────────────────────
    var startBtn = document.getElementById("start-webcam");
    var stopBtn  = document.getElementById("stop-webcam");

    if (startBtn) {
        startBtn.addEventListener("click", function() {
            var feed  = document.getElementById("webcam-feed");
            var box   = document.getElementById("webcam-feed-container");
            var empty = document.getElementById("webcam-empty");

            if (feed)  feed.src           = "/webcam_feed?confidence=0.25&t=" + Date.now();
            if (box)   box.style.display   = "block";
            if (empty) empty.style.display = "none";
            startBtn.style.display = "none";
            if (stopBtn) stopBtn.style.display = "flex";
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener("click", function() {
            var feed  = document.getElementById("webcam-feed");
            var box   = document.getElementById("webcam-feed-container");
            var empty = document.getElementById("webcam-empty");

            fetch("/stop_webcam", { method: "POST" }).catch(function() {});
            if (feed)  feed.src           = "";
            if (box)   box.style.display   = "none";
            if (empty) empty.style.display = "flex";
            stopBtn.style.display = "none";
            if (startBtn) startBtn.style.display = "flex";
        });
    }

    // ── Image Fullscreen ─────────────────────────────────────────────
    window.openFullscreen = function(el) {
        if (!el) return;
        if (el.requestFullscreen) el.requestFullscreen();
        else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen();
        else if (el.mozRequestFullScreen) el.mozRequestFullScreen();
    };

}); // end DOMContentLoaded