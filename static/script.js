function openTab(evt, tabName) {
    document.querySelectorAll(".tab-content").forEach(t => t.style.display = "none");
    document.querySelectorAll(".tab-link").forEach(b => b.classList.remove("active"));
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.classList.add("active");
}

document.addEventListener("DOMContentLoaded", () => {
    document.querySelector(".tab-link").click();
    setupSliders();
    imageHandler();
    videoHandler();
    webcamHandler();
});

// -------- SLIDER --------
function setupSliders() {
    document.querySelectorAll(".conf-slider").forEach(s => {
        const span = document.getElementById(`${s.id}-value`);
        span.textContent = s.value;
        s.addEventListener("input", () => span.textContent = s.value);
    });
}

// -------- IMAGE --------
function imageHandler() {
    const form = document.getElementById("image-form");
    const out = document.getElementById("image-results");

    form.onsubmit = e => {
        e.preventDefault();

        const fd = new FormData(form);
        fd.append("confidence", document.getElementById("image-conf").value);

        out.innerHTML = "Processing image...";

        fetch("/detect_image", { method: "POST", body: fd })
            .then(r => r.json())
            .then(d => {
                out.innerHTML = `
                    <p>${d.message}</p>

                    <div class="image-compare">
                        <div>
                            <h4>Original Image</h4>
                            <img src="${d.original}">
                        </div>

                        <div>
                            <h4>Processed Image</h4>
                            <img src="${d.processed}">
                        </div>
                    </div>
                `;
            });
    };
}

// -------- VIDEO --------
function videoHandler() {
    const form = document.getElementById("video-form");
    const out = document.getElementById("video-results");

    form.onsubmit = e => {
        e.preventDefault();

        const fd = new FormData(form);
        fd.append("confidence", document.getElementById("video-conf").value);

        out.innerHTML = "Processing video...";

        fetch("/detect_video", { method: "POST", body: fd })
            .then(() => {
                out.innerHTML = `
                    <video controls autoplay muted width="100%">
                        <source src="/play_video" type="video/mp4">
                    </video>
                `;
            });
    };
}

// -------- WEBCAM --------
function webcamHandler() {
    const start = document.getElementById("start-webcam");
    const stop = document.getElementById("stop-webcam");
    const feed = document.getElementById("webcam-feed");
    const box = document.getElementById("webcam-feed-container");

    start.onclick = () => {
        box.style.display = "block";
        feed.src = `/webcam_feed?confidence=${document.getElementById("webcam-conf").value}&t=${Date.now()}`;
        start.style.display = "none";
        stop.style.display = "inline-block";
    };

    stop.onclick = () => {
        fetch("/stop_webcam", { method: "POST" });
        feed.src = "";
        box.style.display = "none";
        stop.style.display = "none";
        start.style.display = "inline-block";
    };
}