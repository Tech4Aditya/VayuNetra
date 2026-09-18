const API_BASE = "http://localhost:8000";

const $ = (id) => document.getElementById(id);

const STOPS = [
    { t: 0.00, c: [10, 10, 14] },
    { t: 0.35, c: [90, 90, 90] },
    { t: 0.55, c: [220, 220, 220] },
    { t: 0.65, c: [255, 255, 0] },
    { t: 0.75, c: [255, 140, 0] },
    { t: 0.85, c: [255, 0, 0] },
    { t: 1.00, c: [160, 0, 200] }
];


function color(v) {
    v = Math.max(0, Math.min(1, Number(v) || 0));

    for (let i = 0; i < STOPS.length - 1; i++) {
        const a = STOPS[i];
        const b = STOPS[i + 1];

        if (v >= a.t && v <= b.t) {
            const f = (v - a.t) / (b.t - a.t || 1);

            return a.c.map((x, j) =>
                Math.round(x + f * (b.c[j] - x))
            );
        }
    }

    return STOPS[STOPS.length - 1].c;
}


function buildLegend() {
    const legend = $("legendBar");

    if (!legend) return;

    legend.innerHTML = "";

    for (let i = 0; i <= 50; i++) {
        const div = document.createElement("div");
        const c = color(i / 50);

        div.style.background = `rgb(${c[0]}, ${c[1]}, ${c[2]})`;

        legend.appendChild(div);
    }
}


function draw(frames) {
    if (!Array.isArray(frames) || frames.length === 0) {
        return;
    }

    const frame = frames[frames.length - 1];

    if (!Array.isArray(frame) || frame.length === 0) {
        return;
    }

    const canvas = $("frameCanvas");

    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    const size = frame.length;

    canvas.width = size;
    canvas.height = size;

    const imageData = ctx.createImageData(size, size);

    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {

            const value = frame[y][x];

            const rgb = color(value);

            const index = (y * size + x) * 4;

            imageData.data[index] = rgb[0];
            imageData.data[index + 1] = rgb[1];
            imageData.data[index + 2] = rgb[2];
            imageData.data[index + 3] = 255;
        }
    }

    ctx.putImageData(imageData, 0, 0);

    $("frameSource").textContent =
        `SYNTHETIC DEMO · FRAME ${frames.length}`;
}


function uploadPreview(file) {

    if (!file) return;

    const image = new Image();

    image.onload = function () {

        const canvas = $("frameCanvas");
        const ctx = canvas.getContext("2d");

        canvas.width = 64;
        canvas.height = 64;

        ctx.clearRect(0, 0, 64, 64);

        ctx.drawImage(
            image,
            0,
            0,
            64,
            64
        );

        URL.revokeObjectURL(image.src);
    };

    image.src = URL.createObjectURL(file);
}


function renderBars(containerId, probabilities) {

    const box = $(containerId);

    if (!box) return;

    box.innerHTML = "";

    if (!probabilities) return;

    const entries = Object.entries(probabilities);

    if (entries.length === 0) return;

    entries
        .sort((a, b) => Number(b[1]) - Number(a[1]))
        .forEach(([label, probability]) => {

            const pct = Math.max(
                0,
                Math.min(
                    100,
                    Number(probability) * 100
                )
            );

            const row = document.createElement("div");

            row.className = "bar-row";

            row.innerHTML = `
                <div class="bar-top">
                    <span>${escapeHtml(label)}</span>
                    <span>${pct.toFixed(1)}%</span>
                </div>

                <div class="bar-bg">
                    <div
                        class="bar-fill"
                        style="width:${pct.toFixed(1)}%"
                    ></div>
                </div>
            `;

            box.appendChild(row);
        });
}


function escapeHtml(value) {

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function clearUI() {

    $("presence").textContent = "—";

    $("presenceBar").style.width = "0%";

    $("presenceState").textContent = "WAITING";

    $("predCategory").textContent = "—";

    $("categoryBars").innerHTML = "";

    $("intensityValue").textContent = "—";

    $("intensityNote").textContent =
        "No intensity estimate available.";

    $("predTrend").textContent = "—";

    $("trendBars").innerHTML = "";

    $("trackDelta").textContent = "—";
}


function renderPrediction(pred) {

    if (!pred) return;

    // -----------------------------
    // DETECTION
    // -----------------------------

    const presence = Number(
        pred?.identification?.cyclone_present_probability ?? 0
    );

    const threshold = Number(
        pred?.identification?.threshold ?? 0.5
    );

    const cyclonePresent =
        pred?.identification?.cyclone_present ??
        (presence >= threshold);

    $("presence").textContent =
        `${(presence * 100).toFixed(1)}%`;

    $("presenceBar").style.width =
        `${Math.min(100, Math.max(0, presence * 100))}%`;

    $("threshold").textContent =
        threshold.toFixed(2);

    $("presenceState").textContent =
        cyclonePresent ? "DETECTED" : "CLEAR";

    // FIXED:
    // var(--green) -> var(--seafoam)
    // var(--dim)   -> var(--text-dim)

    $("presenceState").style.color =
        cyclonePresent
            ? "var(--seafoam)"
            : "var(--text-dim)";


    // -----------------------------
    // CLASSIFICATION
    // -----------------------------

    if (cyclonePresent) {

        $("predCategory").textContent =
            pred?.classification?.predicted_category ?? "—";

        renderBars(
            "categoryBars",
            pred?.classification?.category_probabilities ?? {}
        );

    } else {

        $("predCategory").textContent =
            "No Cyclone Detected";

        $("categoryBars").innerHTML =
            `<p class="note">
                Category suppressed below detection threshold.
            </p>`;
    }


    // -----------------------------
    // INTENSITY
    // -----------------------------

    const intensity =
        pred?.intensity?.normalized_intensity;

    if (typeof intensity === "number") {

        $("intensityValue").textContent =
            intensity.toFixed(4);

    } else {

        $("intensityValue").textContent =
            "—";
    }

    $("intensityNote").textContent =
        pred?.intensity?.note ??
        "Normalized model output; not a wind-speed measurement.";


    // -----------------------------
    // TEMPORAL PREDICTION
    // -----------------------------

    const prediction =
        pred?.prediction ?? {};

    if (prediction.reliable !== true) {

        $("predTrend").textContent =
            "INSUFFICIENT DATA";

        $("trendBars").innerHTML =
            `<p class="note">
                Upload 2+ chronological frames
                for temporal analysis.
            </p>`;

        $("trackDelta").textContent =
            "Unavailable";

        return;
    }


    const trend =
        prediction.trend ?? "—";

    $("predTrend").textContent =
        trend;

    renderBars(
        "trendBars",
        prediction.trend_probabilities ?? {}
    );


    const delta =
        prediction.predicted_next_step_track_delta;

    if (
        Array.isArray(delta) &&
        delta.length >= 2 &&
        Number.isFinite(Number(delta[0])) &&
        Number.isFinite(Number(delta[1]))
    ) {

        $("trackDelta").textContent =
            `(${Number(delta[0]).toFixed(3)}, ${Number(delta[1]).toFixed(3)})`;

    } else {

        $("trackDelta").textContent =
            "Unavailable";
    }
}


function setStatus(message, success = false) {

    const status = $("statusText");

    status.textContent = message;

    // FIXED:
    // var(--green) -> var(--seafoam)
    // var(--muted) -> var(--text-muted)

    status.style.color =
        success
            ? "var(--seafoam)"
            : "var(--text-muted)";
}


async function runDemo() {

    clearUI();

    setStatus(
        "Fetching demonstration sequence…"
    );

    try {

        const response =
            await fetch(
                `${API_BASE}/demo_sequence`
            );

        if (!response.ok) {

            throw new Error(
                `demo_sequence failed: ${response.status}`
            );
        }

        const demo =
            await response.json();


        // Draw satellite frame

        draw(demo.frames);


        // Ground truth

        $("gtCategory").textContent =
            demo.true_category ?? "—";

        $("gtTrend").textContent =
            demo.true_trend ?? "—";

        $("framesReceived").textContent =
            demo.frames?.length ?? "—";

        $("temporalMode").textContent =
            (demo.frames?.length || 0) > 1
                ? "MULTI-FRAME"
                : "SINGLE";

        $("sequenceTag").textContent =
            "DEMO SEQUENCE";


        setStatus(
            "Running local inference…"
        );


        // Prediction

        const predictionResponse =
            await fetch(
                `${API_BASE}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        frames: demo.frames
                    })
                }
            );


        if (!predictionResponse.ok) {

            throw new Error(
                `predict failed: ${predictionResponse.status}`
            );
        }


        const prediction =
            await predictionResponse.json();


        renderPrediction(prediction);


        setStatus(
            "Models responding · demo complete",
            true
        );


    } catch (error) {

        console.error(
            "Demo error:",
            error
        );

        setStatus(
            "Backend unavailable — check localhost:8000"
        );

        $("systemState").textContent =
            "NODE OFFLINE";

        $("systemDot").style.background =
            "var(--red)";
    }
}


async function processUpload() {

    const input =
        $("imageInput");

    const files =
        input.files;


    if (!files || files.length === 0) {

        setStatus(
            "Select at least one image."
        );

        return;
    }


    clearUI();

    setStatus(
        `Processing ${files.length} image(s)…`
    );


    const formData =
        new FormData();


    for (let i = 0; i < files.length; i++) {

        formData.append(
            "images",
            files[i]
        );
    }


    try {

        const response =
            await fetch(
                `${API_BASE}/predict_image`,
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                errorText
            );
        }


        const prediction =
            await response.json();


        // Show last uploaded image

        uploadPreview(
            files[files.length - 1]
        );


        renderPrediction(
            prediction
        );


        $("frameSource").textContent =
            `EXTERNAL IMAGE · ${files.length} FRAME(S)`;


        $("gtCategory").textContent =
            "— external image";

        $("gtTrend").textContent =
            "—";

        $("framesReceived").textContent =
            prediction.frames_received ??
            files.length;

        $("temporalMode").textContent =
            files.length > 1
                ? "MULTI-FRAME"
                : "SINGLE";

        $("sequenceTag").textContent =
            "UPLOAD";


        setStatus(
            "Upload processed",
            true
        );


    } catch (error) {

        console.error(
            "Upload error:",
            error
        );

        setStatus(
            "Upload failed — check backend."
        );
    }
}


async function checkBackend() {

    try {

        const response =
            await fetch(
                `${API_BASE}/health`
            );


        if (!response.ok) {

            throw new Error(
                "Backend health check failed"
            );
        }


        const data =
            await response.json();


        $("systemState").textContent =
            data?.status
                ? String(data.status).toUpperCase()
                : "NODE ONLINE";


        $("systemDetail").textContent =
            "localhost:8000";


        // FIXED:
        // var(--green) -> var(--seafoam)

        $("systemDot").style.background =
            "var(--seafoam)";


        setStatus(
            "Backend online",
            true
        );


    } catch (error) {

        console.error(
            "Health check:",
            error
        );

        $("systemState").textContent =
            "NODE OFFLINE";

        $("systemDetail").textContent =
            "localhost:8000";

        $("systemDot").style.background =
            "var(--red)";

        setStatus(
            "Backend offline"
        );
    }
}


function updateClock() {

    const now =
        new Intl.DateTimeFormat(
            "en-IN",
            {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: false,
                timeZone: "Asia/Kolkata"
            }
        ).format(new Date());


    $("clock").textContent =
        `${now} IST`;
}


// -----------------------------
// INITIALIZATION
// -----------------------------

document.addEventListener(
    "DOMContentLoaded",
    function () {

        buildLegend();

        updateClock();

        setInterval(
            updateClock,
            1000
        );


        $("demoBtn").addEventListener(
            "click",
            runDemo
        );


        $("processBtn").addEventListener(
            "click",
            processUpload
        );


        $("refreshBtn").addEventListener(
            "click",
            checkBackend
        );


        checkBackend();

        runDemo();
    }
);