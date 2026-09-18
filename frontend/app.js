/* ============================================================
   VAYUNETRA — VAYU DRISHTI
   SAFE FRONTEND CONTROLLER
   ============================================================ */

const API_BASE = "http://localhost:8000";

const $ = id => document.getElementById(id);


/* ============================================================
   SAFE DOM
   ============================================================ */

function text(id, value) {
    const el = $(id);
    if (el) {
        el.textContent = value ?? "—";
    }
}

function width(id, value) {
    const el = $(id);
    if (!el) return;

    const n = Math.max(
        0,
        Math.min(100, Number(value) || 0)
    );

    el.style.width = `${n}%`;
}

function html(id, value) {
    const el = $(id);
    if (el) {
        el.innerHTML = value || "";
    }
}


/* ============================================================
   STATUS
   ============================================================ */

function status(message, good = false) {

    const el = $("statusText");

    if (!el) return;

    el.textContent = message;

    el.style.color =
        good
            ? "#159b96"
            : "#71807d";
}


/* ============================================================
   CLOCK
   ============================================================ */

function clock() {

    const el = $("clock");

    if (!el) return;

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

    el.textContent = `${now} IST`;
}


/* ============================================================
   SATELLITE COLOURS
   ============================================================ */

const STOPS = [
    [0.00, [10, 10, 14]],
    [0.35, [90, 90, 90]],
    [0.55, [220, 220, 220]],
    [0.65, [255, 255, 0]],
    [0.75, [255, 140, 0]],
    [0.85, [255, 0, 0]],
    [1.00, [160, 0, 200]]
];

function getColor(value) {

    let v = Number(value);

    if (!Number.isFinite(v)) {
        v = 0;
    }

    v = Math.max(0, Math.min(1, v));

    for (
        let i = 0;
        i < STOPS.length - 1;
        i++
    ) {

        const a = STOPS[i];
        const b = STOPS[i + 1];

        if (
            v >= a[0] &&
            v <= b[0]
        ) {

            const ratio =
                (v - a[0]) /
                (b[0] - a[0]);

            return [
                Math.round(
                    a[1][0] +
                    ratio *
                    (b[1][0] - a[1][0])
                ),

                Math.round(
                    a[1][1] +
                    ratio *
                    (b[1][1] - a[1][1])
                ),

                Math.round(
                    a[1][2] +
                    ratio *
                    (b[1][2] - a[1][2])
                )
            ];
        }
    }

    return [160, 0, 200];
}


/* ============================================================
   LEGEND
   ============================================================ */

function buildLegend() {

    const el = $("legendBar");

    if (!el) return;

    el.innerHTML = "";

    for (let i = 0; i < 32; i++) {

        const block =
            document.createElement("div");

        const [r, g, b] =
            getColor(i / 31);

        block.style.background =
            `rgb(${r},${g},${b})`;

        el.appendChild(block);
    }
}


/* ============================================================
   DRAW SATELLITE FRAME
   ============================================================ */

function drawFrame(frames) {

    if (
        !Array.isArray(frames) ||
        frames.length === 0
    ) {
        return;
    }

    const canvas =
        $("frameCanvas");

    if (!canvas) return;

    const frame =
        frames[frames.length - 1];

    if (
        !Array.isArray(frame) ||
        frame.length === 0
    ) {
        return;
    }

    const ctx =
        canvas.getContext("2d");

    if (!ctx) return;

    const height =
        frame.length;

    const width =
        Array.isArray(frame[0])
            ? frame[0].length
            : 128;

    /*
     * Safety guard.
     * Never allow a malformed backend response
     * to allocate a ridiculous canvas.
     */

    if (
        width <= 0 ||
        height <= 0 ||
        width > 1024 ||
        height > 1024
    ) {
        console.warn(
            "Invalid satellite frame dimensions:",
            width,
            height
        );

        return;
    }

    canvas.width = width;
    canvas.height = height;

    const image =
        ctx.createImageData(
            width,
            height
        );

    for (
        let y = 0;
        y < height;
        y++
    ) {

        const row = frame[y];

        if (!Array.isArray(row)) {
            continue;
        }

        for (
            let x = 0;
            x < width;
            x++
        ) {

            const value =
                Number(row[x] ?? 0);

            const [r, g, b] =
                getColor(value);

            const index =
                (y * width + x) * 4;

            image.data[index] = r;
            image.data[index + 1] = g;
            image.data[index + 2] = b;
            image.data[index + 3] = 255;
        }
    }

    ctx.putImageData(
        image,
        0,
        0
    );

    text(
        "frameSource",
        `SATELLITE · ${frames.length} FRAME(S)`
    );
}


/* ============================================================
   FETCH WITH TIMEOUT
   ============================================================ */

async function fetchWithTimeout(
    url,
    options = {},
    timeout = 10000
) {

    const controller =
        new AbortController();

    const timer =
        setTimeout(
            () => controller.abort(),
            timeout
        );

    try {

        const response =
            await fetch(
                url,
                {
                    ...options,
                    signal:
                        controller.signal
                }
            );

        return response;

    } finally {

        clearTimeout(timer);
    }
}


/* ============================================================
   BACKEND HEALTH
   ============================================================ */

async function checkBackend() {

    status(
        "Checking local inference node…"
    );

    try {

        const response =
            await fetchWithTimeout(
                `${API_BASE}/health`,
                {
                    cache: "no-store"
                },
                3000
            );

        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}`
            );
        }

        const data =
            await response.json();

        text(
            "systemState",
            data.status
                ? String(
                    data.status
                ).toUpperCase()
                : "NODE ONLINE"
        );

        const dot =
            $("systemDot");

        if (dot) {
            dot.style.background =
                "#239b70";
        }

        status(
            "Backend online",
            true
        );

    } catch (error) {

        console.warn(
            "Backend health check failed:",
            error
        );

        text(
            "systemState",
            "NODE OFFLINE"
        );

        const dot =
            $("systemDot");

        if (dot) {
            dot.style.background =
                "#d54b4b";
        }

        status(
            "Backend unavailable — check localhost:8000"
        );
    }
}


/* ============================================================
   CLEAR RESULTS
   ============================================================ */

function clearResults() {

    text(
        "presence",
        "—"
    );

    width(
        "presenceBar",
        0
    );

    text(
        "presenceState",
        "WAITING"
    );

    text(
        "predCategory",
        "—"
    );

    text(
        "predCategorySecondary",
        "—"
    );

    text(
        "predTrend",
        "—"
    );

    text(
        "intensityValue",
        "—"
    );

    text(
        "trackDelta",
        "—"
    );

    text(
        "activeCyclones",
        "—"
    );

    text(
        "cycloneCategory",
        "Awaiting analysis"
    );

    text(
        "cycloneWind",
        "—"
    );

    text(
        "cycloneLat",
        "—"
    );

    text(
        "framesReceived",
        "—"
    );

    text(
        "framesReceivedSecondary",
        "—"
    );

    text(
        "temporalMode",
        "—"
    );

    text(
        "temporalModeBottom",
        "—"
    );

    text(
        "gtCategory",
        "—"
    );

    text(
        "gtTrend",
        "—"
    );

    text(
        "intensityNote",
        "Waiting for model inference."
    );

    html(
        "categoryBars",
        ""
    );

    html(
        "trendBars",
        ""
    );
}


/* ============================================================
   ESCAPE HTML
   ============================================================ */

function escapeHTML(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


/* ============================================================
   BARS
   ============================================================ */

function bars(
    containerId,
    probabilities
) {

    const container =
        $(containerId);

    if (!container) return;

    container.innerHTML = "";

    if (
        !probabilities ||
        typeof probabilities !== "object"
    ) {
        return;
    }

    const entries =
        Object.entries(probabilities)
            .sort(
                (a, b) =>
                    Number(b[1]) -
                    Number(a[1])
            );

    for (
        const [label, value]
        of entries
    ) {

        const pct =
            Math.max(
                0,
                Math.min(
                    100,
                    Number(value) * 100
                )
            );

        const row =
            document.createElement("div");

        row.className =
            "bar-row";

        row.innerHTML = `
            <div class="top">
                <span>${escapeHTML(label)}</span>
                <span class="pct">
                    ${pct.toFixed(1)}%
                </span>
            </div>

            <div class="bar-bg">
                <div
                    class="bar-fill"
                    style="width:${pct}%"
                ></div>
            </div>
        `;

        container.appendChild(row);
    }
}


/* ============================================================
   RENDER PREDICTION
   ============================================================ */

function renderPrediction(data) {

    if (!data) return;

    console.log(
        "Prediction:",
        data
    );

    const identification =
        data.identification || {};

    const classification =
        data.classification || {};

    const intensity =
        data.intensity || {};

    const prediction =
        data.prediction || {};


    /* DETECTION */

    const probability =
        Number(
            identification
                .cyclone_present_probability
                ?? 0
        );

    const detected =
        identification.cyclone_present
            !== undefined
            ? Boolean(
                identification.cyclone_present
            )
            : probability >= 0.5;


    text(
        "presence",
        `${(
            probability * 100
        ).toFixed(1)}%`
    );

    width(
        "presenceBar",
        probability * 100
    );

    text(
        "presenceState",
        detected
            ? "DETECTED"
            : "CLEAR"
    );


    /* CATEGORY */

    const category =
        classification
            .predicted_category
            ?? "—";

    text(
        "predCategory",
        detected
            ? category
            : "No Cyclone Detected"
    );

    text(
        "predCategorySecondary",
        detected
            ? category
            : "CLEAR SCENE"
    );


    if (detected) {

        bars(
            "categoryBars",
            classification
                .category_probabilities
                ?? {}
        );

    } else {

        html(
            "categoryBars",
            `
            <div class="note">
                Category suppressed below
                detection threshold.
            </div>
            `
        );
    }


    /* INTENSITY */

    const normalized =
        intensity.normalized_intensity;

    if (
        typeof normalized === "number" &&
        Number.isFinite(normalized)
    ) {

        text(
            "intensityValue",
            normalized.toFixed(4)
        );
    }


    text(
        "intensityNote",
        intensity.note
            ??
        "Normalized model output."
    );


    /* TEMPORAL */

    if (
        prediction.reliable === true
    ) {

        text(
            "predTrend",
            prediction.trend
                ?? "—"
        );

        bars(
            "trendBars",
            prediction
                .trend_probabilities
                ?? {}
        );


        const delta =
            prediction
                .predicted_next_step_track_delta;


        if (
            Array.isArray(delta) &&
            delta.length >= 2
        ) {

            const dx =
                Number(delta[0]);

            const dy =
                Number(delta[1]);

            if (
                Number.isFinite(dx) &&
                Number.isFinite(dy)
            ) {

                text(
                    "trackDelta",
                    `(${dx.toFixed(3)}, ${dy.toFixed(3)})`
                );
            }
        }

    } else {

        text(
            "predTrend",
            "INSUFFICIENT DATA"
        );

        text(
            "trackDelta",
            "Unavailable"
        );

        html(
            "trendBars",
            `
            <div class="note">
                Multiple chronological frames
                are required.
            </div>
            `
        );
    }


    /* PRIMARY CARD */

    text(
        "activeCyclones",
        detected
            ? "01"
            : "00"
    );

    text(
        "cycloneCategory",
        detected
            ? category
            : "No cyclone detected"
    );
}


/* ============================================================
   DEMO
   ============================================================ */

async function runDemo() {

    clearResults();

    status(
        "Loading satellite demonstration…"
    );

    try {

        const response =
            await fetchWithTimeout(
                `${API_BASE}/demo_sequence`,
                {
                    cache: "no-store"
                },
                10000
            );


        if (!response.ok) {

            throw new Error(
                `Demo HTTP ${response.status}`
            );
        }


        const demo =
            await response.json();


        console.log(
            "Demo response:",
            demo
        );


        if (
            !Array.isArray(
                demo.frames
            )
        ) {

            throw new Error(
                "Backend returned invalid frames."
            );
        }


        drawFrame(
            demo.frames
        );


        const count =
            demo.frames.length;


        text(
            "framesReceived",
            count
        );

        text(
            "framesReceivedSecondary",
            count
        );

        text(
            "temporalMode",
            count > 1
                ? "MULTI-FRAME"
                : "SINGLE"
        );

        text(
            "temporalModeBottom",
            count > 1
                ? "MULTI-FRAME"
                : "SINGLE"
        );

        text(
            "sequenceTag",
            "DEMO SEQUENCE"
        );


        text(
            "gtCategory",
            demo.true_category
                ?? "—"
        );

        text(
            "gtTrend",
            demo.true_trend
                ?? "—"
        );


        status(
            "Running local inference…"
        );


        /*
         * IMPORTANT:
         * Timeout prevents a broken backend/model
         * from freezing the user experience.
         */

        const predictionResponse =
            await fetchWithTimeout(
                `${API_BASE}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            frames:
                                demo.frames
                        })
                },
                15000
            );


        if (!predictionResponse.ok) {

            throw new Error(
                `Prediction HTTP ${predictionResponse.status}`
            );
        }


        const prediction =
            await predictionResponse.json();


        renderPrediction(
            prediction
        );


        status(
            "Analysis complete",
            true
        );


    } catch (error) {

        console.error(
            "Demo failed:",
            error
        );


        if (
            error.name ===
            "AbortError"
        ) {

            status(
                "Backend request timed out — page remains responsive."
            );

        } else {

            status(
                "Analysis failed — check backend console."
            );
        }
    }
}


/* ============================================================
   IMAGE UPLOAD
   ============================================================ */

function previewImage(file) {

    if (!file) return;

    const canvas =
        $("frameCanvas");

    if (!canvas) return;

    const image =
        new Image();

    const url =
        URL.createObjectURL(file);


    image.onload = () => {

        const ctx =
            canvas.getContext("2d");

        if (!ctx) {
            URL.revokeObjectURL(url);
            return;
        }

        canvas.width =
            image.naturalWidth || 128;

        canvas.height =
            image.naturalHeight || 128;

        ctx.drawImage(
            image,
            0,
            0,
            canvas.width,
            canvas.height
        );

        URL.revokeObjectURL(url);
    };


    image.onerror = () => {
        URL.revokeObjectURL(url);
    };


    image.src = url;
}


async function processUpload() {

    const input =
        $("imageInput");

    if (!input) {

        status(
            "Upload control not found."
        );

        return;
    }


    const files =
        Array.from(
            input.files || []
        );


    if (files.length === 0) {

        status(
            "Select a satellite image first."
        );

        return;
    }


    previewImage(
        files[files.length - 1]
    );


    status(
        `Processing ${files.length} image(s)…`
    );


    const form =
        new FormData();


    /*
     * Keep the same field name used by
     * the existing backend.
     */

    for (
        const file of files
    ) {

        form.append(
            "images",
            file
        );
    }


    try {

        const response =
            await fetchWithTimeout(
                `${API_BASE}/predict_image`,
                {
                    method: "POST",
                    body: form
                },
                20000
            );


        if (!response.ok) {

            throw new Error(
                `Upload HTTP ${response.status}`
            );
        }


        const result =
            await response.json();


        renderPrediction(
            result
        );


        const received =
            result.frames_received
                ?? files.length;


        text(
            "framesReceived",
            received
        );

        text(
            "framesReceivedSecondary",
            received
        );

        text(
            "temporalMode",
            files.length > 1
                ? "MULTI-FRAME"
                : "SINGLE"
        );


        text(
            "sequenceTag",
            "UPLOAD"
        );


        text(
            "gtCategory",
            "External image"
        );


        text(
            "gtTrend",
            "—"
        );


        text(
            "frameSource",
            `UPLOAD · ${files.length} IMAGE(S)`
        );


        status(
            "Image analysis complete",
            true
        );


    } catch (error) {

        console.error(
            "Upload error:",
            error
        );


        if (
            error.name ===
            "AbortError"
        ) {

            status(
                "Upload timed out."
            );

        } else {

            status(
                "Upload failed — check backend console."
            );
        }
    }
}


/* ============================================================
   BUTTONS
   ============================================================ */

function setupButtons() {

    const demo =
        $("demoBtn");

    if (demo) {

        demo.addEventListener(
            "click",
            runDemo
        );
    }


    const process =
        $("processBtn");

    if (process) {

        process.addEventListener(
            "click",
            processUpload
        );
    }


    const refresh =
        $("refreshBtn");

    if (refresh) {

        refresh.addEventListener(
            "click",
            checkBackend
        );
    }


    const settings =
        $("settingsBtn");

    if (settings) {

        settings.addEventListener(
            "click",
            () => {

                status(
                    "Settings module coming soon."
                );
            }
        );
    }


    const input =
        $("imageInput");

    if (input) {

        input.addEventListener(
            "change",
            () => {

                const files =
                    Array.from(
                        input.files || []
                    );

                if (
                    files.length > 0
                ) {

                    previewImage(
                        files[files.length - 1]
                    );

                    text(
                        "frameSource",
                        `READY · ${files.length} IMAGE(S)`
                    );

                    status(
                        `${files.length} image(s) ready`
                    );
                }
            }
        );
    }
}


/* ============================================================
   START
   ============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        console.log(
            "Vayu Drishti frontend loaded."
        );

        buildLegend();

        clock();

        setInterval(
            clock,
            1000
        );

        setupButtons();

        /*
         * Only health check automatically.
         * NO demo inference on page load.
         */

        checkBackend();
    }
);