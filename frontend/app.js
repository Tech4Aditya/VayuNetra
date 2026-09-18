/* ============================================================
   VAYUNETRA — CYCLONE INTELLIGENCE CONSOLE
   Frontend Controller
   ============================================================ */


/* ============================================================
   CONFIG
   ============================================================ */

const API_BASE = "http://127.0.0.1:8000";

const $ = (id) =>
    document.getElementById(id);


/* ============================================================
   SAFE DOM HELPERS
   ============================================================ */

function text(id, value) {

    const element = $(id);

    if (!element) return;

    element.textContent =
        value === undefined ||
        value === null
            ? "—"
            : String(value);
}


function width(id, value) {

    const element = $(id);

    if (!element) return;

    element.style.width = `${value}%`;
}


function html(id, value) {

    const element = $(id);

    if (!element) return;

    element.innerHTML = value;
}


/* ============================================================
   STATUS
   ============================================================ */

function status(message, success = false) {

    text(
        "statusText",
        message
    );

    const element =
        $("statusText");

    if (!element) return;

    element.style.color =
        success
            ? "var(--green)"
            : "var(--muted)";
}


/* ============================================================
   TIME
   ============================================================ */

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

    text(
        "clock",
        `${now} IST`
    );
}


/* ============================================================
   SATELLITE COLOUR MAP
   ============================================================ */

const STOPS = [

    {
        t: 0.00,
        c: [10, 10, 14]
    },

    {
        t: 0.35,
        c: [90, 90, 90]
    },

    {
        t: 0.55,
        c: [220, 220, 220]
    },

    {
        t: 0.65,
        c: [255, 255, 0]
    },

    {
        t: 0.75,
        c: [255, 140, 0]
    },

    {
        t: 0.85,
        c: [255, 0, 0]
    },

    {
        t: 1.00,
        c: [160, 0, 200]
    }
];


function satelliteColor(value) {

    let v =
        Number(value);

    if (!Number.isFinite(v)) {
        v = 0;
    }

    v =
        Math.max(
            0,
            Math.min(
                1,
                v
            )
        );

    for (
        let i = 0;
        i < STOPS.length - 1;
        i++
    ) {

        const a =
            STOPS[i];

        const b =
            STOPS[i + 1];

        if (
            v >= a.t &&
            v <= b.t
        ) {

            const factor =
                (v - a.t) /
                (b.t - a.t || 1);

            return a.c.map(
                (x, j) =>
                    Math.round(
                        x +
                        factor *
                        (b.c[j] - x)
                    )
            );
        }
    }

    return STOPS[
        STOPS.length - 1
    ].c;
}


/* ============================================================
   LEGEND
   ============================================================ */

function buildLegend() {

    const legend =
        $("legendBar");

    if (!legend) return;

    legend.innerHTML = "";

    for (
        let i = 0;
        i <= 50;
        i++
    ) {

        const element =
            document.createElement(
                "div"
            );

        const color =
            satelliteColor(
                i / 50
            );

        element.style.background =
            `rgb(${color[0]}, ${color[1]}, ${color[2]})`;

        legend.appendChild(
            element
        );
    }
}


/* ============================================================
   DRAW TCIR FRAME
   Backend frame format:
   [height][width][4]
   Channel order:
   IR, WV, VIS, PMW
   ============================================================ */

function drawFrame(frames) {

    if (
        !Array.isArray(frames) ||
        frames.length === 0
    ) {
        return;
    }

    const frame =
        frames[
            frames.length - 1
        ];

    if (
        !Array.isArray(frame) ||
        frame.length === 0
    ) {
        return;
    }

    const canvas =
        $("frameCanvas");

    if (!canvas) return;

    const ctx =
        canvas.getContext("2d");

    if (!ctx) return;

    const height =
        frame.length;

    const widthValue =
        Array.isArray(frame[0])
            ? frame[0].length
            : 0;

    if (
        widthValue === 0
    ) {
        return;
    }

    canvas.width =
        widthValue;

    canvas.height =
        height;

    const imageData =
        ctx.createImageData(
            widthValue,
            height
        );

    for (
        let y = 0;
        y < height;
        y++
    ) {

        for (
            let x = 0;
            x < widthValue;
            x++
        ) {

            const pixel =
                frame[y][x];

            let value;

            /*
             * TCIR frame is normally:
             * [IR, WV, VIS, PMW]
             */

            if (
                Array.isArray(pixel)
            ) {

                /*
                 * Use IR channel.
                 */

                value =
                    Number(
                        pixel[0]
                    );

            } else {

                value =
                    Number(pixel);
            }

            /*
             * Backend returns values
             * normalized between 0 and 1
             * for temporal inference.
             *
             * If a raw 0-255 value somehow
             * arrives, normalize it here.
             */

            if (
                Number.isFinite(value) &&
                value > 1
            ) {

                value /=
                    255;
            }

            const rgb =
                satelliteColor(
                    value
                );

            const index =
                (
                    y *
                    widthValue +
                    x
                ) * 4;

            imageData.data[index] =
                rgb[0];

            imageData.data[index + 1] =
                rgb[1];

            imageData.data[index + 2] =
                rgb[2];

            imageData.data[index + 3] =
                255;
        }
    }

    ctx.putImageData(
        imageData,
        0,
        0
    );

    text(
        "frameSource",
        `TCIR REAL SATELLITE · FRAME ${frames.length}`
    );
}


/* ============================================================
   UPLOAD PREVIEW
   ============================================================ */

function previewImage(file) {

    if (!file) return;

    const image =
        new Image();

    image.onload =
        function () {

            const canvas =
                $("frameCanvas");

            if (!canvas) return;

            const ctx =
                canvas.getContext("2d");

            const size =
                256;

            canvas.width =
                size;

            canvas.height =
                size;

            ctx.clearRect(
                0,
                0,
                size,
                size
            );

            ctx.drawImage(
                image,
                0,
                0,
                size,
                size
            );

            URL.revokeObjectURL(
                image.src
            );
        };

    image.src =
        URL.createObjectURL(
            file
        );
}


/*
 * Compatibility alias.
 */

function uploadPreview(file) {

    previewImage(file);
}


/* ============================================================
   ESCAPE HTML
   ============================================================ */

function escapeHtml(value) {

    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}


/* ============================================================
   PROBABILITY BARS
   ============================================================ */

function renderBars(
    containerId,
    probabilities
) {

    const box =
        $(containerId);

    if (!box) return;

    box.innerHTML = "";

    if (!probabilities) {
        return;
    }

    const entries =
        Object.entries(
            probabilities
        );

    if (
        entries.length === 0
    ) {
        return;
    }

    entries
        .sort(
            (a, b) =>
                Number(b[1]) -
                Number(a[1])
        )
        .forEach(
            ([label, probability]) => {

                const pct =
                    Math.max(
                        0,
                        Math.min(
                            100,
                            Number(
                                probability
                            ) * 100
                        )
                    );

                const row =
                    document.createElement(
                        "div"
                    );

                row.className =
                    "bar-row";

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

                box.appendChild(
                    row
                );
            }
        );
}


/*
 * Compatibility alias.
 */

function bars(
    containerId,
    probabilities
) {

    renderBars(
        containerId,
        probabilities
    );
}


/* ============================================================
   CLEAR RESULTS
   ============================================================ */

function clearResults() {

    text(
        "activeCyclones",
        "—"
    );

    text(
        "cycloneName",
        "—"
    );

    text(
        "cycloneCategory",
        "—"
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
        "threshold",
        "0.50"
    );

    text(
        "predCategory",
        "—"
    );

    text(
        "predCategorySecondary",
        "—"
    );

    html(
        "categoryBars",
        ""
    );

    text(
        "intensityValue",
        "—"
    );

    text(
        "pressureValue",
        "—"
    );

    text(
        "sizeValue",
        "—"
    );

    text(
        "intensityNote",
        "No intensity estimate available."
    );

    text(
        "predTrend",
        "—"
    );

    html(
        "trendBars",
        ""
    );

    text(
        "trackDelta",
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
        "frameSource",
        "NO DATA"
    );
}


/*
 * Compatibility alias.
 */

function clearUI() {

    clearResults();
}


/* ============================================================
   FETCH WITH TIMEOUT
   ============================================================ */

async function fetchWithTimeout(
    url,
    options = {},
    timeout = 15000
) {

    const controller =
        new AbortController();

    const timer =
        setTimeout(
            () =>
                controller.abort(),
            timeout
        );

    try {

        return await fetch(
            url,
            {
                ...options,
                signal:
                    controller.signal
            }
        );

    } finally {

        clearTimeout(
            timer
        );
    }
}


/* ============================================================
   RENDER REAL TCIR PREDICTION
   ============================================================ */

function renderPrediction(
    prediction
) {

    if (!prediction) {
        return;
    }

    console.log(
        "Rendering prediction:",
        prediction
    );


    /* ========================================================
       IDENTIFICATION
       ======================================================== */

    const identification =
        prediction.identification ||
        {};

    let presence =
        Number(
            identification
                .cyclone_present_probability
        );

    /*
     * The real TCIR temporal model is
     * an intensity model, not a trained
     * presence detector.
     *
     * Therefore don't fabricate a
     * presence probability.
     */

    if (
        !Number.isFinite(
            presence
        )
    ) {

        presence =
            null;
    }

    const threshold =
        Number(
            identification.threshold ??
            0.5
        );


    if (
        presence !== null
    ) {

        text(
            "presence",
            `${(
                presence * 100
            ).toFixed(1)}%`
        );

        width(
            "presenceBar",
            Math.max(
                0,
                Math.min(
                    100,
                    presence * 100
                )
            )
        );

        const detected =
            identification
                .cyclone_present ??
            presence >= threshold;

        text(
            "presenceState",
            detected
                ? "DETECTED"
                : "CLEAR"
        );

        const state =
            $("presenceState");

        if (state) {

            state.style.color =
                detected
                    ? "var(--green)"
                    : "var(--dim)";
        }

    } else {

        text(
            "presence",
            "N/A"
        );

        width(
            "presenceBar",
            0
        );

        text(
            "presenceState",
            "INTENSITY MODEL"
        );
    }


    text(
        "threshold",
        threshold.toFixed(2)
    );


    /* ========================================================
       CLASSIFICATION
       ======================================================== */

    const classification =
        prediction.classification ||
        {};

    const category =
        classification
            .predicted_category ??
        "—";

    text(
        "predCategory",
        category
    );

    text(
        "predCategorySecondary",
        category
    );

    /*
     * If the backend provides probabilities,
     * show them.
     *
     * Otherwise explain that category is
     * deterministically derived from wind.
     */

    if (
        classification
            .category_probabilities
    ) {

        renderBars(
            "categoryBars",
            classification
                .category_probabilities
        );

    } else {

        html(
            "categoryBars",
            `
            <div class="note">
                Category derived from predicted
                maximum sustained wind using
                IMD intensity thresholds.
            </div>
            `
        );
    }


    /* ========================================================
       INTENSITY
       ======================================================== */

    const intensity =
        prediction.intensity ||
        {};

    const wind =
        Number(
            intensity.wind_kt
        );

    const pressure =
        Number(
            intensity.pressure_hpa
        );

    const size =
        Number(
            intensity.size_nmi
        );


    if (
        Number.isFinite(
            wind
        )
    ) {

        text(
            "intensityValue",
            `${wind.toFixed(1)} kt`
        );

        text(
            "cycloneWind",
            `${wind.toFixed(1)} kt`
        );

    } else {

        text(
            "intensityValue",
            "—"
        );

        text(
            "cycloneWind",
            "—"
        );
    }


    if (
        Number.isFinite(
            pressure
        )
    ) {

        text(
            "pressureValue",
            `${pressure.toFixed(1)} hPa`
        );

    } else {

        text(
            "pressureValue",
            "—"
        );
    }


    if (
        Number.isFinite(
            size
        )
    ) {

        text(
            "sizeValue",
            `${size.toFixed(1)} nmi`
        );

    } else {

        text(
            "sizeValue",
            "—"
        );
    }


    text(
        "intensityNote",
        intensity.note ??
        "TCIR temporal model prediction."
    );


    /* ========================================================
       TEMPORAL CONTEXT
       ======================================================== */

    const temporalContext =
        prediction.temporal_context ||
        {};

    const sequenceLength =
        Number(
            temporalContext
                .sequence_length ??
            4
        );

    const intervalHours =
        Number(
            temporalContext
                .interval_hours ??
            3
        );

    const historyHours =
        Number(
            temporalContext
                .history_hours ??
            (
                (sequenceLength - 1) *
                intervalHours
            )
        );


    text(
        "framesReceived",
        sequenceLength
    );

    text(
        "framesReceivedSecondary",
        sequenceLength
    );

    text(
        "temporalMode",
        `${sequenceLength}-FRAME / ${intervalHours}H`
    );

    text(
        "temporalModeBottom",
        `${sequenceLength}-FRAME · ${historyHours}H HISTORY`
    );


    /* ========================================================
       TREND
       ======================================================== */

    /*
     * Current TCIR temporal model predicts
     * intensity/pressure/size.
     *
     * It does NOT output a temporal trend
     * classification or track delta.
     */

    text(
        "predTrend",
        "INTENSITY FORECAST"
    );

    html(
        "trendBars",
        `
        <div class="note">
            Temporal CNN + GRU uses
            ${sequenceLength} chronological frames
            across ${historyHours} hours of history.
            Current model outputs intensity,
            pressure and size rather than
            a categorical trend.
        </div>
        `
    );

    text(
        "trackDelta",
        "Not predicted"
    );


    /* ========================================================
       PRIMARY DASHBOARD
       ======================================================== */

    text(
        "activeCyclones",
        "01"
    );

    text(
        "cycloneCategory",
        category
    );


    /* ========================================================
       MODEL INFO
       ======================================================== */

    if (
        prediction.model
    ) {

        console.log(
            "Model:",
            prediction.model
        );
    }

    if (
        prediction.checkpoint
    ) {

        console.log(
            "Checkpoint:",
            prediction.checkpoint
        );
    }
}


/* ============================================================
   REAL TCIR DEMO
   ============================================================ */

async function runDemo() {

    clearResults();

    status(
        "Loading real TCIR demonstration sequence…"
    );

    try {

        /* ----------------------------------------------------
           STEP 1 — GET REAL TCIR SEQUENCE
           ---------------------------------------------------- */

        const response =
            await fetchWithTimeout(
                `${API_BASE}/tcir_demo`,
                {
                    cache: "no-store"
                },
                15000
            );

        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                `TCIR demo HTTP ${response.status}: ${errorText}`
            );
        }

        const demo =
            await response.json();

        console.log(
            "TCIR demo response:",
            demo
        );


        if (
            !Array.isArray(
                demo.frames
            ) ||
            demo.frames.length === 0
        ) {

            throw new Error(
                "Backend returned no TCIR frames."
            );
        }


        /* ----------------------------------------------------
           STEP 2 — VALIDATE SEQUENCE
           ---------------------------------------------------- */

        if (
            demo.frames.length !== 4
        ) {

            console.warn(
                `Expected 4 frames, received ${demo.frames.length}`
            );
        }


        const frameCount =
            demo.frames.length;


        text(
            "framesReceived",
            frameCount
        );

        text(
            "framesReceivedSecondary",
            frameCount
        );


        text(
            "temporalMode",
            `${frameCount}-FRAME`
        );


        status(
            `Loaded ${frameCount} real TCIR frames. Running temporal analysis…`
        );


        /* ----------------------------------------------------
           STEP 3 — DRAW LATEST FRAME
           ---------------------------------------------------- */

        drawFrame(
            demo.frames
        );


        /* ----------------------------------------------------
           STEP 4 — SEND ACTUAL FRAMES
           ----------------------------------------------------

           IMPORTANT:

           TCIRSequenceIn expects:

               frames: list

           NOT:

               h5_indices

           ---------------------------------------------------- */

        const predictionResponse =
            await fetchWithTimeout(
                `${API_BASE}/predict_tcir_sequence`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        frames:
                            demo.frames
                    })
                },
                30000
            );


        if (
            !predictionResponse.ok
        ) {

            const errorText =
                await predictionResponse.text();

            throw new Error(
                `TCIR prediction HTTP ${predictionResponse.status}: ${errorText}`
            );
        }


        /* ----------------------------------------------------
           STEP 5 — READ PREDICTION
           ---------------------------------------------------- */

        const prediction =
            await predictionResponse.json();

        console.log(
            "TCIR prediction response:",
            prediction
        );


        /* ----------------------------------------------------
           STEP 6 — RENDER MODEL OUTPUT
           ---------------------------------------------------- */

        renderPrediction(
            prediction
        );


        /* ----------------------------------------------------
           STEP 7 — DEMO METADATA
           ---------------------------------------------------- */

        const cycloneId =
            demo.cyclone_id ??
            demo.temporal_context?.cyclone_id ??
            "TCIR TEST SAMPLE";


        const timestamp =
            demo.target_timestamp ??
            demo.timestamp ??
            demo.temporal_context?.target_timestamp ??
            "—";


        text(
            "cycloneName",
            cycloneId
        );


        text(
            "frameSource",
            `TCIR · ${timestamp}`
        );


        /*
         * TCIR sample itself establishes
         * that this is a cyclone test sample.
         *
         * This is NOT a learned presence detector.
         */

        text(
            "presence",
            "KNOWN"
        );

        width(
            "presenceBar",
            100
        );

        text(
            "presenceState",
            "TCIR TEST SAMPLE"
        );


        /* ----------------------------------------------------
           STEP 8 — GROUND TRUTH
           ---------------------------------------------------- */

        const groundTruth =
            demo.ground_truth ||
            {};


        if (
            groundTruth.category !==
            undefined
        ) {

            text(
                "gtCategory",
                groundTruth.category
            );

        } else {

            text(
                "gtCategory",
                "TCIR reference"
            );
        }


        if (
            groundTruth.trend !==
            undefined
        ) {

            text(
                "gtTrend",
                groundTruth.trend
            );

        } else {

            text(
                "gtTrend",
                "—"
            );
        }


        /* ----------------------------------------------------
           STEP 9 — TEMPORAL METADATA
           ---------------------------------------------------- */

        const temporalContext =
            prediction.temporal_context ||
            demo.temporal_context ||
            {};


        const interval =
            Number(
                temporalContext
                    .interval_hours ??
                3
            );


        const history =
            Number(
                temporalContext
                    .history_hours ??
                (
                    (frameCount - 1) *
                    interval
                )
            );


        text(
            "temporalMode",
            `${frameCount}-FRAME / ${interval}H`
        );


        text(
            "temporalModeBottom",
            `${frameCount}-FRAME · ${history}H HISTORY`
        );


        text(
            "framesReceived",
            frameCount
        );


        text(
            "framesReceivedSecondary",
            frameCount
        );


        /* ----------------------------------------------------
           STEP 10 — HDF5 INDEX DEBUG INFO
           ---------------------------------------------------- */

        if (
            demo.temporal_context &&
            Array.isArray(
                demo.temporal_context
                    .h5_indices
            )
        ) {

            console.log(
                "TCIR HDF5 indices:",
                demo.temporal_context
                    .h5_indices
            );
        }


        /* ----------------------------------------------------
           STEP 11 — SUCCESS
           ---------------------------------------------------- */

        status(
            "TCIR temporal analysis complete.",
            true
        );


        console.log(
            "TCIR analysis completed successfully:",
            {
                frames:
                    frameCount,

                cyclone_id:
                    cycloneId,

                timestamp:
                    timestamp,

                prediction:
                    prediction,

                ground_truth:
                    groundTruth,

                temporal_context:
                    temporalContext
            }
        );


    } catch (error) {

        console.error(
            "TCIR demo error:",
            error
        );


        if (
            error.name ===
            "AbortError"
        ) {

            status(
                "TCIR analysis timed out."
            );

        } else {

            status(
                `Analysis failed: ${error.message}`
            );
        }


        const state =
            $("systemState");

        if (state) {

            state.textContent =
                "NODE OFFLINE";
        }


        const dot =
            $("systemDot");

        if (dot) {

            dot.style.background =
                "var(--red)";
        }
    }
}


/* ============================================================
   IMAGE UPLOAD
   ============================================================ */

async function processUpload() {

    const input =
        $("imageInput");

    if (!input) {

        status(
            "Image input not found."
        );

        return;
    }


    const files =
        Array.from(
            input.files || []
        );


    if (
        files.length === 0
    ) {

        status(
            "Select at least one image."
        );

        return;
    }


    clearResults();


    status(
        `Processing ${files.length} image(s)…`
    );


    const formData =
        new FormData();


    for (
        let i = 0;
        i < files.length;
        i++
    ) {

        formData.append(
            "images",
            files[i]
        );
    }


    try {

        const response =
            await fetchWithTimeout(
                `${API_BASE}/predict_image`,
                {
                    method: "POST",
                    body: formData
                },
                30000
            );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                errorText ||
                `HTTP ${response.status}`
            );
        }


        const prediction =
            await response.json();


        console.log(
            "Upload prediction:",
            prediction
        );


        /* Show last uploaded image */

        uploadPreview(
            files[
                files.length - 1
            ]
        );


        /* Render prediction */

        renderPrediction(
            prediction
        );


        /* Metadata */

        text(
            "frameSource",
            `EXTERNAL IMAGE · ${files.length} FRAME(S)`
        );


        text(
            "gtCategory",
            "— external image"
        );


        text(
            "gtTrend",
            "—"
        );


        const received =
            prediction.frames_received ??
            files.length;


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
            "temporalModeBottom",
            files.length > 1
                ? "EXTERNAL MULTI-FRAME"
                : "EXTERNAL SINGLE FRAME"
        );


        status(
            "Image analysis complete.",
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
   BACKEND HEALTH
   ============================================================ */

async function checkBackend() {

    try {

        const response =
            await fetchWithTimeout(
                `${API_BASE}/health`,
                {
                    cache: "no-store"
                },
                5000
            );


        if (!response.ok) {

            throw new Error(
                `Health HTTP ${response.status}`
            );
        }


        const data =
            await response.json();


        text(
            "systemState",
            data?.status
                ? String(
                    data.status
                ).toUpperCase()
                : "NODE ONLINE"
        );


        text(
            "systemDetail",
            "127.0.0.1:8000"
        );


        const dot =
            $("systemDot");

        if (dot) {

            dot.style.background =
                "var(--green)";
        }


        status(
            "Backend online · models responding.",
            true
        );


        console.log(
            "Backend health:",
            data
        );


    } catch (error) {

        console.error(
            "Health check:",
            error
        );


        text(
            "systemState",
            "NODE OFFLINE"
        );


        text(
            "systemDetail",
            "127.0.0.1:8000"
        );


        const dot =
            $("systemDot");

        if (dot) {

            dot.style.background =
                "var(--red)";
        }


        status(
            "Backend offline."
        );
    }
}


/* ============================================================
   BUTTON SETUP
   ============================================================ */

function setupButtons() {


    /* DEMO */

    const demoButton =
        $("demoBtn");

    if (demoButton) {

        demoButton.addEventListener(
            "click",
            runDemo
        );
    }


    /* PROCESS */

    const processButton =
        $("processBtn");

    if (processButton) {

        processButton.addEventListener(
            "click",
            processUpload
        );
    }


    /* REFRESH */

    const refreshButton =
        $("refreshBtn");

    if (refreshButton) {

        refreshButton.addEventListener(
            "click",
            checkBackend
        );
    }


    /* SETTINGS */

    const settingsButton =
        $("settingsBtn");

    if (settingsButton) {

        settingsButton.addEventListener(
            "click",
            () => {

                status(
                    "Settings module coming soon."
                );
            }
        );
    }


    /* IMAGE INPUT */

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
                    files.length === 0
                ) {
                    return;
                }


                previewImage(
                    files[
                        files.length - 1
                    ]
                );


                text(
                    "frameSource",
                    `READY · ${files.length} IMAGE(S)`
                );


                status(
                    `${files.length} image(s) ready`
                );
            }
        );
    }
}


/* ============================================================
   INITIALIZATION
   ============================================================ */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        console.log(
            "VayuNetra frontend loaded."
        );


        buildLegend();


        updateClock();


        setInterval(
            updateClock,
            1000
        );


        setupButtons();


        /*
         * Only perform health check
         * automatically.
         *
         * Demo inference happens only
         * when the user clicks:
         * "View Live Analysis".
         */

        checkBackend();
    }
);