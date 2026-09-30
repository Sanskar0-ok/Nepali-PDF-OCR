const fileInput = document.getElementById("pdfFile");
const convertBtn = document.getElementById("convertBtn");
const viewer = document.getElementById("viewer");
const status = document.getElementById("status");
const fileName = document.getElementById("fileName");


let currentPages = [];


/* =========================================================
   FILE SELECTION
   ========================================================= */

fileInput.addEventListener("change", () => {

    const file = fileInput.files[0];

    if (!file) {
        fileName.textContent = "No file selected";
        return;
    }

    fileName.textContent = file.name;
});


/* =========================================================
   OPEN PDF
   ========================================================= */

convertBtn.addEventListener("click", async () => {

    const file = fileInput.files[0];

    if (!file) {

        status.textContent =
            "Please select a PDF first.";

        return;
    }


    convertBtn.disabled = true;

    status.textContent =
        "Processing PDF...";

    viewer.innerHTML = "";

    currentPages = [];


    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );


    try {

        const response =
            await fetch(
                "/convert",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );
        }


        const data =
            await response.json();


        if (data.error) {

            status.textContent =
                "Error: " + data.error;

            return;
        }


        if (
            !data.pages ||
            !Array.isArray(data.pages)
        ) {

            throw new Error(
                "Invalid response from server."
            );
        }


        currentPages =
            data.pages;


        status.textContent =
            `Loaded ${data.pages.length} page(s).`;


        renderPages(
            data.pages
        );


    } catch (error) {

        console.error(error);

        status.textContent =
            "Something went wrong while processing the PDF.";

    } finally {

        convertBtn.disabled = false;

    }
});


/* =========================================================
   RENDER ALL PAGES
   ========================================================= */

function renderPages(pages) {

    viewer.innerHTML = "";


    pages.forEach(
        (pageData, pageIndex) => {

            const page =
                createPage(
                    pageData,
                    pageIndex
                );

            viewer.appendChild(
                page
            );

        }
    );
}


/* =========================================================
   CREATE PAGE
   ========================================================= */

function createPage(
    pageData,
    pageIndex
) {

    const page =
        document.createElement("div");


    page.className =
        "page";


    const pageWidth =
        getSafeNumber(
            pageData.width,
            1
        );


    const pageHeight =
        getSafeNumber(
            pageData.height,
            1
        );


    page.style.width =
        `${pageWidth}px`;


    page.style.height =
        `${pageHeight}px`;


    page.dataset.page =
        pageData.page ||
        pageIndex + 1;


    const textLayer =
        document.createElement("div");


    textLayer.className =
        "text-layer";


    textLayer.style.width =
        `${pageWidth}px`;


    textLayer.style.height =
        `${pageHeight}px`;


    buildTextLines(
        textLayer,
        pageData
    );


    page.appendChild(
        textLayer
    );


    return page;
}


/* =========================================================
   BUILD TEXT LINES
   ========================================================= */

function buildTextLines(
    textLayer,
    pageData
) {

    if (
        !pageData.lines ||
        !Array.isArray(pageData.lines)
    ) {
        return;
    }


    pageData.lines.forEach(
        line => {

            const element =
                createTextLine(
                    line,
                    pageData
                );


            textLayer.appendChild(
                element
            );

        }
    );
}


/* =========================================================
   CREATE ONE TEXT LINE
   ========================================================= */

function createTextLine(line, pageData) {

    const element = document.createElement("div");

    element.className = "text-line";

    const x = Math.max(
        0,
        getSafeNumber(line.x, 0)
    );

    const y = Math.max(
        0,
        getSafeNumber(line.y, 0)
    );

    const originalWidth = Math.max(
        1,
        getSafeNumber(line.width, 1)
    );

    const height = Math.max(
        1,
        getSafeNumber(line.height, 12)
    );

    const pageWidth = Math.max(
        1,
        getSafeNumber(pageData.width, x + originalWidth)
    );

    /*
     * NEVER allow the text box to extend outside
     * the right edge of the PDF page.
     */
    const width = Math.min(
        originalWidth,
        pageWidth - x
    );

    element.style.left = `${x}px`;
    element.style.top = `${y}px`;

    element.style.width = `${Math.max(1, width)}px`;

    /*
     * Give every PDF line exactly its own vertical
     * area. Nothing from another line can occupy it.
     */
    element.style.height = `${height}px`;
    element.style.minHeight = `${height}px`;
    element.style.maxHeight = `${height}px`;

    element.style.overflow = "hidden";

    /*
     * IMPORTANT:
     * Do NOT wrap vertically.
     *
     * Every sentence stays on one visual line.
     * The font is reduced instead if necessary.
     */
    element.style.whiteSpace = "normal";
    element.style.wordBreak = "normal";
    element.style.overflowWrap = "break-word";
    element.style.wordWrap = "normal";

    element.style.boxSizing = "border-box";

    /*
     * ---------------------------------------------------------
     * FONT
     * ---------------------------------------------------------
     */

    const originalFontSize = Math.max(
        8,
        getSafeNumber(line.font_size, 14)
    );

    /*
     * Keep the font comfortably inside the line height.
     */
    let fontSize = Math.min(
        originalFontSize,
        Math.max(8, height * 0.75),
        18
    );

    element.style.fontSize = `${fontSize}px`;

    element.style.lineHeight =
        `${Math.min(height, fontSize * 1.05)}px`;

    const fontFamily =
        safeFontFamily(line.font_family);

    element.style.fontFamily =
        `"${fontFamily}", "Noto Sans Devanagari", sans-serif`;

    element.style.fontWeight =
        line.bold ? "700" : "400";

    element.style.fontStyle =
        line.italic ? "italic" : "normal";


    /*
     * ---------------------------------------------------------
     * TEXT
     * ---------------------------------------------------------
     */

    if (
        line.segments &&
        Array.isArray(line.segments) &&
        line.segments.length
    ) {

        line.segments.forEach(segment => {

            const span =
                document.createElement("span");

            span.textContent =
                segment.text || "";

            const segmentFont =
                safeFontFamily(
                    segment.font_family
                );

            span.style.fontFamily =
                `"${segmentFont}", "Noto Sans Devanagari", sans-serif`;

            const segmentSize =
                Math.max(
                    8,
                    getSafeNumber(
                        segment.font_size,
                        fontSize
                    )
                );

            span.style.fontSize =
                `${Math.min(segmentSize, fontSize)}px`;

            span.style.fontWeight =
                segment.bold ? "700" : "400";

            span.style.fontStyle =
                segment.italic ? "italic" : "normal";

            span.style.whiteSpace =
                "nowrap";

            span.style.wordBreak =
                "normal";

            span.style.overflowWrap =
                "normal";

            element.appendChild(span);
        });

    } else {

        element.textContent =
            line.text || "";
    }


    /*
     * ---------------------------------------------------------
     * AUTOMATIC FONT SHRINK
     * ---------------------------------------------------------
     *
     * If the sentence is wider than its PDF line,
     * progressively reduce the font until it fits.
     *
     * This prevents one sentence from covering
     * the sentence below it.
     */

    const minimumFontSize = 7;

    let attempts = 0;

    while (
        element.scrollWidth > width &&
        fontSize > minimumFontSize &&
        attempts < 30
    ) {

        fontSize -= 0.25;

        element.style.fontSize =
            `${fontSize}px`;

        element.style.lineHeight =
            `${Math.min(
                height,
                fontSize * 1.05
            )}px`;

        /*
         * Keep child spans synchronized.
         */
        element
            .querySelectorAll("span")
            .forEach(span => {

                const current =
                    parseFloat(
                        span.dataset.originalSize ||
                        span.style.fontSize
                    ) || fontSize;

                if (!span.dataset.originalSize) {
                    span.dataset.originalSize =
                        current;
                }

                span.style.fontSize =
                    `${Math.min(
                        current,
                        fontSize
                    )}px`;
            });

        attempts++;
    }


    /*
     * Final hard boundary.
     */
    element.style.maxWidth =
        `${Math.max(1, width)}px`;

    return element;
}

/* =========================================================
   SAFE NUMBER
   ========================================================= */

function getSafeNumber(
    value,
    fallback
) {

    const number =
        Number(value);


    if (
        !Number.isFinite(number) ||
        number <= 0
    ) {

        return fallback;
    }


    return number;
}


/* =========================================================
   SAFE FONT FAMILY
   ========================================================= */

function safeFontFamily(
    font
) {

    if (
        typeof font !== "string" ||
        !font.trim()
    ) {

        return "Kalimati";
    }


    /*
     * Remove characters that could break
     * the CSS font-family declaration.
     */

    return font
        .replace(/["']/g, "")
        .trim()
        .slice(0, 100);
}