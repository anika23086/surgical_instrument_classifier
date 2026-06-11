// Surgical Vision AI - Client Side Logic

document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------
    // State Variables
    // ----------------------------------------------------
    let catalogItems = [];
    let activeCategory = "all";
    let searchQuery = "";

    // ----------------------------------------------------
    // DOM Elements
    // ----------------------------------------------------
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzoneContent = document.getElementById("dropzoneContent");
    const dropzonePreview = document.getElementById("dropzonePreview");
    const previewImg = document.getElementById("previewImg");
    const removeBtn = document.getElementById("removeBtn");
    
    // Result panels
    const stateAwaiting = document.getElementById("stateAwaiting");
    const stateLoading = document.getElementById("stateLoading");
    const resultsActive = document.getElementById("resultsActive");
    
    // Result hero elements
    const queryHeroImg = document.getElementById("queryHeroImg");
    const catalogHeroImg = document.getElementById("catalogHeroImg");
    const heroMatchPct = document.getElementById("heroMatchPct");
    const heroMatchName = document.getElementById("heroMatchName");
    const heroMatchSku = document.getElementById("heroMatchSku");
    const heroMatchSize = document.getElementById("heroMatchSize");
    const heroMatchPage = document.getElementById("heroMatchPage");
    const matchesList = document.getElementById("matchesList");

    // Catalog elements
    const catalogGrid = document.getElementById("catalogGrid");
    const catalogSearch = document.getElementById("catalogSearch");
    const categoryTabs = document.getElementById("categoryTabs");

    // Dialog elements
    const detailsDialog = document.getElementById("detailsDialog");
    const dialogImg = document.getElementById("dialogImg");
    const dialogCategory = document.getElementById("dialogCategory");
    const dialogSku = document.getElementById("dialogSku");
    const dialogName = document.getElementById("dialogName");
    const dialogSizes = document.getElementById("dialogSizes");
    const dialogPage = document.getElementById("dialogPage");
    const dialogGridId = document.getElementById("dialogGridId");
    const dialogCloseBtn = document.getElementById("dialogCloseBtn");
    const dialogSearchBtn = document.getElementById("dialogSearchBtn");
    let dialogActiveImgPath = "";

    // ----------------------------------------------------
    // Header Navigation Scroll Logic
    // ----------------------------------------------------
    const navButtons = document.querySelectorAll(".nav-btn");
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.dataset.target;
            const element = document.getElementById(targetId);
            if (element) {
                const headerOffset = 90; // Offset for the sticky header
                const elementPosition = element.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;
                
                window.scrollTo({
                    top: offsetPosition,
                    behavior: "smooth"
                });
            }
        });
    });

    // Active Section Tracking (Scroll-Spy)
    const sections = [
        { id: "scannerSection", btnId: "navBtnScanner" },
        { id: "ingestionSection", btnId: "navBtnIngestion" },
        { id: "databaseSection", btnId: "navBtnDatabase" }
    ];

    window.addEventListener("scroll", () => {
        let currentSection = "";
        const scrollPosition = window.scrollY + 180;

        sections.forEach(sec => {
            const el = document.getElementById(sec.id);
            if (el) {
                const top = el.offsetTop;
                const height = el.offsetHeight;
                if (scrollPosition >= top && scrollPosition < top + height) {
                    currentSection = sec.btnId;
                }
            }
        });

        // Default to database section when scrolled to the very bottom of the page
        if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 50) {
            currentSection = "navBtnDatabase";
        }

        sections.forEach(sec => {
            const btn = document.getElementById(sec.btnId);
            if (btn) {
                if (sec.btnId === currentSection) {
                    btn.classList.add("active");
                } else {
                    btn.classList.remove("active");
                }
            }
        });
    });

    // Helper to map messy raw PDF category strings to clean standard names
    function getCleanCategory(cat) {
        if (!cat) return "Other";
        const catLower = String(cat).toLowerCase().trim();
        if (catLower.includes("forceps")) return "Forceps";
        if (catLower.includes("scissors")) return "Scissors";
        if (catLower.includes("retractor")) return "Retractors";
        if (catLower.includes("clamp")) return "Clamps";
        if (catLower.includes("holloware")) return "Holloware";
        if (catLower.includes("scale")) return "Scales";
        if (catLower.includes("autoclave") || catLower.includes("sterilizer")) return "Autoclaves";
        if (catLower.includes("furniture")) return "Furniture";
        if (catLower.includes("rubber")) return "Rubber Products";
        if (catLower.includes("ophthalmic")) return "Ophthalmic";
        return "Other";
    }

    // ----------------------------------------------------
    // Catalog Management
    // ----------------------------------------------------
    async function loadCatalog() {
        try {
            const res = await fetch("/api/catalog");
            catalogItems = await res.json();
            
            // Clean up some catalog data programmatically
            catalogItems.forEach(item => {
                // Map messy raw categories to clean names
                item.category = getCleanCategory(item.category);
                // Standardize size display
                if (!item.size) item.size = "Standard Bx Size";
            });

            // Dynamically update the index cache indicator text
            const statusText = document.querySelector(".status-text");
            if (statusText) {
                const uniqueClasses = new Set(catalogItems.map(item => item.id)).size;
                statusText.textContent = `Index Cached: ${uniqueClasses} Instruments`;
            }

            renderCatalogGrid();
            setupCategoryFilterTabs();
        } catch (err) {
            console.error("Failed to load catalog database:", err);
            catalogGrid.innerHTML = `
                <div class="error-msg" style="grid-column: 1/-1; text-align: center; color: hsl(0, 80%, 60%); padding: 2rem;">
                    <i data-lucide="alert-octagon" style="width: 40px; height: 40px; margin-bottom: 0.5rem;"></i>
                    <h3>Error Loading Catalog Database</h3>
                    <p>${err.message}</p>
                </div>
            `;
            lucide.createIcons();
        }
    }

    function renderCatalogGrid() {
        catalogGrid.innerHTML = "";
        const cleanQuery = searchQuery.toLowerCase().trim();
        
        // Search-First Empty State: If category is "all" and search query is empty, show prompt
        if (activeCategory === "all" && !cleanQuery) {
            catalogGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 4rem 2rem;">
                    <i data-lucide="search" style="width: 48px; height: 48px; margin-bottom: 1rem; color: var(--accent-cyan); opacity: 0.6;"></i>
                    <h3 style="font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1.15rem; color: var(--text-primary); margin-bottom: 0.5rem;">Explore Catalog Database</h3>
                    <p style="max-width: 400px; margin: 0 auto; font-size: 0.85rem; line-height: 1.4;">Type in the search bar or click any category tab above to browse the instrument database.</p>
                </div>
            `;
            lucide.createIcons();
            return;
        }
        
        const filteredItems = catalogItems.filter(item => {
            const matchCategory = activeCategory === "all" || item.category === activeCategory;
            const matchSearch = !cleanQuery || 
                (item.name && item.name.toLowerCase().includes(cleanQuery)) ||
                (item.sku && String(item.sku).toLowerCase().includes(cleanQuery)) ||
                (item.category && item.category.toLowerCase().includes(cleanQuery));
                
            return matchCategory && matchSearch;
        });

        if (filteredItems.length === 0) {
            catalogGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 3rem;">
                    <i data-lucide="info" style="width: 32px; height: 32px; margin-bottom: 0.5rem; color: var(--accent-cyan);"></i>
                    <p>No matching surgical instruments found in database.</p>
                </div>
            `;
            lucide.createIcons();
            return;
        }

        filteredItems.forEach(item => {
            const card = document.createElement("div");
            card.className = "catalog-card";
            
            const imagePath = `/${item.image_path}`;
            
            card.innerHTML = `
                <div class="catalog-img-box">
                    <img src="${imagePath}" alt="${item.name}" loading="lazy">
                </div>
                <div class="catalog-card-info">
                    <span class="catalog-card-tag">${item.category}</span>
                    <h3 class="catalog-card-title">${item.name}</h3>
                    <span class="catalog-card-sku">SKU: ${item.sku}</span>
                </div>
            `;
            
            card.addEventListener("click", () => showDetailModal(item));
            catalogGrid.appendChild(card);
        });
        
        lucide.createIcons();
    }

    function setupCategoryFilterTabs() {
        // Collect all unique categories from the catalog
        const categories = new Set();
        catalogItems.forEach(item => {
            if (item.category) categories.add(item.category);
        });

        // Determine a clean alphabetical ordering with "Forceps" first if present
        const standardOrder = ["Forceps", "Scissors", "Retractors", "Clamps", "Holloware", "Scales", "Autoclaves", "Furniture", "Rubber Products", "Ophthalmic", "Other"];
        const sortedCategories = Array.from(categories).sort((a, b) => {
            let idxA = standardOrder.indexOf(a);
            let idxB = standardOrder.indexOf(b);
            if (idxA === -1) idxA = 99;
            if (idxB === -1) idxB = 99;
            return idxA - idxB;
        });

        // Clear existing static tabs beyond "All Items"
        categoryTabs.innerHTML = `<button class="tab-btn active" data-category="all">All Items</button>`;

        sortedCategories.forEach(cat => {
            const btn = document.createElement("button");
            btn.className = "tab-btn";
            btn.dataset.category = cat;
            btn.textContent = cat;
            categoryTabs.appendChild(btn);
        });

        // Add event listeners to tabs
        const tabs = categoryTabs.querySelectorAll(".tab-btn");
        tabs.forEach(tab => {
            tab.addEventListener("click", () => {
                tabs.forEach(t => t.classList.remove("active"));
                tab.classList.add("active");
                activeCategory = tab.dataset.category;
                renderCatalogGrid();
            });
        });
    }

    // ----------------------------------------------------
    // Search Filtering
    // ----------------------------------------------------
    catalogSearch.addEventListener("input", (e) => {
        searchQuery = e.target.value;
        renderCatalogGrid();
    });

    // ----------------------------------------------------
    // Detailed View Dialog Model
    // ----------------------------------------------------
    function showDetailModal(item) {
        const imagePath = `/${item.image_path}`;
        dialogActiveImgPath = imagePath;

        dialogImg.src = imagePath;
        dialogCategory.textContent = item.category;
        dialogSku.textContent = `SKU: ${item.sku}`;
        dialogName.textContent = item.name;
        dialogSizes.textContent = item.size || "Standard Size";
        dialogPage.textContent = `Page ${item.page}`;
        dialogGridId.textContent = item.id;
        
        const dialogDescription = document.getElementById("dialogDescription");
        if (dialogDescription) {
            dialogDescription.textContent = item.description || "This surgical instrument is precision-crafted from high-grade medical steel, designed to meet rigorous clinical standards for hospital surgical workflows.";
        }
        
        // Dynamically render catalogs
        const dialogCatalogs = document.getElementById("dialogCatalogs");
        if (dialogCatalogs) {
            let catalogsList = [];
            try {
                catalogsList = typeof item.catalogs === 'string' ? JSON.parse(item.catalogs) : item.catalogs;
            } catch (e) {
                catalogsList = [];
            }
            if (!Array.isArray(catalogsList)) {
                catalogsList = [];
            }
            if (catalogsList.length === 0) {
                catalogsList = [{
                    "catalog": item.category === "Medical Rubber Products" ? "Medical Rubber Products Catalog" : (item.category === "Ophthalmic Instruments" ? "Ophthalmic Instruments Catalog" : (item.category === "Hospital Furniture" ? "Hospital Furniture Catalog" : (item.category === "Hospital Holloware" ? "Hospital Holloware Catalog" : "Surgical Instruments Catalog"))),
                    "page": item.page
                }];
            }
            dialogCatalogs.innerHTML = catalogsList.map(c => `
                <span class="catalog-badge">
                    <i data-lucide="book-open"></i>
                    ${c.catalog} (Page ${c.page})
                </span>
            `).join('');
        }
        
        // Show dialog using standard top-layer modal API
        detailsDialog.showModal();
        lucide.createIcons();
    }

    // Close Dialog events
    dialogCloseBtn.addEventListener("click", () => {
        detailsDialog.close();
    });

    // Click outside dialog to close
    detailsDialog.addEventListener("click", (e) => {
        const rect = detailsDialog.getBoundingClientRect();
        const isInDialog = (
            rect.top <= e.clientY && 
            e.clientY <= rect.top + rect.height &&
            rect.left <= e.clientX && 
            e.clientX <= rect.left + rect.width
        );
        if (!isInDialog) {
            detailsDialog.close();
        }
    });

    // Query similar item from modal
    dialogSearchBtn.addEventListener("click", async () => {
        detailsDialog.close();
        if (dialogActiveImgPath) {
            await performClassificationFromUrl(dialogActiveImgPath);
        }
    });

    // ----------------------------------------------------
    // Classification Engine (Upload / Drag & Drop)
    // ----------------------------------------------------
    
    // Drag and drop event styling
    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        }, false);
    });

    // Handle dropped files
    dropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Bind file browse button
    dropzone.querySelector(".select-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("Error: Please select a valid surgical instrument image file.");
            return;
        }

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropzoneContent.classList.add("hidden");
            dropzonePreview.classList.remove("hidden");
            
            // Launch scanner
            performClassification(file);
        };
        reader.readAsDataURL(file);
    }

    // Remove preview image
    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.value = "";
        previewImg.src = "";
        dropzonePreview.classList.add("hidden");
        dropzoneContent.classList.remove("hidden");
        dropzonePreview.classList.remove("scanning");
        
        // Reset results panels to default
        stateAwaiting.classList.remove("hidden");
        stateLoading.classList.add("hidden");
        resultsActive.classList.add("hidden");
    });

    // ----------------------------------------------------
    // Run Search / Embeddings Similarity Matches
    // ----------------------------------------------------
    async function performClassification(file) {
        // Toggle loader panels
        stateAwaiting.classList.add("hidden");
        stateLoading.classList.remove("hidden");
        resultsActive.classList.add("hidden");
        
        // Start scanner animation
        dropzonePreview.classList.add("scanning");

        const formData = new FormData();
        formData.append("image", file);

        try {
            const res = await fetch("/api/classify", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || "Internal model matching error.");
            }

            const data = await res.json();
            displayResults(data.results, previewImg.src);
        } catch (err) {
            console.error("Match error:", err);
            stateLoading.classList.add("hidden");
            stateAwaiting.classList.remove("hidden");
            alert(`Classification Error: ${err.message}`);
        } finally {
            // Stop scanning laser
            dropzonePreview.classList.remove("scanning");
        }
    }

    // Direct classification from a URL (e.g. sample clicks or detail modal queries)
    async function performClassificationFromUrl(url) {
        // Display preview image from URL
        previewImg.src = url;
        dropzoneContent.classList.add("hidden");
        dropzonePreview.classList.remove("hidden");
        
        // Toggle loader panels
        stateAwaiting.classList.add("hidden");
        stateLoading.classList.remove("hidden");
        resultsActive.classList.add("hidden");
        dropzonePreview.classList.add("scanning");

        try {
            const response = await fetch(url);
            const blob = await response.blob();
            const filename = url.split("/").pop();
            const file = new File([blob], filename, { type: "image/png" });
            
            const formData = new FormData();
            formData.append("image", file);

            const res = await fetch("/api/classify", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || "Internal model matching error.");
            }

            const data = await res.json();
            displayResults(data.results, url);
        } catch (err) {
            console.error("Match error:", err);
            stateLoading.classList.add("hidden");
            stateAwaiting.classList.remove("hidden");
            alert(`Classification Error: ${err.message}`);
        } finally {
            dropzonePreview.classList.remove("scanning");
        }
    }

    function displayResults(results, queryImgSrc) {
        if (!results || results.length === 0) {
            stateLoading.classList.add("hidden");
            stateAwaiting.classList.remove("hidden");
            alert("No instruments found matching this criteria.");
            return;
        }

        // Hide loader, show active state
        stateLoading.classList.add("hidden");
        resultsActive.classList.remove("hidden");

        // 1. Render Top Match (Rank 1 Hero Match)
        const topMatch = results[0];
        const similarityPct = (topMatch.similarity * 100).toFixed(1);

        queryHeroImg.src = queryImgSrc;
        catalogHeroImg.src = `/${topMatch.image_path}`;
        heroMatchPct.textContent = `${similarityPct}% MATCH`;
        heroMatchPct.className = `badge ${topMatch.similarity > 0.85 ? 'badge-success' : ''}`;
        heroMatchName.textContent = topMatch.name;
        heroMatchSku.textContent = `SKU: ${topMatch.sku}`;
        heroMatchSize.textContent = `Sizes: ${topMatch.size || 'Standard Size'}`;
        heroMatchPage.textContent = `Catalog Page: ${topMatch.page}`;

        // Render dynamic parent catalogs for hero match
        const heroMatchCatalogs = document.getElementById("heroMatchCatalogs");
        if (heroMatchCatalogs) {
            let catalogsList = [];
            try {
                catalogsList = typeof topMatch.catalogs === 'string' ? JSON.parse(topMatch.catalogs) : topMatch.catalogs;
            } catch (e) {
                catalogsList = [];
            }
            if (!Array.isArray(catalogsList)) {
                catalogsList = [];
            }
            if (catalogsList.length === 0) {
                catalogsList = [{
                    "catalog": topMatch.category === "Medical Rubber Products" ? "Medical Rubber Products Catalog" : (topMatch.category === "Ophthalmic Instruments" ? "Ophthalmic Instruments Catalog" : "Surgical Instruments Catalog"),
                    "page": topMatch.page
                }];
            }
            heroMatchCatalogs.innerHTML = catalogsList.map(c => `
                <span class="catalog-badge">
                    <i data-lucide="book-open"></i>
                    ${c.catalog} (Page ${c.page})
                </span>
            `).join('');
        }

        // 2. Render subsequent matches in detailed list
        matchesList.innerHTML = "";
        
        // Loop starting from rank 2 (index 1) to end (index 4)
        for (let i = 1; i < results.length; i++) {
            const match = results[i];
            const simVal = (match.similarity * 100).toFixed(1);
            const itemRow = document.createElement("div");
            itemRow.className = "match-list-item";
            
            itemRow.innerHTML = `
                <div class="match-thumb">
                    <img src="/${match.image_path}" alt="${match.name}">
                </div>
                <div class="match-info">
                    <span class="match-title">${match.name}</span>
                    <span class="match-sku">SKU: ${match.sku} | Page: ${match.page}</span>
                </div>
                <div class="match-meter-wrapper">
                    <span class="match-similarity-val">${simVal}%</span>
                    <div class="match-meter-bar-bg">
                        <div class="match-meter-bar-fill" style="width: 0%"></div>
                    </div>
                </div>
            `;
            
            // Add click detailed modal to results cards as well!
            itemRow.addEventListener("click", () => showDetailModal(match));
            matchesList.appendChild(itemRow);
            
            // Trigger beautiful progressive bar animation on next frame
            setTimeout(() => {
                const fill = itemRow.querySelector(".match-meter-bar-fill");
                if (fill) fill.style.width = `${simVal}%`;
            }, 100);
        }

        lucide.createIcons();
    }

    // ----------------------------------------------------
    // Sample Selector Cards
    // ----------------------------------------------------
    const sampleCards = document.querySelectorAll(".sample-card");
    sampleCards.forEach(card => {
        card.addEventListener("click", async () => {
            const url = card.dataset.img;
            if (url) {
                await performClassificationFromUrl(url);
            }
        });
    });

    // ----------------------------------------------------
    // Catalog Ingestion Pipeline Logic
    // ----------------------------------------------------
    let activeJobId = null;
    let reviewItems = [];
    let currentPdfFilename = "";
    let pollingInterval = null;

    // DOM elements for Ingestion
    const pdfDropzone = document.getElementById("pdfDropzone");
    const pdfFileInput = document.getElementById("pdfFileInput");
    const settingsBtn = document.getElementById("settingsBtn");
    const apiKeySetup = document.getElementById("apiKeySetup");
    const apiKeyInput = document.getElementById("apiKeyInput");
    const saveKeyBtn = document.getElementById("saveKeyBtn");
    
    // Ingestion Lightbox Elements
    const imagePreviewDialog = document.getElementById("imagePreviewDialog");
    const previewDialogImg = document.getElementById("previewDialogImg");
    const previewDialogMeta = document.getElementById("previewDialogMeta");
    const previewDialogCloseBtn = document.getElementById("previewDialogCloseBtn");
    
    const pipelineProgress = document.getElementById("pipelineProgress");
    const pipelineStatusText = document.getElementById("pipelineStatusText");
    const reviewSection = document.getElementById("reviewSection");
    const reviewGrid = document.getElementById("reviewGrid");
    const reviewBadgeText = document.getElementById("reviewBadgeText");
    const actionBarCount = document.getElementById("actionBarCount");
    
    const addItemLink = document.getElementById("addItemLink");
    const addItemForm = document.getElementById("addItemForm");
    const saveAddedItemBtn = document.getElementById("saveAddedItemBtn");
    const btnCancelIngestion = document.getElementById("btnCancelIngestion");
    const btnApproveIngestion = document.getElementById("btnApproveIngestion");
    
    const trainingProgressSection = document.getElementById("trainingProgressSection");
    const trainingProgressFill = document.getElementById("trainingProgressFill");
    const trainingEpochText = document.getElementById("trainingEpochText");
    
    const pipelineComplete = document.getElementById("pipelineComplete");
    const pipelineCompleteMsg = document.getElementById("pipelineCompleteMsg");
    const statNewCount = document.getElementById("statNewCount");
    const statElapsed = document.getElementById("statElapsed");
    const btnDismissPipeline = document.getElementById("btnDismissPipeline");
    
    const pipelineError = document.getElementById("pipelineError");
    const pipelineErrorMsg = document.getElementById("pipelineErrorMsg");
    const btnDismissError = document.getElementById("btnDismissError");

    // 1. Check API Key configuration status
    async function checkApiKeyStatus() {
        try {
            const res = await fetch("/api/settings");
            const data = await res.json();
            if (data.has_api_key) {
                apiKeySetup.classList.add("hidden");
                pdfDropzone.classList.remove("hidden");
            } else {
                apiKeySetup.classList.remove("hidden");
                pdfDropzone.classList.add("hidden");
            }
        } catch (err) {
            console.error("Error checking API key status:", err);
        }
    }

    // 2. Settings button toggle key form
    settingsBtn.addEventListener("click", () => {
        apiKeySetup.classList.toggle("hidden");
        if (apiKeySetup.classList.contains("hidden")) {
            // Check if key is configured to decide if we can show dropzone
            fetch("/api/settings")
                .then(res => res.json())
                .then(data => {
                    if (data.has_api_key) {
                        pdfDropzone.classList.remove("hidden");
                    }
                });
        } else {
            pdfDropzone.classList.add("hidden");
        }
    });

    // 3. Save key button event
    saveKeyBtn.addEventListener("click", async () => {
        const key = apiKeyInput.value.trim();
        if (!key) {
            alert("Please enter a valid Groq API key.");
            return;
        }

        saveKeyBtn.disabled = true;
        saveKeyBtn.textContent = "Saving...";

        try {
            const res = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ groq_api_key: key })
            });

            const data = await res.json();
            if (res.ok && data.success) {
                apiKeyInput.value = "";
                alert("Groq API key saved successfully!");
                await checkApiKeyStatus();
            } else {
                alert("Error saving API key: " + (data.error || "Unknown error"));
            }
        } catch (err) {
            console.error("Error saving settings:", err);
            alert("Failed to save settings: " + err.message);
        } finally {
            saveKeyBtn.disabled = false;
            saveKeyBtn.textContent = "Save Key";
        }
    });

    // 4. PDF Drag and Drop / Browse
    ["dragenter", "dragover"].forEach(eventName => {
        pdfDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            pdfDropzone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        pdfDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            pdfDropzone.classList.remove("dragover");
        }, false);
    });

    pdfDropzone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0 && files[0].type === "application/pdf") {
            uploadCatalogPDF(files[0]);
        } else {
            alert("Please drop a valid PDF catalog file.");
        }
    });

    pdfDropzone.querySelector(".select-btn").addEventListener("click", (e) => {
        e.stopPropagation();
        pdfFileInput.click();
    });

    pdfFileInput.addEventListener("change", (e) => {
        if (pdfFileInput.files.length > 0) {
            uploadCatalogPDF(pdfFileInput.files[0]);
        }
    });

    // 5. Upload PDF function
    async function uploadCatalogPDF(file) {
        // Hide upload panel elements, show stepper progress
        pdfDropzone.classList.add("hidden");
        settingsBtn.style.display = "none";
        apiKeySetup.classList.add("hidden");
        pipelineProgress.classList.remove("hidden");
        
        // Reset stepper visual state
        resetStepper();
        updateStepper(1); // Step 1 Active
        pipelineStatusText.textContent = "Uploading PDF catalog...";

        const formData = new FormData();
        formData.append("catalog", file);

        try {
            const res = await fetch("/api/upload-catalog", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || "Internal error uploading PDF catalog.");
            }

            const data = await res.json();
            activeJobId = data.job_id;
            
            // Mark Upload stage complete and start AI processing (Step 2)
            setStepCompleted(1);
            updateStepper(2); // Step 2 active
            pipelineStatusText.textContent = "PDF uploaded successfully. Starting AI extraction...";

            // Start polling
            startPolling(activeJobId);

        } catch (err) {
            console.error("PDF Upload failed:", err);
            showPipelineError(err.message);
        }
    }

    // 6. Polling & Ingestion Lifecycle
    function startPolling(jobId) {
        if (pollingInterval) clearInterval(pollingInterval);
        
        pollingInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/pipeline-status/${jobId}`);
                if (!res.ok) throw new Error("Failed to retrieve pipeline status.");
                
                const data = await res.json();
                
                if (data.error) {
                    clearInterval(pollingInterval);
                    showPipelineError(data.error);
                    return;
                }

                // Update text progress description
                pipelineStatusText.innerHTML = `${data.message} <span class="status-highlight">(${data.progress_pct}%)</span>`;

                const stage = data.stage;
                if (stage === "extracting" || stage === "queued") {
                    updateStepper(2);
                } else if (stage === "review") {
                    clearInterval(pollingInterval);
                    setStepCompleted(2);
                    updateStepper(3);
                    pipelineStatusText.textContent = "AI Ingestion completed. Awaiting review...";
                    // Fetch review items
                    loadReviewItems(jobId);
                } else if (stage === "deduplicating" || stage === "processing" || stage === "updating" || stage === "training") {
                    // Switch to training screen
                    reviewSection.classList.add("hidden");
                    trainingProgressSection.classList.remove("hidden");
                    setStepCompleted(3);
                    updateStepper(4);
                    
                    // Parse training details
                    if (stage === "training") {
                        trainingProgressFill.style.width = `${data.progress_pct}%`;
                        trainingEpochText.innerHTML = data.message;
                    } else {
                        trainingProgressFill.style.width = `95%`;
                        trainingEpochText.textContent = data.message;
                    }
                } else if (stage === "reloading" || stage === "complete") {
                    clearInterval(pollingInterval);
                    setStepCompleted(4);
                    showPipelineComplete(data.stats);
                }

            } catch (err) {
                console.error("Status polling error:", err);
            }
        }, 2000);
    }

    // 7. Load review items from API
    async function loadReviewItems(jobId) {
        try {
            const res = await fetch(`/api/pipeline-preview/${jobId}`);
            if (!res.ok) throw new Error("Failed to load catalog preview.");
            
            const data = await res.json();
            reviewItems = data.items;
            currentPdfFilename = data.pdf_filename || "";
            
            // Set initial PDF viewer source to the first page (or page 1)
            const firstPage = (reviewItems.length > 0) ? (reviewItems[0].page || 1) : 1;
            const pdfViewer = document.getElementById("reviewPdfViewer");
            if (pdfViewer && currentPdfFilename) {
                pdfViewer.src = `/uploads/${currentPdfFilename}#page=${firstPage}`;
            }
            
            reviewSection.classList.remove("hidden");
            renderReviewCards();
        } catch (err) {
            console.error("Error loading preview items:", err);
            showPipelineError(err.message);
        }
    }

    // 8. Render review grid
    function renderReviewCards() {
        reviewGrid.innerHTML = "";
        
        if (reviewItems.length === 0) {
            reviewGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 2rem;">
                    <i data-lucide="info" style="width: 32px; height: 32px; margin-bottom: 0.5rem; color: var(--accent-cyan);"></i>
                    <p>No instruments extracted. All items deleted or skipped.</p>
                </div>
            `;
            reviewBadgeText.textContent = "0 items";
            actionBarCount.textContent = "0 items ready";
            btnApproveIngestion.disabled = true;
            lucide.createIcons();
            return;
        }

        // Sort items: amber border (low confidence or missing Name/SKU) float to the top
        const sortedItems = [...reviewItems].sort((a, b) => {
            const aNeedsAttention = a.confidence === 'low' || !a.name || !a.sku;
            const bNeedsAttention = b.confidence === 'low' || !b.name || !b.sku;
            if (aNeedsAttention && !bNeedsAttention) return -1;
            if (!aNeedsAttention && bNeedsAttention) return 1;
            return 0;
        });

        sortedItems.forEach((item, index) => {
            const isConfidenceLow = item.confidence === 'low' || !item.name || !item.sku;

            const card = document.createElement("div");
            card.className = `review-card ${isConfidenceLow ? 'needs-attention' : 'confident'}`;
            
            // Image source — base64 if it's new/AI-extracted, or server image path if raw file path
            let imgSrc = "";
            if (item.image_base64) {
                imgSrc = `data:image/png;base64,${item.image_base64}`;
            } else if (item.raw_image_path) {
                imgSrc = `/${item.raw_image_path}`;
            }

            card.innerHTML = `
                <div class="review-card-thumb clickable-thumb" title="Click to view larger image">
                    <img src="${imgSrc}" alt="${item.name || 'Extracted Instrument'}">
                </div>
                <div class="review-card-content">
                    <input type="text" class="editable-field field-name" value="${item.name || ''}" placeholder="Name (Required)" data-index="${index}" data-field="name">
                    <input type="text" class="editable-field field-sku" value="${item.sku || ''}" placeholder="SKU (Required)" data-index="${index}" data-field="sku">
                    <input type="text" class="editable-field field-category" value="${item.category || ''}" placeholder="Category" data-index="${index}" data-field="category">
                </div>
                <button class="review-card-delete" data-index="${index}" title="Remove item">
                    <i data-lucide="trash-2"></i>
                </button>
            `;

            // Bind thumbnail click to open lightbox modal
            card.querySelector(".review-card-thumb").addEventListener("click", () => {
                openImagePreviewModal(imgSrc, item.name || "Extracted Instrument", item.page || 0, index + 1);
            });

            // Bind inline edit events to update global array on change
            const inputs = card.querySelectorAll(".editable-field");
            inputs.forEach(input => {
                input.addEventListener("input", (e) => {
                    const idx = parseInt(e.target.dataset.index);
                    const field = e.target.dataset.field;
                    const val = e.target.value;
                    
                    // Update the sorted item
                    sortedItems[idx][field] = val;
                    
                    // Synchronize back to the main reviewItems array
                    const originalRawPath = sortedItems[idx].raw_image_path;
                    const mainItem = reviewItems.find(itm => itm.raw_image_path === originalRawPath);
                    if (mainItem) {
                        mainItem[field] = val;
                    }

                    // Dynamically update card border if they fill out required fields (name and sku)
                    const hasRequired = sortedItems[idx].name && sortedItems[idx].name.trim() !== "" &&
                                        sortedItems[idx].sku && sortedItems[idx].sku.trim() !== "";
                    if (hasRequired) {
                        card.className = 'review-card confident';
                    } else {
                        card.className = 'review-card needs-attention';
                    }
                    
                    // Re-calculate how many need attention
                    updateAttentionStats(sortedItems);
                });
            });

            // Bind delete button
            card.querySelector(".review-card-delete").addEventListener("click", (e) => {
                e.stopPropagation();
                card.classList.add("deleting");
                
                setTimeout(() => {
                    const idx = parseInt(card.querySelector(".review-card-delete").dataset.index);
                    const deletedRawPath = sortedItems[idx].raw_image_path;
                    
                    // Remove from global reviewItems
                    reviewItems = reviewItems.filter(itm => itm.raw_image_path !== deletedRawPath);
                    
                    renderReviewCards();
                }, 300);
            });

            // Bind card click & input focus to auto-scroll the PDF viewer
            const focusHandler = () => {
                const pageNum = item.page || 1;
                const pdfViewer = document.getElementById("reviewPdfViewer");
                if (pdfViewer && currentPdfFilename) {
                    const currentSrc = pdfViewer.src;
                    const newSrc = `/uploads/${currentPdfFilename}#page=${pageNum}`;
                    if (!currentSrc.endsWith(`#page=${pageNum}`)) {
                        pdfViewer.src = newSrc;
                    }
                }
            };
            
            card.addEventListener("click", focusHandler);
            card.querySelectorAll("input").forEach(inp => {
                inp.addEventListener("focus", focusHandler);
            });

            reviewGrid.appendChild(card);
        });

        // Set action counts
        reviewBadgeText.textContent = `${reviewItems.length} items detected`;
        updateAttentionStats(sortedItems);
        btnApproveIngestion.disabled = false;
        
        lucide.createIcons();
    }

    function updateAttentionStats(items) {
        let attentionCount = 0;
        items.forEach(item => {
            if (!item.name || !item.name.trim() || !item.sku || !item.sku.trim()) {
                attentionCount++;
            }
        });

        const badge = document.getElementById("reviewBadge");
        if (attentionCount > 0) {
            badge.className = "review-badge attention";
            badge.innerHTML = `<i data-lucide="alert-circle"></i> ${attentionCount} need attention`;
        } else {
            badge.className = "review-badge ready";
            badge.innerHTML = `<i data-lucide="check-circle-2"></i> All items verified`;
        }

        actionBarCount.textContent = `${reviewItems.length} items ready, ${attentionCount} need review`;
        lucide.createIcons();
    }

    // 9. Show Add Missing Item Form
    addItemLink.addEventListener("click", () => {
        addItemForm.classList.toggle("hidden");
    });

    saveAddedItemBtn.addEventListener("click", async () => {
        const nameInput = document.getElementById("addName");
        const skuInput = document.getElementById("addSku");
        const catSelect = document.getElementById("addCategory");
        const imgInput = document.getElementById("addImg");

        const name = nameInput.value.trim();
        const sku = skuInput.value.trim();
        const category = catSelect.value;
        const file = imgInput.files[0];

        if (!name) {
            alert("Instrument Name is required.");
            return;
        }
        if (!sku) {
            alert("Instrument SKU is required.");
            return;
        }
        if (!file) {
            alert("Please select an image file for this instrument.");
            return;
        }

        saveAddedItemBtn.disabled = true;
        saveAddedItemBtn.textContent = "Uploading Image...";

        try {
            // First, upload the image to /api/upload-image
            const formData = new FormData();
            formData.append("image", file);

            const uploadRes = await fetch("/api/upload-image", {
                method: "POST",
                body: formData
            });

            if (!uploadRes.ok) throw new Error("Failed to upload custom instrument image.");
            const uploadData = await uploadRes.json();

            // Next, read image locally for local thumbnail preview
            const reader = new FileReader();
            reader.onload = (e) => {
                const newItem = {
                    name: name,
                    sku: sku,
                    category: category,
                    description: "Manually added instrument during catalog ingestion review.",
                    size: "Standard Size",
                    page: 1,
                    confidence: "high",
                    image_base64: e.target.result.split(",")[1],
                    raw_image_path: uploadData.raw_image_path
                };

                // Add to global items
                reviewItems.push(newItem);
                
                // Clear fields
                nameInput.value = "";
                skuInput.value = "";
                imgInput.value = "";
                
                // Close form and re-render review screen
                addItemForm.classList.add("hidden");
                renderReviewCards();
            };
            reader.readAsDataURL(file);

        } catch (err) {
            console.error("Failed to add missing item:", err);
            alert("Error adding item: " + err.message);
        } finally {
            saveAddedItemBtn.disabled = false;
            saveAddedItemBtn.textContent = "Add";
        }
    });

    // 10. Ingest Approve Event
    btnApproveIngestion.addEventListener("click", async () => {
        // Validate that all items have Name and SKU
        const invalid = reviewItems.some(itm => !itm.name.trim() || !itm.sku.trim());
        if (invalid) {
            alert("Please verify all items have a valid Name and SKU before approving.");
            return;
        }

        btnApproveIngestion.disabled = true;
        btnApproveIngestion.textContent = "Submitting Approval...";

        try {
            const res = await fetch(`/api/pipeline-approve/${activeJobId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ items: reviewItems })
            });

            if (!res.ok) throw new Error("Failed to submit approved catalog items.");

            // Transition UI stepper to stage 4 (Train Model)
            reviewSection.classList.add("hidden");
            trainingProgressSection.classList.remove("hidden");
            setStepCompleted(3);
            updateStepper(4);
            
            // Resume polling
            startPolling(activeJobId);

        } catch (err) {
            console.error("Approval submission failed:", err);
            showPipelineError(err.message);
        } finally {
            btnApproveIngestion.disabled = false;
            btnApproveIngestion.innerHTML = `<i data-lucide="play"></i> Approve & Train`;
            lucide.createIcons();
        }
    });

    // 11. Cancel Ingestion
    btnCancelIngestion.addEventListener("click", () => {
        if (confirm("Are you sure you want to cancel catalog ingestion? All extracted items will be discarded.")) {
            resetPipelineUI();
        }
    });

    // Dismiss Complete
    btnDismissPipeline.addEventListener("click", () => {
        resetPipelineUI();
    });

    // Dismiss Error
    btnDismissError.addEventListener("click", () => {
        resetPipelineUI();
    });

    // 12. Helper UI Ingestion methods
    function resetPipelineUI() {
        if (pollingInterval) {
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
        activeJobId = null;
        reviewItems = [];
        
        pipelineProgress.classList.add("hidden");
        reviewSection.classList.add("hidden");
        trainingProgressSection.classList.add("hidden");
        pipelineComplete.classList.add("hidden");
        pipelineError.classList.add("hidden");
        addItemForm.classList.add("hidden");
        
        pdfDropzone.classList.remove("hidden");
        settingsBtn.style.display = "flex";
        pdfFileInput.value = "";
        
        loadCatalog();
    }

    function showPipelineError(msg) {
        if (pollingInterval) clearInterval(pollingInterval);
        pipelineProgress.classList.add("hidden");
        reviewSection.classList.add("hidden");
        trainingProgressSection.classList.add("hidden");
        pipelineComplete.classList.add("hidden");
        
        pipelineError.classList.remove("hidden");
        pipelineErrorMsg.textContent = msg;
    }

    function showPipelineComplete(stats) {
        pipelineProgress.classList.add("hidden");
        reviewSection.classList.add("hidden");
        trainingProgressSection.classList.add("hidden");
        pipelineError.classList.add("hidden");
        
        pipelineComplete.classList.remove("hidden");
        
        if (stats) {
            statNewCount.textContent = stats.new_items || 0;
            statElapsed.textContent = "Saved";
            pipelineCompleteMsg.textContent = `Successfully completed! Added ${stats.new_items || 0} instruments (${stats.duplicates || 0} duplicates skipped). Database size is now ${stats.total_database || 0} items.`;
        }
    }

    // Stepper Styling Helpers
    function resetStepper() {
        for (let i = 1; i <= 4; i++) {
            const step = document.getElementById(`step-${i}`);
            step.classList.remove("active", "completed");
            step.querySelector(".step-dot").innerHTML = i;
            if (i < 4) {
                const connector = document.getElementById(`connector-${i}`);
                connector.classList.remove("active", "completed");
            }
        }
    }

    function updateStepper(stepNum) {
        const step = document.getElementById(`step-${stepNum}`);
        if (step) {
            step.classList.add("active");
            step.classList.remove("completed");
        }
        
        if (stepNum > 1) {
            const conn = document.getElementById(`connector-${stepNum - 1}`);
            if (conn) {
                conn.classList.add("active");
                conn.classList.remove("completed");
            }
        }
    }

    function setStepCompleted(stepNum) {
        const step = document.getElementById(`step-${stepNum}`);
        if (step) {
            step.classList.add("completed");
            step.classList.remove("active");
            step.querySelector(".step-dot").innerHTML = `<i data-lucide="check" style="width:16px;height:16px;"></i>`;
        }
        
        const conn = document.getElementById(`connector-${stepNum}`);
        if (conn) {
            conn.classList.add("completed");
            conn.classList.remove("active");
        }
        lucide.createIcons();
    }

    // ----------------------------------------------------
    // Ingestion Lightbox Modal Controls
    // ----------------------------------------------------
    function openImagePreviewModal(imgSrc, name, page, itemNumber) {
        if (!imagePreviewDialog || !previewDialogImg || !previewDialogMeta) return;
        
        previewDialogImg.src = imgSrc;
        let metaText = `Item ${itemNumber}`;
        if (name && name !== "Extracted Instrument") {
            metaText += ` - ${name}`;
        }
        if (page) {
            metaText += ` (Page ${page})`;
        }
        previewDialogMeta.textContent = metaText;
        imagePreviewDialog.showModal();
        lucide.createIcons();
    }

    if (previewDialogCloseBtn && imagePreviewDialog) {
        previewDialogCloseBtn.addEventListener("click", () => {
            imagePreviewDialog.close();
        });
    }

    if (imagePreviewDialog) {
        // Click outside dialog to close
        imagePreviewDialog.addEventListener("click", (e) => {
            const rect = imagePreviewDialog.getBoundingClientRect();
            const isInDialog = (
                rect.top <= e.clientY && 
                e.clientY <= rect.top + rect.height &&
                rect.left <= e.clientX && 
                e.clientX <= rect.left + rect.width
            );
            if (!isInDialog) {
                imagePreviewDialog.close();
            }
        });
    }

    // Perform check on load
    checkApiKeyStatus();

    // ----------------------------------------------------
    // Initialize Dashboard
    // ----------------------------------------------------
    loadCatalog();
});
