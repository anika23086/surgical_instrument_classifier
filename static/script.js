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
    // Catalog Management
    // ----------------------------------------------------
    async function loadCatalog() {
        try {
            const res = await fetch("/api/catalog");
            catalogItems = await res.json();
            
            // Clean up some catalog data programmatically
            catalogItems.forEach(item => {
                // If category is null or empty, assign a default
                if (!item.category) item.category = "General Surgical";
                // Standardize size display
                if (!item.size) item.size = "Standard Bx Size";
            });

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
        
        const filteredItems = catalogItems.filter(item => {
            const matchCategory = activeCategory === "all" || item.category === activeCategory;
            const cleanQuery = searchQuery.toLowerCase().trim();
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
            
            // Using Flask custom routing to fetch the cropped catalog images
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

        // Clear existing static tabs beyond "All Items"
        categoryTabs.innerHTML = `<button class="tab-btn active" data-category="all">All Items</button>`;

        categories.forEach(cat => {
            // Shorten name if needed for UI aesthetics
            let displayName = cat;
            if (cat === "Surgical Forceps") displayName = "Forceps";
            else if (cat === "Surgical Scissors") displayName = "Scissors";
            else if (cat === "Surgical Retractors") displayName = "Retractors";
            
            const btn = document.createElement("button");
            btn.className = "tab-btn";
            btn.dataset.category = cat;
            btn.textContent = displayName;
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
        
        // Show dialog using standard top-layer modal API
        detailsDialog.showModal();
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
    // Initialize Dashboard
    // ----------------------------------------------------
    loadCatalog();
});
