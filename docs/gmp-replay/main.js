document.addEventListener('DOMContentLoaded', function() {

    // --- Configuration ---
    const CONFIG = {
        AUTO_PAN_TIMEOUT_MS: 5000,
        PLAYBACK_INTERVAL_MS: 700,
        FIT_ANIMATION_MS: 500,
        FLASH_DURATION_MS: 500,
        NODE_UPDATE_TRANSITION_MS: 300,
        EDGE_UPDATE_TRANSITION_MS: 200,
        PULSE_DURATION_MS: 250,
        LAYOUT_MAX_SIMULATION_TIME_MS: 4000,
        LAYOUT_EDGE_LENGTH: 120,
        LAYOUT_PADDING: 50,
    };

    // --- Application State ---
    const state = {
        cy: null,
        replayData: null,
        initialState: null, // To avoid re-parsing JSON
        eventsByGen: {},
        nodeHistories: {},
        currentGeneration: 0,
        maxGeneration: 0,
        isPlaying: false,
        animationInterval: null,
        selectedNodeId: null,
        lastUserInteractionTime: 0,
        tippyInstance: null,
        currentLayout: null,
    };

    // --- UI Element References ---
    const ui = {
        filePicker: document.getElementById('file-picker'),
        playPauseBtn: document.getElementById('play-pause-btn'),
        prevGenBtn: document.getElementById('prev-gen-btn'),
        nextGenBtn: document.getElementById('next-gen-btn'),
        resetViewBtn: document.getElementById('reset-view-btn'),
        timelineSlider: document.getElementById('timeline-slider'),
        generationDisplay: document.getElementById('generation-display'),
        nodeCountDisplay: document.getElementById('node-count-display'),
        selectedNodeInfo: document.getElementById('selected-node-info'),
        selectedNodeTitle: document.getElementById('selected-node-title'),
    };

    // --- Initialization ---

    /**
     * Main entry point for the application.
     */
    function initialize() {
        bindUIEventListeners();
        initTippy();

        // --- Auto-load default replay data ---
        const defaultReplayUrl = 'https://raw.githubusercontent.com/karlo-babic/graph-meme-pool/refs/heads/main/examples/simulation_replay.json'; 
        loadDataFromUrl(defaultReplayUrl);
    }

    function bindUIEventListeners() {
        ui.filePicker.addEventListener('change', handleFileSelect);
        document.getElementById('load-url-btn').addEventListener('click', () => {
            const urlInput = document.getElementById('url-input');
            if (urlInput.value) {
                loadDataFromUrl(urlInput.value);
            }
        });
        ui.playPauseBtn.addEventListener('click', togglePlayPause);
        ui.prevGenBtn.addEventListener('click', () => {
            jumpToGeneration(Math.max(0, state.currentGeneration - 1));
        });
        ui.nextGenBtn.addEventListener('click', () => {
            stepToGeneration(state.currentGeneration + 1);
        });
        ui.resetViewBtn.addEventListener('click', () => {
            if (state.cy) {
                state.cy.animate({
                    fit: { padding: CONFIG.LAYOUT_PADDING },
                    duration: CONFIG.FIT_ANIMATION_MS
                });
                state.lastUserInteractionTime = Date.now();
            }
        });
        ui.timelineSlider.addEventListener('input', (e) => {
            jumpToGeneration(parseInt(e.target.value, 10));
        });
    }

    function initTippy() {
        state.tippyInstance = tippy(document.body, {
            content: 'Loading...',
            trigger: 'manual',
            allowHTML: true,
            theme: 'gmp',
            placement: 'top',
            arrow: true,
        });
    }

    function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);
                loadAndInitializeSimulation(data);
            } catch (error) {
                console.error("Error parsing replay file:", error);
                alert('Error parsing JSON file. Please check the file format and console for details.');
            }
        };
        reader.readAsText(file);
    }

    /**
     * Fetches replay data from a URL and initializes the simulation.
     * @param {string} url - The URL of the JSON replay file.
     */
    async function loadDataFromUrl(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            loadAndInitializeSimulation(data);
        } catch (error) {
            console.error("Failed to load replay data from URL:", error);
            alert(`Failed to load replay data from ${url}. Please check the URL and the browser console for more details.`);
        }
    }

    /**
     * Loads simulation data, processes it, and initializes the visualization.
     * @param {object} data - The parsed replay data from the JSON file.
     */
    function loadAndInitializeSimulation(data) {
        stopPlayback(); // Stop any ongoing simulation

        state.replayData = data;
        state.initialState = data.initialState; // Cache initial state

        // Group events by generation for efficient lookup
        state.eventsByGen = {};
        data.events.forEach(event => {
            if (!state.eventsByGen[event.gen]) {
                state.eventsByGen[event.gen] = [];
            }
            state.eventsByGen[event.gen].push(event);
        });

        state.maxGeneration = Object.keys(state.eventsByGen).reduce((max, gen) => Math.max(max, parseInt(gen, 10)), 0);

        if (state.cy) {
            state.cy.destroy();
        }
        initCytoscape();

        ui.timelineSlider.max = state.maxGeneration;
        ui.timelineSlider.disabled = false;

        jumpToGeneration(0);
        startPlayback(); // Auto-play on load
    }

    // --- Cytoscape Setup ---

    function initCytoscape() {
        state.cy = cytoscape({
            container: document.getElementById('cy'),
            style: [
                {
                    selector: 'node',
                    style: {
                        'label': 'data(id)', 'font-size': '10px', 'color': '#fff',
                        'text-outline-color': '#000', 'text-outline-width': 2,
                        'background-color': 'mapData(score, 0, 1, #2b83ba, #d7191c)',
                        'width': 'mapData(outgoingWeight, 0, 5, 20, 80)',
                        'height': 'mapData(outgoingWeight, 0, 5, 20, 80)',
                        'border-width': 0,
                        'transition-property': 'background-color, width, height, border-width',
                        'transition-duration': `${CONFIG.NODE_UPDATE_TRANSITION_MS}ms`
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'curve-style': 'bezier', 'target-arrow-shape': 'triangle',
                        'line-color': 'mapData(weight, 0.0, 1.0, #e0e0e0, #333333)',
                        'target-arrow-color': 'mapData(weight, 0.0, 1.0, #e0e0e0, #333333)',
                        'width': 'mapData(weight, 0.0, 1.0, 0.5, 4)',
                        'transition-property': 'line-color, target-arrow-color',
                        'transition-duration': `${CONFIG.EDGE_UPDATE_TRANSITION_MS}ms`
                    }
                },
                { selector: '.selected', style: { 'border-width': '5px', 'border-color': 'gold' } },
                { selector: '.flash-border', style: { 'border-width': '8px', 'border-color': '#ffaa00' } },
            ]
        });

        bindCytoscapeEventListeners();
    }

    function bindCytoscapeEventListeners() {
        const { cy, tippyInstance } = state;

        cy.on('mouseover', 'node', (e) => {
            const node = e.target;
            const content = `<strong>ID:</strong> ${node.data('id')}<br>
                             <strong>Score:</strong> ${node.data('score') ? node.data('score').toFixed(3) : 'N/A'}<br>
                             <strong>Meme:</strong> ${node.data('meme')}`;
            tippyInstance.setProps({ getReferenceClientRect: node.popperRef().getBoundingClientRect });
            tippyInstance.setContent(content);
            tippyInstance.show();
        });

        cy.on('mouseout', 'node', () => tippyInstance.hide());
        cy.on('pan zoom', () => { state.lastUserInteractionTime = Date.now(); });

        cy.on('click', 'node', (e) => {
            state.selectedNodeId = e.target.id();
            reapplySelectionStyle();
            updateSelectedNodePanel();
        });

        cy.on('click', (e) => {
            if (e.target === cy) { // Deselect if background is clicked
                state.selectedNodeId = null;
                reapplySelectionStyle();
                updateSelectedNodePanel();
            }
        });
    }

    // --- Replay and State Management ---

    function togglePlayPause() {
        if (!state.replayData) return;

        if (state.currentGeneration >= state.maxGeneration) {
            jumpToGeneration(0);
            if (state.isPlaying) stopPlayback();
            return;
        }

        if (state.isPlaying) {
            stopPlayback();
        } else {
            startPlayback();
        }
    }

    function startPlayback() {
        if (state.isPlaying) return;
        state.isPlaying = true;
        state.animationInterval = setInterval(() => {
            const nextGen = state.currentGeneration + 1;
            if (nextGen <= state.maxGeneration) {
                stepToGeneration(nextGen);
            } else {
                stopPlayback();
            }
        }, CONFIG.PLAYBACK_INTERVAL_MS);
        updatePlayPauseButtonState();
    }

    function stopPlayback() {
        if (!state.isPlaying) return;
        state.isPlaying = false;
        clearInterval(state.animationInterval);
        state.animationInterval = null;
        updatePlayPauseButtonState();
    }

    /**
     * Rebuilds the graph state to a specific generation. Can be slow for high generation counts.
     * @param {number} targetGen - The generation to jump to.
     */
    function jumpToGeneration(targetGen) {
        if (!state.replayData) return;
        stopPlayback();

        state.currentGeneration = Math.max(0, Math.min(targetGen, state.maxGeneration));

        if (state.currentLayout) state.currentLayout.stop();

        const nodes = state.initialState.nodes.map(n => ({ ...n, group: 'nodes' }));
        const edges = state.initialState.edges.map(e => ({ ...e, group: 'edges' }));

        state.cy.elements().remove();
        state.cy.add([...nodes, ...edges]);

        state.nodeHistories = {};
        nodes.forEach(n => {
            state.nodeHistories[n.data.id] = [{ gen: 0, meme: n.data.meme }];
        });

        for (let gen = 1; gen <= state.currentGeneration; gen++) {
            (state.eventsByGen[gen] || []).forEach(event => applyEvent(event, false));
        }

        calculateOutgoingWeights();
        reapplySelectionStyle();
        updateUI();
        runLayout();
    }

    /**
     * Advances the simulation by one generation, applying events with animations.
     * @param {number} nextGen - The generation to step to.
     */
    function stepToGeneration(nextGen) {
        if (!state.replayData || nextGen < 0 || nextGen > state.maxGeneration) {
            stopPlayback();
            return;
        }

        let topologyChanged = false;
        const eventsForGen = state.eventsByGen[nextGen] || [];

        if (state.currentLayout) state.currentLayout.stop();

        // Handle node additions first to provide anchor points for layout
        const nodesToAdd = eventsForGen.filter(e => e.type === 'NODE_ADD');
        if (nodesToAdd.length > 0) {
            topologyChanged = true;
            const edgesInGen = eventsForGen.filter(e => e.type === 'EDGE_ADD');
            
            nodesToAdd.forEach(event => {
                const nodeId = event.node.data.id;
                // Find neighbors based on edges in this generation to position the new node intelligently.
                const neighborIds = new Set();
                edgesInGen.forEach(edgeEvent => {
                    if (edgeEvent.edge.source === nodeId) neighborIds.add(edgeEvent.edge.target);
                    if (edgeEvent.edge.target === nodeId) neighborIds.add(edgeEvent.edge.source);
                });

                let position = { x: Math.random() * 500, y: Math.random() * 500 };
                const neighbors = state.cy.nodes().filter(n => neighborIds.has(n.id()));

                if (neighbors.nonempty()) {
                    const avgPos = neighbors.reduce((acc, n) => ({ x: acc.x + n.position('x'), y: acc.y + n.position('y') }), { x: 0, y: 0 });
                    position = {
                        x: avgPos.x / neighbors.length + (Math.random() - 0.5) * 30,
                        y: avgPos.y / neighbors.length + (Math.random() - 0.5) * 30
                    };
                }
                applyEvent({ ...event, position }, true);
            });
        }
        
        // Apply all other events for the generation
        eventsForGen.filter(e => e.type !== 'NODE_ADD').forEach(event => {
            if (applyEvent(event, true)) {
                topologyChanged = true;
            }
        });

        state.currentGeneration = nextGen;
        calculateOutgoingWeights();
        reapplySelectionStyle();
        updateUI();

        if (topologyChanged) {
            runLayout();
        }
    }

    /**
     * Applies a single simulation event to the Cytoscape graph.
     * @param {object} event - The event object from the replay data.
     * @param {boolean} withAnimation - Whether to apply animations for this event.
     * @returns {boolean} - True if the event changed the graph topology (nodes/edges).
     */
    function applyEvent(event, withAnimation) {
        switch (event.type) {
            case 'NODE_ADD':
                state.cy.add({ data: event.node.data, group: 'nodes', position: event.position });
                if (!state.nodeHistories[event.node.data.id]) state.nodeHistories[event.node.data.id] = [];
                state.nodeHistories[event.node.data.id].push({ gen: event.gen, meme: event.node.data.meme });
                return true;

            case 'NODE_REMOVE':
                state.cy.getElementById(event.nodeId).remove();
                return true;

            case 'EDGE_ADD':
                state.cy.add({ data: event.edge, group: 'edges' });
                return true;

            case 'EDGE_REMOVE':
                state.cy.edges(`[source = "${event.source}"][target = "${event.target}"]`).remove();
                return true;

            case 'MEME_UPDATE': {
                const node = state.cy.getElementById(event.nodeId);
                if (node.nonempty()) {
                    node.data({ meme: event.newMeme, score: event.newScore });
                    if (!state.nodeHistories[event.nodeId]) state.nodeHistories[event.nodeId] = [];
                    state.nodeHistories[event.nodeId].push({ gen: event.gen, meme: event.newMeme });
                    if (withAnimation) {
                        node.addClass('flash-border');
                        setTimeout(() => node.removeClass('flash-border'), CONFIG.FLASH_DURATION_MS);
                    }
                }
                return false;
            }

            case 'PROPAGATION': {
                // This event is captured but has no visual effect.
                return false;
            }

            default:
                return false;
        }
    }

    function calculateOutgoingWeights() {
        state.cy.nodes().forEach(node => {
            const weight = node.outgoers('edge').reduce((sum, edge) => sum + edge.data('weight'), 0);
            node.data('outgoingWeight', weight);
        });
    }

    function runLayout() {
        if (state.cy.nodes().length === 0) return;
        if (state.currentLayout) state.currentLayout.stop();

        const layoutOptions = {
            name: 'cola',
            animate: true,
            maxSimulationTime: CONFIG.LAYOUT_MAX_SIMULATION_TIME_MS,
            fit: false,
            padding: CONFIG.LAYOUT_PADDING,
            edgeLength: CONFIG.LAYOUT_EDGE_LENGTH,
            stop: () => {
                // After layout settles, fit the view if user hasn't interacted recently
                if (Date.now() - state.lastUserInteractionTime > CONFIG.AUTO_PAN_TIMEOUT_MS) {
                    state.cy.animate({ fit: { padding: CONFIG.LAYOUT_PADDING }, duration: CONFIG.FIT_ANIMATION_MS });
                }
            }
        };

        state.currentLayout = state.cy.layout(layoutOptions);
        state.currentLayout.run();
    }

    function reapplySelectionStyle() {
        state.cy.elements().removeClass('selected');
        if (state.selectedNodeId) {
            const selected = state.cy.getElementById(state.selectedNodeId);
            if (selected.nonempty()) {
                selected.addClass('selected');
            } else {
                state.selectedNodeId = null; // Node was removed, clear selection
            }
        }
    }

    // --- UI Update Functions ---

    function updateUI() {
        ui.generationDisplay.textContent = `${state.currentGeneration} / ${state.maxGeneration}`;
        ui.nodeCountDisplay.textContent = state.cy.nodes().length;
        ui.timelineSlider.value = state.currentGeneration;
        updatePlayPauseButtonState();
        updateSelectedNodePanel();
    }

    function updatePlayPauseButtonState() {
        if (state.currentGeneration >= state.maxGeneration) {
            ui.playPauseBtn.textContent = 'Replay';
        } else if (state.isPlaying) {
            ui.playPauseBtn.textContent = 'Pause';
        } else {
            ui.playPauseBtn.textContent = 'Play';
        }
    }

    function updateSelectedNodePanel() {
        if (!state.selectedNodeId || state.cy.getElementById(state.selectedNodeId).empty()) {
            state.selectedNodeId = null;
            ui.selectedNodeTitle.textContent = "Selected Node Info";
            ui.selectedNodeInfo.innerHTML = '<p>Click a node to see its details.</p>';
            return;
        }

        const node = state.cy.getElementById(state.selectedNodeId);
        const data = node.data();
        ui.selectedNodeTitle.textContent = `Node ${data.id} Details`;

        const history = (state.nodeHistories[data.id] || []).filter(entry => entry.gen <= state.currentGeneration);
        let historyHTML = '';
        if (history.length > 0) {
            historyHTML += '<strong>Meme History:</strong><div id="selected-node-history">';
            history.slice().reverse().forEach(entry => {
                historyHTML += `<p><strong>Gen ${entry.gen}:</strong> ${entry.meme}</p>`;
            });
            historyHTML += '</div>';
        }

        ui.selectedNodeInfo.innerHTML = `
            <p><strong>ID:</strong> ${data.id}</p>
            <p><strong>Group:</strong> ${data.group || 'N/A'}</p>
            <p><strong>Score:</strong> ${typeof data.score === 'number' ? data.score.toFixed(3) : 'N/A'}</p>
            ${historyHTML}
        `;
    }

    // --- Entry Point ---
    initialize();
});