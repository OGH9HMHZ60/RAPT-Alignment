class ParangonadaViewer {
    constructor(canvasId, title) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.title = title;
        this.data = null;
        this.gtData = null;
        this.showGT = false;
        
        // Window config
        this.scoreTimeRange = [0, 50]; // Beats
        this.perfTimeRange = [0, 20];  // Seconds
        this.pitchRange = [21, 108];   // MIDI A0 to C8
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        // Fit parent width
        const parent = this.canvas.parentElement;
        this.canvas.width = parent.clientWidth;
        this.canvas.height = 400; // Fixed height
        this.draw();
    }

    async loadData(predUrl, gtUrl) {
        try {
            const res = await fetch(predUrl);
            this.data = await res.json();
            
            const resGT = await fetch(gtUrl);
            this.gtData = await resGT.json();
            
            // Auto fit ranges based on data
            this.fitAll();
            this.draw();
        } catch (e) {
            console.error("Error loading data. If running locally via file://, use a dev server.", e);
            
            // Fallback mock text
            this.ctx.fillStyle = 'red';
            this.ctx.fillText("Failed to load JSONs. Run a local web server to view demo.", 20, 20);
        }
    }

    fitAll() {
        if (!this.data) return;
        let maxS = 0, maxP = 0;
        this.data.score_notes.forEach(n => maxS = Math.max(maxS, n.offset));
        this.data.performance_notes.forEach(n => maxP = Math.max(maxP, n.offset));
        this.scoreTimeRange = [0, Math.min(maxS, 50)]; 
        this.perfTimeRange = [0, Math.min(maxP, 30)];
    }

    setRegion(scoreStart, scoreEnd, perfStart, perfEnd) {
        this.scoreTimeRange = [scoreStart, scoreEnd];
        this.perfTimeRange = [perfStart, perfEnd];
        this.draw();
    }

    toggleGT(show) {
        this.showGT = show;
        this.draw();
    }

    mapScoreX(t) {
        return ((t - this.scoreTimeRange[0]) / (this.scoreTimeRange[1] - this.scoreTimeRange[0])) * this.canvas.width;
    }
    
    mapPerfX(t) {
        return ((t - this.perfTimeRange[0]) / (this.perfTimeRange[1] - this.perfTimeRange[0])) * this.canvas.width;
    }
    
    mapY(p, isScore) {
        const pNorm = 1 - ((p - this.pitchRange[0]) / (this.pitchRange[1] - this.pitchRange[0]));
        const halfH = this.canvas.height / 2;
        // Padding top and bottom inside the half
        const top = isScore ? 10 : halfH + 10;
        const bottom = isScore ? halfH - 10 : this.canvas.height - 10;
        return top + pNorm * (bottom - top);
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!this.data) return;

        // Draw split line
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.canvas.height / 2);
        this.ctx.lineTo(this.canvas.width, this.canvas.height / 2);
        this.ctx.strokeStyle = '#dee2e6';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Title
        this.ctx.fillStyle = '#343a40';
        this.ctx.font = "bold 14px system-ui";
        this.ctx.fillText(this.title, 10, 20);

        // Labels
        this.ctx.font = "12px system-ui";
        this.ctx.fillStyle = '#6c757d';
        this.ctx.fillText("Score Notes (Beats)", 10, 40);
        this.ctx.fillText("Performance Notes (Seconds)", 10, this.canvas.height / 2 + 20);

        // Precompute note positions
        const sPos = {};
        this.data.score_notes.forEach(n => {
            const x = this.mapScoreX(n.onset);
            const w = Math.max(2, this.mapScoreX(n.offset) - x);
            const y = this.mapY(n.pitch, true);
            sPos[n.id] = {x, y, w, inView: x > -w && x < this.canvas.width};
            
            if(sPos[n.id].inView) {
                this.ctx.fillStyle = '#0d6efd';
                this.ctx.fillRect(x, y-2, w, 4);
            }
        });

        const pPos = {};
        this.data.performance_notes.forEach(n => {
            const x = this.mapPerfX(n.onset);
            const w = Math.max(2, this.mapPerfX(n.offset) - x);
            const y = this.mapY(n.pitch, false);
            pPos[n.id] = {x, y, w, inView: x > -w && x < this.canvas.width};
            
            if(pPos[n.id].inView) {
                this.ctx.fillStyle = '#198754';
                this.ctx.fillRect(x, y-2, w, 4);
            }
        });

        // Draw Ground Truth Alignments (Thick light green)
        if (this.showGT && this.gtData) {
            this.gtData.alignment.forEach(aln => {
                if (aln.label === 'match' && sPos[aln.score_id] && pPos[aln.performance_id]) {
                    const s = sPos[aln.score_id];
                    const p = pPos[aln.performance_id];
                    if (s.inView || p.inView) {
                        this.ctx.beginPath();
                        this.ctx.moveTo(s.x, s.y);
                        this.ctx.lineTo(p.x, p.y);
                        this.ctx.strokeStyle = 'rgba(25, 135, 84, 0.4)';
                        this.ctx.lineWidth = 4;
                        this.ctx.stroke();
                    }
                }
            });
        }

        // Draw Prediction Alignments
        this.data.alignment.forEach(aln => {
            if (aln.label === 'match' && sPos[aln.score_id] && pPos[aln.performance_id]) {
                const s = sPos[aln.score_id];
                const p = pPos[aln.performance_id];
                if (s.inView || p.inView) {
                    this.ctx.beginPath();
                    this.ctx.moveTo(s.x, s.y);
                    this.ctx.lineTo(p.x, p.y);
                    this.ctx.strokeStyle = 'rgba(220, 53, 69, 0.6)'; // Red for prediction
                    this.ctx.lineWidth = 1;
                    this.ctx.stroke();
                }
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const viewerNakamura = new ParangonadaViewer('canvas-nakamura', 'Nakamura (SOTA baseline)');
    const viewerRAPT = new ParangonadaViewer('canvas-rapt', 'RAPT (Ours)');

    viewerNakamura.loadData('static/demo_data/chopin_op38_p11_nakamura.json', 'static/demo_data/chopin_op38_p11_ground_truth.json');
    viewerRAPT.loadData('static/demo_data/chopin_op38_p11_rapt.json', 'static/demo_data/chopin_op38_p11_ground_truth.json');

    document.getElementById('btn-gt').addEventListener('change', (e) => {
        const show = e.target.checked;
        viewerNakamura.toggleGT(show);
        viewerRAPT.toggleGT(show);
    });

    document.getElementById('btn-region1').addEventListener('click', () => {
        // Zoom to opening (Beats 0-16, Seconds 0-8 approx)
        viewerNakamura.setRegion(0, 16, 0, 8);
        viewerRAPT.setRegion(0, 16, 0, 8);
    });

    document.getElementById('btn-region2').addEventListener('click', () => {
        // Zoom to final cadence (Beats 250-270, Seconds 110-140 approx)
        viewerNakamura.setRegion(250, 272, 115, 140);
        viewerRAPT.setRegion(250, 272, 115, 140);
    });
    
    document.getElementById('btn-reset').addEventListener('click', () => {
        viewerNakamura.fitAll(); viewerNakamura.draw();
        viewerRAPT.fitAll(); viewerRAPT.draw();
    });
});
