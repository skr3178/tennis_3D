┌──────┬─────────┬──────────┬───────┬───────────────────────────────────┐    
│ Axis │   a     │    b     │  r²   │              Verdict              │   
│      │ (slope) │ (offset) │       │                                   │    
├──────┼─────────┼──────────┼───────┼───────────────────────────────────┤    
│ x    │ 0.931   │ +0.013 m │ 0.961 │ ✅ near-perfect lateral           │  
├──────┼─────────┼──────────┼───────┼───────────────────────────────────┤    
│ y    │ 0.047   │ +1.222 m │ 0.002 │ ❌ height is essentially constant │
│      │         │          │       │  ~1.2 m, not predicted            │    
├──────┼─────────┼──────────┼───────┼───────────────────────────────────┤  
│ z    │ 0.667   │ -1.740 m │ 0.487 │ ❌ depth has 33% compression +    │    
│      │         │          │       │ 1.7 m bias                        │
└──────┴─────────┴──────────┴───────┴───────────────────────────────────┘    
                                                        
For comparison, the paper's pred_unrefined: every axis at a≈1.000, b≈0,      
r²≈0.999.                                                                  
                                                        
Diagnosis: the height-prediction LSTM is essentially not working on real data
— outputting near-constant y around 1.22 m regardless of input. Depth is  
partially broken (scaled by 0.67 + 1.7 m offset). Lateral x is fine because  
that's largely fixed by the camera ray geometry, not by the LSTM.            
            
Cause #1 (convention) and #2 (distribution) ruled out. Cause #3 (incomplete  
training) plus a likely #4 — Unity → real generalization gap in the height 
LSTM are the prime suspects. The paper notes (line 451 of the PDF) that    
"training on synthetic then fine-tuning on real achieves the best NRMSE" —   
exactly this gap, addressed by paper authors via fine-tuning.
                                                        
Three productive next steps, in order of cost:            
                                                        
1. Test old checkpoint (checkpoints_5k_v2/best.pt, ep 550, trained on the  
un-filtered set) on the same clip — 30 sec. If it also fails this hard, our  
pipeline is fine and the issue is the training-from-Unity-only paradigm. If
it fails less, our filter introduced a regression.                           
2. Resume training to ep 1400 — ~1.5 h. Probably moves x and z to ≈ paper
levels but won't close the y gap unless the height LSTM was just slow to     
converge.                                                                  
3. Fine-tune on real-clip data (paper's recommendation) — would need a small
real dataset with ground-truth 3D, of which tennis_real.json provides 21     
clips. Highest expected gain.      

Cause #1 ruled out. OpenGL vs OpenCV camera produced essentially identical
predictions (max diff 0.07 m << 4.5 m gap). Convention conversion is correct.

2. Training distribution z-range mismatch — Unity training data has z roughly
±30 m; this clip's gt z stays in [3, 11] m. The model may not have learned
the right normalization for depth.

Where our model fails (clearly):
- ✅ Camera is correct (gt 3D → 2D reprojects at 1.1 px mean — sub-pixel
agreement with TrackNet's labels)
- ✅ Lateral x: std 0.41 m (vs paper 0.006 m) — small absolute, ~70× ratio
- ✅ Height y: std 0.96 m (vs paper 0.021 m) — small mean, ~50× ratio
- ❌ Depth z: std 5.69 m (vs paper 0.20 m) — 28× worse — this dominates


LSTM_Model/inference_output/eval_game1_clip1_50fps/diagnostic.png      
                                                                            
What it shows (after linearly interpolating the 25 fps 2D track up to 50 fps 
to match the model's training rate):
So the fps fix moved the failure mode from "height-LSTM totally inert" to    
"height-LSTM working but noisy and over-amplified." That's progress, not
closure — the residual ~4 m gap is the Unity → real physics-distribution gap 
(no air drag, idealized bounces in Unity).                                    