## Experiments

### OWT
#### General
- I ran the sid with CE, sid with elbo corrected, and ancestral with elbo [array 0-2].
All first with log-linear schedule, then with cosine-squared schedule [array 0-1].
All runs with 32, 64, 128 and 256 steps. 
- Log-linear with sid collapses. I have not tried with ancestral, but it should be worse.

#### Greedy Effect
- Ancestral: improves slightly PPL but degrades entropy significantly. 
- Sid: No significant effect. 

#### Codes for the runs
  The current task list:                                                                                                        │samples.json (untracked)                                                                 
  - 0: duo_sid_sidelbo                                                                                                          │──────────────────────────────────────────────────────────────────────────────────────── 
  - 1: ce_denoiser_sid                                                                                                          │New file not yet staged.                                                                 
  - 2: duo_ancestral                                                                                                            │Run `git add :/samples.json` to see line counts.                                         
  - 3: ce_denoiser_restart                                                                                                      │                                                                                         
  - 4: ce_denoiser_ancestral (the control)   

### TODO Next.
- Select a schedule for sid and ablate temperature.
- Depending on the results, run experiments with partial sid, ancestral_sid, and ancestral_restart.
  - Ancestral restart: the best sampler with the CE model. At lam 0.8 it lowers gen ppl by about 17% over plain ancestral (68.1 → 56.4) at│    +2].                                                                                 
    almost the same entropy.
    - lam is defined in t, not α. Under cosine p0.5, lam 0.75 restarts at α ≈ 0.96, so it barely restarts at all (112.8 / 5.48).  │    +worse.                                                                              
      With non-log-linear schedules, use RESTART_ALPHA instead.                                                                   │  9 +                                                                                    
    - Not tested yet: DUO with the correction. Without it, the sid phase collapses.   
    - Not tested: Ancestral restart with ancestral sampling. 
- Run similar experiments with TinyThing, with corrected sid. 
- Re-run the sudoku experiments from scratch.
- See it elbo training works on molecules. 