## links containing tennis dataset download

https://nycu1-my.sharepoint.com/personal/tik_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Ftik%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FOpenDataset%2FTrackNet%5FTennis%2FDataset%2Ezip&parent=%2Fpersonal%2Ftik%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FOpenDataset%2FTrackNet%5FTennis&ga=1

https://drive.google.com/drive/folders/11r0RUaQHX7I3ANkaYG4jOxXK1OYo01Ut



ToDO 

2. Pre-known bboxes from our
YOLO+PlayerTracker — biggest possible
win, also a quality boost
- The detect/segment/track stage is
~50-60% of runtime (3 of 4 min on the
2-sec test).
- It runs Detectron2-X101 + SAM2 on
every frame to find people generically,
then tracks. We already have
YOLO+PlayerTracker tuned for our exact
tennis videos (used by MediaPipe and
VideoPose3D pipelines) — and it's more
reliable for our footage (100% bbox
coverage vs PromptHMR's
spectator-included 8 tracks).
- Feeding our bboxes directly skips
Detectron2 + SAM2 + tracking entirely.
- Requires modifying pipeline.py to
accept pre-computed tracks (~30-50 LOC
change).
- Risk: PromptHMR may use SAM2 masks
downstream (need to verify); if so,
would still need SAM2 to fill mask
field.