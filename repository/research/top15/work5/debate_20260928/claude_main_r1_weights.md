# Claude (main) round-1 check of the loss-weight finding (VERIFIED, bank CSVs + full_bank_batches.loss_weights semantics)
Voice-head cells = (label, source, generator) over voice-present rows with a known VOICE_FAKE; every cell of a class gets
equal total weight, so a source's share of its class weight = its number of cells / the class's cells.
| training pool | FAKE voice class | REAL voice class |
|---|---|---|
| original train | 358 cells: speech 98.6% / singing 1.4%; mlaad 57.3%, asvspoof2021_df 30.7%, asvspoof2019_la 5.3%, asvspoof2021_la 3.6%, speechfake 1.1% | 17 cells: speech 52.9% / singing 47.1% (each source ~5.9%) |
| P1 (train+held-out) with rule L | 385 cells: speech 92.7% / singing 7.3%; mlaad 54.0%, asvspoof2021_df 28.6% | 20 cells: speech 50% / singing 50% |
("singing" = sources other than the speech corpora and constructed speech mixes.) Consequence (INFERRED): inside the voice heads
the fake class is ~93-99% TTS/VC speech while the real class is ~50% singing -> a "singing voice => real" shortcut is
encouraged; AI-sung vocals on the LB would be under-detected (VOICE and FILE). Not yet measured: final optimizer weights after
duration balancing and cache filters; the music-head cells.
