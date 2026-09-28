# Executed notebook aggregation errata

The immutable executed notebook and GPU archive remain unchanged. Its final aggregation cell produced reporting metadata that is superseded by `final_aggregation.corrected.json`.

- Global provenance components are **256**, computed with `eight_fixed_io.connected_components` over `group_id` and `parent_ids`; the sealed summary's 178 omitted singleton groups.
- E1/E3/E4/E6/E8 use the verified role-specific population metadata in `postrun_e2/fixed_experiment_population_denominators.json`. E2 and E5 retain their four-component constructed/probe slices; E7 retains its ten speaker components.
- E8 has `fixed_updates: null` and `fit_kind: non_iterative_closed_form_real_only_gaussian`.
- The archived `progress.json` marker ending at E5 is stale. Durable reports for E1-E8 are complete.
- No score, prediction, checkpoint, fitted model, or immutable archive member was changed. No GPU rerun occurred.

The CPU post-run E2 Music/File supplement remains separately labelled and verified under `postrun_e2/`; it was not part of the GPU archive.
