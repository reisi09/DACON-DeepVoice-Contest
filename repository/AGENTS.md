
## VESSL workspace operations

- Reuse an existing workspace; do not create a new workspace unless the user explicitly changes this instruction.
- The latest B full-data run uses `wsp-9683432r72m7`. Preserve it in the stopped state after verified downloads.
- Use CLI/API only; do not manipulate the user's screen.
- Complete storage checks and uploads before starting GPU compute. Monitor every 5 minutes, stay quiet on unchanged state, and stop compute after verified result downloads.
- Keep each run's original absolute deadline. Upload the next run's configuration/bootstrap before restarting a reused workspace so an old experiment is not accidentally relaunched.
