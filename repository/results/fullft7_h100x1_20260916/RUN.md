# H100 × 1 live run

User authorized environment creation and immediate experiment execution on 2026-09-16.

- Job: job-gxeb388rndlt
- Console: https://cloud.vessl.ai/qwdkw/Default/jobs/job-gxeb388rndlt
- State confirmed: running; dependency installation complete; notebook reached cell 10 (audio extraction).
- Hardware: H100 SXM 80GB × 1, 15 vCPU, 200 GB RAM, 200 GB temporary storage.
- Price returned by VESSL: 2.98 credits/hour. Balance before launch: see billing_before.json.
- Run ID: run-20260916-044239
- Output prefix: dacon-v6-training/fullft7/h100x1-20260916/runs/run-20260916-044239/
- Notebook default flags remain False; run_job.py explicitly enables input staging, checks, preprocessing, and seven sequential trials for this job.
- Training stops at creation epoch + 15.5 hours, reserving 30 minutes for artifact copying. The batch command terminates within creation epoch + 16 hours, including queue/setup time. Maximum 16 GPU-hours at this rate is 47.68 credits; storage costs are separate.
- run_job.py mirrors complete files to Object Storage while running and on normal/error exit. Source partial checkpoints and session locks are excluded. Previously persisted run output is restored by start_job.sh if the same run ID is intentionally resumed.
- No epoch result or score is claimed in this launch record. HTDemucs preprocessing and model training follow extraction automatically.

Local launcher checks validated real notebook dataclass compilation, runtime flag overrides, finished-file copying, exclusion of partial checkpoints, and subprocess shutdown. No full fine-tuning accuracy or runtime result is implied.
