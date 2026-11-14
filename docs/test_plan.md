# LeafGuard: Test Plan, Test Cases and Test Report

## 1. Scope and objectives

Show that LeafGuard does what the design says (see [hld.md](hld.md) and [lld.md](lld.md)),
at three levels:

| Level | What is under test | How it is isolated | Where |
|---|---|---|---|
| **Unit** | Pure helpers (images, drift, split, gate rules), stage tracking, config, the backend API and its error contract, the serving wrapper | Synthetic leaf images drawn in `conftest.py`. The model server and MLflow are replaced by stubs or a closed port. Torch uses a 3-parameter test network. | `tests/unit/` |
| **Integration** | validate → split → preprocess with a **real local Spark session** on a miniature dataset that contains a corrupt file, a tiny image and a duplicate | Temporary folder, no Pushgateway | `tests/integration/` |
| **End-to-end** | The running Docker Compose stack through nginx: pages, runtime config, real photo → prediction, feedback, error shape, model info, pipeline console data, Prometheus targets | None: real containers and a real promoted model | `tests/e2e/` |
| **System / lifecycle** (manual, scripted) | Pipeline reproducibility, gate rejection of a bad candidate, rollback, drift detection and alerting, monitoring in Grafana | Real stack | §6 below |

**Out of scope:** model accuracy beyond the gate floor (not assessed), load testing beyond
the demo traffic, and browser automation of the UI (the UI is checked manually, see §6).

## 2. Approach and tools

* **pytest** with markers `spark` and `e2e`, so the fast unit tests run on every
  `dvc repro` and the slower tests run on demand.
* **pytest-cov** for coverage (`reports/coverage.xml`), **pytest-html** for a readable
  report (`reports/tests.html`), and **JUnit XML** for counts (`reports/*junit.xml`).
* **CI gate:** the `unit_tests` DVC stage writes `reports/ci_gate.json` only when every
  test passes, and `validate` depends on that file. So a single failing test stops the
  pipeline before any data is processed or model trained. `lint` (ruff + black) runs
  before the tests in the same way.

| Command | Runs |
|---|---|
| `make lint` | ruff + black checks |
| `make test` | unit tests + coverage + HTML report |
| `make test-spark` | Spark integration test |
| `make test-e2e` | end-to-end tests (stack must be up) |
| `dvc repro` | lint → unit tests → whole pipeline |

## 3. Test environment

| Item | Value |
|---|---|
| Machine | Ubuntu (Linux 7.0, aarch64), 4 CPUs, 8 GB RAM, no GPU |
| Python (pipeline, tests) | 3.14.4 in `~/aiops` venv; PySpark 4.1.1 on Java 17; PyTorch 2.15 CPU; MLflow 3.17.2 |
| Containers | Docker 29.8, Compose v5.6; backend Python 3.12; model server and MLflow Python 3.14 |
| Data | `data/raw`: 1,440 images, DVC md5 `4ea2db67…` |

## 4. Acceptance criteria

The release is accepted when **all** of the following hold:

| ID | Criterion | Threshold |
|---|---|---|
| AC1 | Code style | ruff and black report no issues |
| AC2 | Unit tests | 100 % pass |
| AC3 | Spark integration test | passes |
| AC4 | End-to-end tests | 100 % pass against the running stack |
| AC5 | Pipeline | `dvc repro` completes every stage with status `success` |
| AC6 | Model quality gate | Serving model golden-set accuracy ≥ 0.70 and p95 model latency ≤ 400 ms |
| AC7 | Responsiveness | `/api/v1/predict` answers a typical photo in < 1 s through nginx |
| AC8 | Robustness | Every invalid input gets a 4xx with the standard error shape; nothing returns a stack trace |
| AC9 | Safe deployment | A candidate that fails the gate is not served; rollback restores an earlier version |
| AC10 | Monitoring | All Prometheus targets `up`; drift is detected and raises an alert |

## 5. Test cases

### 5.1 Unit tests (`tests/unit`, 55 cases)

| ID | Test | Expected |
|---|---|---|
| U01 | `test_a_normal_photo_passes_validation` | A leaf passes and the 4 drift stats are computed |
| U02–U04 | `test_bad_uploads_are_rejected_with_a_useful_code` ×3 | empty → 400 NO_FILE; garbage → 400 INVALID_IMAGE; PDF → 415 UNSUPPORTED_TYPE |
| U05 | `test_a_tiny_image_is_rejected` | 32 px → IMAGE_TOO_SMALL |
| U06 | `test_an_oversized_upload_is_rejected` | 6 MB → 413 FILE_TOO_LARGE |
| U07 | `test_every_known_class_has_plain_language_text` | All 6 classes have a name, description and advice; unknown names get a fallback |
| U08 | `test_health_is_always_available` | `/health` → 200 `{"status":"ok"}` |
| U09 | `test_ready_reports_503_when_the_model_server_is_down` | `/ready` → 503 MODEL_UNAVAILABLE |
| U10 | `test_a_confident_prediction_comes_back_in_full` | 200; status ok; top-3 sorted; advice and disclaimer present; `X-Request-ID` header |
| U11 | `test_a_low_score_is_reported_as_uncertain` | Flat scores → `uncertain`, hint set, no advice |
| U12 | `test_a_picture_that_is_not_a_leaf_is_reported_as_uncertain` | Red square → `uncertain` |
| U13 | `test_the_model_being_down_returns_503_in_the_standard_error_shape` | 503 with exactly `{code, message, request_id}` |
| U14 | `test_a_missing_file_is_a_422_in_the_standard_error_shape` | 422 INVALID_REQUEST naming `file` |
| U15 | `test_feedback_needs_a_class_when_the_prediction_was_wrong` | 422 MISSING_CLASS; 422 UNKNOWN_CLASS |
| U16 | `test_agreeing_with_a_prediction_is_accepted` | 200 `stored: true` |
| U17 | `test_the_class_list_is_published_for_the_ui` | 6 classes |
| U18 | `test_metrics_are_exposed_in_the_prometheus_format` | 4 key metric families present |
| U19 | `test_the_pipeline_endpoints_cope_with_no_reports_yet` | Empty lists, `gate: null`, DAG text |
| U20 | `test_the_pipeline_summary_shows_the_newest_run_of_each_stage` | Newest (failed) run wins; error text kept |
| U21 | `test_reports_are_read_when_they_exist` | gate.json surfaced |
| U22 | `test_the_serving_version_is_looked_up_again_after_the_cache_expires` | Registry cache respects its TTL |
| U23–U25 | `test_local_path_removes_the_spark_uri_prefix` ×3 | `file:` URIs, percent-encoding |
| U26 | `test_split_and_label_read_from_the_folder_names` | `(val, septoria)` |
| U27 | `test_decode_rejects_rubbish_and_truncated_files` | ImageError raised |
| U28 | `test_perceptual_hash_is_stable_and_tells_images_apart` | Same → equal; different → not |
| U29 | `test_blur_score_drops_when_an_image_is_blurred` | blurred < sharp |
| U30 | `test_green_ratio_separates_a_leaf_from_a_plain_red_picture` | > 0.1 vs 0.0 |
| U31 | `test_inspect_flags_each_kind_of_problem` | too_small, corrupt, clean |
| U32 | `test_resize_square_produces_an_rgb_jpeg_of_the_asked_size` | 64×64 RGB JPEG |
| U33 | `test_model_array_has_the_shape_the_network_expects` | (3,128,128) float32 |
| U34 | `test_psi_is_small_for_the_same_distribution_and_large_after_a_shift` | < 0.05 vs > 0.5 |
| U35 | `test_psi_returns_zero_when_there_is_nothing_to_compare` | 0.0 |
| U36 | `test_split_is_stratified_and_keeps_the_val_folder_as_the_test_set` | 17/3 per class; test only from `val` |
| U37 | `test_the_same_seed_always_gives_the_same_split` | Identical frames; other seed differs |
| U38 | `test_invalid_images_are_dropped_and_the_cap_is_respected` | Invalid excluded; cap 10 |
| U39 | `test_near_duplicates_stay_together_and_never_leak_into_the_test_set` | One split per phash group, never test |
| U40 | `test_asking_for_a_class_that_does_not_exist_is_an_error` | DataError |
| U41 | `test_summary_counts_every_split` | 50 = 34 + 6 + 10 |
| U42 | `test_the_project_params_file_loads_and_has_every_section` | 5 sections |
| U43 | `test_config_errors_are_clear` | ConfigError ×3 |
| U44 | `test_a_successful_stage_is_recorded` | Run log line with status success |
| U45 | `test_a_failing_stage_is_recorded_and_the_error_still_propagates` | status failed, error text, exception re-raised |
| U46 | `test_the_quality_gate_rejects_a_missing_or_tiny_class` | DataError ×2 |
| U47 | `test_a_good_first_model_is_promoted` | passed, 3 reasons |
| U48 | `test_a_model_below_the_accuracy_floor_is_refused` | refused |
| U49 | `test_a_model_worse_than_the_champion_is_refused` | refused ("dropped") |
| U50 | `test_a_small_regression_within_the_allowance_is_accepted` | passed |
| U51 | `test_a_slow_model_is_refused_even_when_accurate` | refused ("exceeds") |
| U52 | `test_the_wrapper_returns_one_probability_per_class` | columns = classes, sum = 1 |
| U53 | `test_the_wrapper_resizes_to_the_trained_size` | (1,3,32,32) |
| U54 | `test_the_model_list_is_empty_not_an_error_when_mlflow_is_down` | `/api/v1/models` → 200 `[]` |
| U55 | `test_model_versions_carry_their_aliases_and_gate_result` | Newest first; champion alias on the right version; accuracy and timestamp parsed |

### 5.2 Integration test (`tests/integration`, 1 case)

| ID | Test | Expected |
|---|---|---|
| I01 | `test_the_data_pipeline_runs_end_to_end` | 35 files read; rejects = {corrupt 1, duplicate 1, too_small 1}; split 18/6/8; 32 images resized to 64×64; baseline over 18 train images with 4 features; run log = validate, split, preprocess |

### 5.3 End-to-end tests (`tests/e2e`, 10 cases)

| ID | Test | Expected |
|---|---|---|
| E01 | `test_the_frontend_serves_all_three_pages` | 3 × 200 |
| E02 | `test_the_frontend_receives_its_runtime_config` | `config.js` defines `LEAFGUARD_API` |
| E03 | `test_the_model_server_is_up_and_the_backend_is_ready` | `/ping` 200; ready with a real version |
| E04 | `test_a_real_photo_is_diagnosed_through_the_frontend_proxy` | 200, valid class, 3 alternatives, < 2 s, request id matches header |
| E05 | `test_feedback_on_a_prediction_is_stored` | `stored: true` |
| E06 | `test_a_non_image_is_refused_with_the_standard_error_shape` | 415 UNSUPPORTED_TYPE |
| E07 | `test_the_serving_model_version_is_published` | alias champion, 6 classes |
| E08 | `test_the_pipeline_console_has_data_to_show` | All 6 data/model stages present; gate decision present |
| E09 | `test_prometheus_scrapes_every_target` | Every target `up` |
| E10 | `test_the_console_lists_registered_versions_with_the_champion` | `/api/v1/models` non-empty; exactly one champion, matching `/model/info` |

### 5.4 System and lifecycle scenarios (scripted, run manually)

| ID | Scenario | Steps | Expected |
|---|---|---|---|
| S01 | Full pipeline | `dvc repro` on a clean checkout | All 8 stages succeed; version registered and promoted |
| S02 | Reproducibility | `dvc repro` again with no changes | Every stage skipped |
| S03 | Bad candidate | `train.epochs: 1`, `gate.min_accuracy: 0.95`, `dvc repro` | Only train/evaluate/gate re-run; gate refuses; champion unchanged; reasons on the console |
| S04 | Experiments | `make experiment` with new epochs / learning rate (SETUP step 10b) | Each run is a new MLflow run and model version; promoted only if it meets the floor and is no more than 0.02 worse than the champion |
| S05 | Rollback | `make rollback TO=1` | `/model/info` and predictions report version 1 |
| S06 | Drift | `make traffic` then `make drift` | `leafguard_drift_score` > 1.0; `LeafGuardInputDrift` fires; after `make traffic` it drops back below 1.0 and the alert resolves |
| S07 | Monitoring | Open Grafana *LeafGuard overview* | Panels show live requests, latency, classes, drift, stage metrics, containers |
| S08 | UI | Upload a photo, a PDF, a 6 MB file; give feedback; resize to phone width | Result card; friendly errors without an upload; feedback thanks; layout adapts |
| S09 | Simple use | Use the Check leaf page as a first-time user: Upload photo → Check this leaf → read the label → Yes/No | Each step is one large button; the verdict is one coloured label (green Healthy, red Disease found, yellow Not sure) |
| S10 | Management console | On the Pipeline console filter run history by stage and status, open a version from *Model versions* | List narrows to the filter, with a success/failure tally; the version opens in MLflow |

## 6. Test report

Run on **14 November 2026** on the final build, from a clean state (`docker compose down -v`,
no previous MLflow runs), following [SETUP.md](../SETUP.md). Model **v4** is serving.

### 6.1 Automated tests

| Suite | Cases | Passed | Failed | Errors | Skipped | Time | Report |
|---|---|---|---|---|---|---|---|
| Lint (ruff + black) | 2 checks | 2 | 0 | 0 | 0 | < 1 s | `reports/lint.txt` |
| Unit | 55 | 55 | 0 | 0 | 0 | 3.4 s | `reports/junit.xml`, `reports/tests.html` |
| Integration (Spark) | 1 | 1 | 0 | 0 | 0 | 8.7 s | `reports/integration_junit.xml` |
| End-to-end (E01–E10) | 10 | 10 | 0 | 0 | 0 | 0.6 s | `reports/e2e_junit.xml` |
| **Total executed** | **66** | **66** | **0** | **0** | **0** | | |

Line coverage of `src/leafguard` and the backend from the unit tests alone: **58 %**
(`reports/coverage.xml`). The Spark stages, training and registry I/O are covered by the
integration test and the system scenarios rather than by unit tests.

### 6.2 System scenarios

| ID | Result | Evidence |
|---|---|---|
| S01 | **Pass**: 8/8 stages succeeded; the data and model stages took 45 s (validate 8.5 s, preprocess 7.0 s, train 19.7 s); v1 registered and promoted (accuracy 0.722) | `reports/pipeline_runs.jsonl`, `dvc.lock` |
| S02 | **Pass**: a second `dvc repro` skipped every stage | `dvc repro` / `dvc status` output |
| S03 | **Pass**: v2 (1 epoch) accuracy 0.519 refused against the 0.95 floor; champion stayed v1 | `make models`: v2 `gate=failed` |
| S04 | **Pass**: 4 MLflow runs, 4 versions. v3 (lr 0.003) 0.743 promoted over v1; v4 (6 epochs) 0.776 promoted over v3 | MLflow *Compare*, registry tags, `reports/gate.json` |
| S05 | **Pass**: rollback to v1 served v1; rolling forward to v4 served v4 (`/ready`: `"model_version":"4"`) | `/api/v1/model/info`, `/ready` |
| S06 | **Pass**: PSI peaked at 12.4 (brightness, blur), 7.5 (saturation), 6.7 (green ratio), all above 1.0; `LeafGuardInputDrift` reached *firing* | `leafguard_drift_score`, Prometheus `/alerts` |
| S07 | **Pass (data source)**: 6/6 Prometheus targets up, 122 predictions recorded and queryable; panels checked by hand | Prometheus `/targets`, Grafana dashboard |
| S08 | **Manual**: pages, `config.js`, error shapes and feedback pass through the same HTTP calls the UI makes (E01–E06); the visual and phone-width check is done by hand | E01–E06 |
| S09 | **Manual**: button flow and the three coloured labels checked by hand in the browser | E04 (the prediction it renders) |
| S10 | **Pass (API)** + manual: the console's version list carries exactly one champion (E10) and the stage data is served (E08); filters and MLflow links checked by hand | E08, E10 |

### 6.3 Acceptance criteria result

| ID | Result | Measured |
|---|---|---|
| AC1 | ✅ Met | ruff: passed, black: passed |
| AC2 | ✅ Met | 55/55 |
| AC3 | ✅ Met | 1/1 |
| AC4 | ✅ Met | 10/10 |
| AC5 | ✅ Met | 8/8 stages `success` |
| AC6 | ✅ Met | champion accuracy 0.776 ≥ 0.70; p95 latency 1.77 ms ≤ 400 ms |
| AC7 | ✅ Met | E04 round trip through the frontend 0.26 s < 2 s budget |
| AC8 | ✅ Met | U02–U06, U13–U15, E06; generic handler returns INTERNAL_ERROR without a trace |
| AC9 | ✅ Met | S03, S05 |
| AC10 | ✅ Met | 6/6 targets up (E09), S06 |

**Verdict: all acceptance criteria are met. The release is accepted.**

### 6.4 Defects found and fixed during testing

| # | Defect | Fix |
|---|---|---|
| 1 | MLflow 2.19 in the containers could not accept models from the MLflow 3.17 client | Pinned the containers to MLflow 3.17.2 on Python 3.14 (environment parity) |
| 2 | MLflow 3 rejected container Host headers; OOM-killed at 768 MB by background job consumers | `--allowed-hosts`, one worker, `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` |
| 3 | `--expose-prometheus` crashed (missing `prometheus-flask-exporter`) | Added to the image |
| 4 | Grafana panels pointed at datasource uid `prometheus`, which provisioning never set | Set the uid in provisioning; later replaced by a `${datasource}` dashboard variable so any data source works |
| 5 | Backend cached the serving version forever, so it reported the wrong version after a rollback | 30 s TTL (U22) and the stale `model_info` series cleared |
| 6 | `make models` showed no aliases on MLflow 3 | Aliases read from the registered model |
| 7 | FastAPI 422 errors used a different shape from the rest of the API | Custom handler returns `INVALID_REQUEST` (U14) |
| 8 | Help page promised a blur check that does not exist | Text corrected |
| 9 | Unit tests reached the live MLflow when `MLFLOW_TRACKING_URI` was exported (caught by the CI gate during `dvc repro`) | `conftest.py` forces stub addresses |
| 10 | Frontend container reported unhealthy (Alpine resolves `localhost` to `::1`) | Healthcheck uses `127.0.0.1` |
| 11 | Restoring the original params after a bad model created no new version: DVC's run cache restored the old outputs, so train and gate never ran | Lifecycle steps use `make experiment` with new parameter values (SETUP step 10b) |
| 12 | `make rollback` rewrote `reports/gate.json`, so the next `dvc repro` re-ran the gate and registered a duplicate version | `make rollback` runs `dvc commit gate` after moving the alias |
| 13 | Normal traffic kept the drift alert firing: the backend measured uploads at full size while the baseline was built from 128 px images (blur PSI 5.9 on normal photos), and an empty bin in a 50-upload window added ~1.1 to PSI | Backend measures drift on the baseline-sized image; PSI uses Jeffreys smoothing; alert threshold 1.0 for the 50-upload window (normal ≤ 0.7, drift ≈ 3.8) |

### 6.5 Known issues (not blocking)

* PyTorch warns that TorchScript is deprecated on Python 3.14. It works today. Moving to
  `torch.export` is future work.
* Golden-set F1 for *early blight* is low (0.48). Model quality is not assessed, and the
  gate floor is still met.
