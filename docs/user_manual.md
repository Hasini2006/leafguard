# LeafGuard: User Manual

LeafGuard looks at a photo of a tomato leaf and tells you whether it looks healthy or
shows signs of a common disease. You do not need any technical knowledge to use it.


## 1. Opening LeafGuard

Open a web browser (Chrome, Firefox, Edge or Safari) and go to:

**http://localhost:8080**

If someone set LeafGuard up on another computer for you, use the address they gave you
instead of `localhost`. The page works on phones as well as on computers.

At the top of the page there are two links:

| Link | What it is for |
|---|---|
| **Check leaf** | Check a leaf. This is the main page. |
| **Help** | A short version of this manual, with a key of the colours. |

The line at the bottom of the page shows which model version is answering, for example
*"Model leafguard version 4 (champion)"*. If it says *"still starting up"*, wait a minute
and reload the page. The link *"For operators: pipeline console"* is for the people who
look after LeafGuard; farmers do not need it.

## 2. Checking a leaf in three steps

**Step 1: Press "Upload photo".**
Pick a photo of one tomato leaf from your phone or computer. On a computer you can also
drag a photo onto the box.

* Accepted: JPG, PNG or WEBP photos up to 5 MB.
* The photo appears in a preview, so you can check it is the right one. Press
  **New photo** to change it.

**Step 2: Press "Check this leaf".**
The button reads *"Checking..."* while it works. This usually takes about a second.

**Step 3: Look at the coloured label.**

| Label | Meaning |
|---|---|
| Green **Healthy** | The leaf looks healthy |
| Red **Disease found** | A disease was found; its name is shown under the label |
| Yellow **Not sure** | The photo was not clear enough: take a new, clearer photo |

The rest of the result card:

| Part of the result | What it means |
|---|---|
| **Confidence** | How strongly the model prefers this answer, as a percentage and a bar |
| **What to do** | General advice for that disease |
| **Other possibilities** | Tap to see the next two closest matches and their scores |
| **Disclaimer** | A reminder to confirm with an expert |

### When the label is yellow "Not sure"

LeafGuard says **Not sure** instead of guessing when the photo is unclear or does not
look like a leaf. You will see photo tips instead of advice. Take a new photo:

* fill the frame with **one** leaf;
* use **daylight** and avoid strong shadows or flash;
* hold the phone **steady** so the photo is sharp;
* use a plain background if you can.

## 3. Telling LeafGuard whether it was right

Under each result it asks **"Was this right?"**

* Press **Yes** if the answer matches what you know.
* Press **No** if it was wrong. A list appears: choose the correct disease and click
  **Send**.

Your answer helps improve the next version of the model. When it is saved, you will see
*"Thank you. This helps improve LeafGuard."* Photos you correct are stored in encrypted
form.

## 4. Recent checks

The *Recent checks* box at the bottom of the page lists your last five results. It is kept
only in your own browser and never leaves your computer.

## 5. What LeafGuard can recognise

| Name shown | What it usually looks like |
|---|---|
| Healthy | No signs of the diseases below |
| Early blight | Dark spots with rings, usually on older, lower leaves |
| Late blight | Large greasy grey-brown patches that spread quickly in cool, damp weather |
| Bacterial spot | Many small, dark, water-soaked spots |
| Leaf mould | Pale yellow patches on top, olive-brown fuzz underneath |
| Septoria leaf spot | Many small round spots with pale centres and dark edges |

Anything else (another disease, pests, a different plant) will be matched to the closest
of these. That is why the uncertain answer and an expert's opinion matter.

## 6. Messages you might see, and what to do

| Message | What to do |
|---|---|
| *That file type is not supported...* | Choose a JPG, PNG or WEBP photo, not a PDF or document. |
| *That photo is ... Please choose one under 5 MB.* | Use a smaller photo size on your phone, or shrink the picture. |
| *That image is only ... pixels.* | The picture is too small. Use the original photo, not a thumbnail. |
| *That file could not be read as an image. It may be damaged.* | Take or export the photo again. |
| *The diagnosis service is starting up or unavailable...* | Wait a minute and press **Check this leaf** again. |
| *Something went wrong on our side.* | Try again. If it keeps happening, tell whoever runs LeafGuard. |

## 7. Limits worth knowing

* It learned from **laboratory photos** of single leaves on plain backgrounds, so photos
  from a field are harder for it.
* The confidence score is **not** the chance that the answer is right. It only shows how
  much the model prefers one answer.
* It looks at one leaf only and knows nothing about the rest of the plant, the soil or the
  weather.

---

## Appendix: for the person running LeafGuard

| Task | Command (in the project folder) |
|---|---|
| Start everything | `make up` (also attaches your Grafana to the stack network) |
| Stop everything | `make down` |
| Is everything running? | `make ps` |
| Rebuild the model after new data | `dvc repro`, then `docker compose restart model-server` |
| Go back to an earlier model | `make models` to list versions, then `make rollback TO=<version>` |
| See pipeline runs, failures and model versions | http://localhost:8080/pipeline.html (link in the page footer) |
| Watch the service | Grafana → *LeafGuard overview* dashboard; alerts at http://localhost:9090/alerts |

The full setup guide is in the project [README](../README.md).
