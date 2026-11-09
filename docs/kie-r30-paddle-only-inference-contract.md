# R30 Paddle-only inputs: inference and hosting contract

Verified against the formatter, OCR configuration, model release and current R30
dataset on **8 November 2026**. This document specifies the preprocessing change
from R28; it is not a proposal for a different input format.

**R30 consumes PaddleOCR's recognized text and its measured geometry directly.
GLM-OCR transcription and GLM↔Paddle line matching are no longer part of the R30
inference path.** The model still receives one ordinary text string named
`positionedText`, not an image, JSON list, or separate coordinate tensor.

The completed R30 release is the rank-64 adapter selected at checkpoint 1044,
approximately epoch 5. The later rank-96 experiment uses the **same input format**;
adapter rank does not change preprocessing. Pin the intended adapter revision
and its accompanying inference assets when deploying.

## 1. What changes from R28

| Concern | R28 | R30 |
| --- | --- | --- |
| Recognized text supplied to the extraction model | GLM-OCR transcription | PaddleOCR `rec_texts` |
| Unit of serialized text | GLM transcription line | One native recognized Paddle region |
| Line order | GLM's transcript order | Paddle's returned recognition-array order |
| Geometry | Paddle regions conservatively matched to GLM lines; sometimes a union of several matched regions | The recognized region's own rectangle |
| Missing coordinate match | An unresolved line could end in ` ||` | Not an allowed region representation: every emitted region has valid geometry |
| Coordinate values seen by the model | Normalized centroid, where resolved | Normalized centroid for every recognized region |
| Paragraph/column organization | Inherited from GLM's transcription | No additional paragraph construction or semantic reordering |
| Text present in one OCR engine but not the other | GLM text remained authoritative | Paddle text alone is authoritative; no GLM filling or merging |
| Target evidence | Reviewed against GLM-conditioned inputs | Rebased/reviewed against Paddle-conditioned inputs |
| Serving preprocessing | Rasterization, two OCR paths, alignment, serialization | Rasterization, PaddleOCR, validation, serialization |

The page-marker syntax, 0–1000 centroid convention, identity prompt and basic
reduced-V7 output representation remain. The **contents and boundaries of the
lines** change substantially, even when both inputs are called `positionedText`.
Do not serve the R30 adapter with R28's GLM-conditioned strings, or assume an R28
adapter is interchangeable with R30 on Paddle-only inputs.

The new path is:

```text
PDF pages → page rasters → Paddle detection + recognition
          → validate recognized text/rectangle pairs in native order
          → serialize one centroid-suffixed line per recognized region
          → tokenize the exact resulting string → R30 extraction model
          → validate generated JSON and document relations
```

## 2. OCR and rasterization used for R30

These are the existing Paddle settings behind the R30 sources, not generic
recommended defaults for whichever Paddle release is installed later.

### Pinned OCR models and runtime

| Component | Recorded setting |
| --- | --- |
| PaddleOCR | `3.7.0` |
| PaddleX | `3.7.2` |
| Paddle runtime | `3.3.1` |
| Detector | `PaddlePaddle/PP-OCRv6_medium_det` |
| Detector revision | `8e0f56fb2ef86b461d99cfc7ac5c137738985f61` |
| Recognizer | `PaddlePaddle/PP-OCRv6_medium_rec` |
| Recognizer revision | `e5a92bcbc5cc1b494628e458d267778f0704fd7c` |
| Document orientation classification | Disabled |
| Document unwarping | Disabled |
| Text-line orientation model | Disabled |
| Word-box output | Disabled; this is recognized-region/line input, not word-level input |
| Recognition score threshold | `0.0` |
| Detection threshold / box threshold / unclip ratio | `0.3` / `0.6` / `1.5` |
| Detection size policy | `limit_side_len=64`, `limit_type="min"`, `max_side_limit=3500` |
| Recognition batch size | `16` |
| Runtime engine | `paddle_static`; HPI, TensorRT and CINN disabled |

The explicit settings and model-directory wiring are in
[the Paddle configuration](../configs/paddle_ocr.all_originals.local.yaml) and
[the backend](../src/document_ocr/paddle_ocr/backend.py). The immutable source-run
settings/model hashes are retained in
[the run provenance](../artifacts/paddle-ocr/filtered-bl-swb-max5-1918-ppocr6-v1/provenance.json).

Use the recognition output as returned at these settings. Do not add a second
confidence filter, discard short fragments, or remove repeated text. In
particular, raising the recognition threshold silently changes the evidence
available to the extraction model.

### Raster and coordinate image

The source PDFs were rendered with PDFium/pypdfium2 at a requested **200 DPI**,
bounded to **3,500 pixels on the longest side** and **12,250,000 pixels** total.
The renderer lowers the scale when needed; it does not assume every document
has the same dimensions. Pixel dimensions are computed with `ceil`, with a
bounded-scale adjustment to avoid exceeding a limit by one pixel.

Rendering uses a white background, RGB PNG output, annotations enabled and form
drawing enabled. PDF intrinsic rotation is already reflected in the displayed
page dimensions; no additional rotation is applied by the render call. R30's
Paddle document rotation/unwarping stages are disabled.

**Normalize against the actual image frame in which Paddle reports the boxes.**
For this run, that frame is `rendered_page`. The backend obtains its dimensions
from `result["doc_preprocessor_res"]["output_img"].shape[:2]` as `(height, width)`.
Use these actual pixel dimensions, not PDF points, nominal A4 dimensions,
detector-internal resized dimensions, or the largest detected text extent.

If a future deployment enables document rotation, cropping or unwarping, the
coordinate image/frame can change. That is a preprocessing change requiring
validation; do not combine transformed boxes with the original page dimensions.

The source collection was limited to at most five pages per PDF and 20,000,000
PDF bytes. These are source/renderer limits, not extra tokens or fields in the
model input. The serializer itself accepts any positive number of page frames;
longer-document behavior is not established merely because serialization works.

## 3. Map Paddle output to page/region records

For each page, keep these recognition arrays paired by the **same index**:

```text
rec_texts[i]  ↔  rec_boxes[i]  ↔  rec_polys[i]  ↔  rec_scores[i]
```

`rec_boxes[i]` supplies `(x1, y1, x2, y2)` in pixel coordinates, origin at the
top-left; x increases rightward and y downward. Use this axis-aligned rectangle
for the centroid, not the average of polygon vertices.

**Do not zip `rec_texts` with `dt_polys`.** Detections and recognized regions can
have different lengths. Detection-only regions are retained as OCR diagnostics,
but are not invented as recognized text lines. This differs from an actual
recognition entry whose text is `""`: that entry is retained with its rectangle.

The repository's `normalize_result(native, width, height)` checks array lengths,
scores, polygon/detection association and geometry. It emits `lines` in
recognition order, with `index`, `text`, `bbox`, `polygon`, `recognition_score`
and `detection_index`. Only text and rectangle-derived centroid enter the model
string. Keep scores, polygons and identifiers outside it for diagnostics/UI use.

The formatter's portable intermediate shape is:

```json
{
  "pages": [
    {
      "index": 0,
      "width": 200,
      "height": 400,
      "regions": [
        {"index": 0, "text": "3270 PACKAGE", "box": [50, 100, 90, 120]},
        {"index": 1, "text": "TCLU", "box": [10, 101, 20, 121]},
        {"index": 2, "text": "5473302", "box": [22, 101, 40, 121]}
      ]
    },
    {"index": 1, "width": 200, "height": 400, "regions": []}
  ]
}
```

Page and region indices are zero-based and contiguous. Region indexing restarts
on each page. Page frames remain in PDF page order, even if page OCR requests
complete asynchronously. When reading unordered storage such as Parquet, restore
the recorded `page_index`/`line_index` order—not a new coordinate-based order.

## 4. Exact text serialization

### One native region, one line

Append this suffix to the **exact recognized text**:

```text
<text> || <x>,<y>
```

There is one ASCII space before and after `||`, no space after the comma, and
no zero-padding. Preserve the region's existing leading/trailing spaces,
punctuation, Unicode, casing and OCR mistakes. Do not trim or collapse them.
The suffix is appended; it does not replace pre-existing whitespace.

For page width `W`, height `H`, and the region rectangle:

```python
x = round((x1 + x2) * 500 / W)
y = round((y1 + y2) * 500 / H)
```

This is the rectangle center normalized independently in x/y to a **0–1000**
grid. Retain original rectangle precision until this last step. Use Python's
nearest-integer, **ties-to-even** rounding: `round(2.5) == 2`,
`round(3.5) == 4`. JavaScript's default `Math.round` is not equivalent at ties.
Do not first round box corners or compute centroids from already-rounded bbox
coordinates.

### Page markers and newlines

Construct each page exactly as:

```python
page_text = f"--- PAGE {page_index + 1} ---\n" + "\n".join(region_lines)
positioned_text = "\n\n".join(page_texts)
```

- The visible page number starts at **1**.
- Use LF (`\n`), not CRLF, for the inserted separators.
- Do not append a final newline or apply `.strip()` to the result.
- An empty page still has its marker **and the marker's LF**. Therefore a
  document ending with an empty page naturally ends with LF; preserve it.
- An empty interior page contributes its marker LF plus the two-LF inter-page
  separator. Implement the formula rather than normalizing blank-line runs.
- Page headers are structural, so they do not receive coordinate suffixes.
- An empty recognized region is still a line such as ` || 2,1`. A whitespace-only
  region retains its whitespace before the suffix.

The intermediate example above produces these exact bytes, expressed as a
Python string literal so the final empty-page newline is visible:

```python
expected = (
    "--- PAGE 1 ---\n"
    "3270 PACKAGE || 350,275\n"
    "TCLU || 75,278\n"
    "5473302 || 155,278\n\n"
    "--- PAGE 2 ---\n"
)
```

### Native order is not paragraph reconstruction

Keep `3270 PACKAGE` before `TCLU` in that example, although its x coordinate is
larger. Do not sort by `(y, x)`, group columns, sort by centroid, merge equal-y
regions, concatenate identifier fragments, or deduplicate repeated declarations.
Paddle's own ordering has already happened upstream.

For example, `TCLU` and `5473302` remain separate input lines. The extraction
model may produce the combined identifier `TCLU5473302` in its output according
to the extraction policy. That does **not** authorize joining the input first.
Likewise, word fragments caused by physical wrapping remain separate input
regions; a policy-correct output description can reconstruct the word.

The input can interleave party names, addresses, captions and neighboring
columns. This is intentional evidence, not a request for the host to decide
field ownership before inference. Coordinates let the model associate it.

### Invalid inputs fail explicitly

Require positive finite page dimensions and finite, nonempty, in-frame boxes:

```text
0 <= x1 < x2 <= W
0 <= y1 < y2 <= H
```

Reject missing geometry, invalid indices, non-string recognition values, and
embedded CR/LF in one region's text. Do not synthesize `0,0`, copy a neighbor's
box, clamp bad geometry, drop the line, or use R28's unresolved ` ||` suffix as
a fallback. A true OCR error is not an empty page. Preserve the source result
for diagnosis and surface a preprocessing failure.

## 5. Portable centroid serializer

This standalone Python implementation takes the `pages` list defined above.
It is equivalent to the R30 centroid view; it does not run OCR or normalize an
arbitrary third-party response. The raw Paddle response must first be validated
and mapped as described in section 3.

```python
import math


def serialize_r30_centroids(pages):
    if not pages:
        raise ValueError("At least one page frame is required")
    output = []
    for page_index, page in enumerate(pages):
        if type(page["index"]) is not int or page["index"] != page_index:
            raise ValueError("Invalid native page order")
        width, height = page["width"], page["height"]
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError("Invalid page dimensions")
        lines = []
        for region_index, region in enumerate(page["regions"]):
            if type(region["index"]) is not int or region["index"] != region_index:
                raise ValueError("Invalid native region order")
            text, box = region["text"], region["box"]
            if not isinstance(text, str) or "\n" in text or "\r" in text:
                raise ValueError("One recognized region must occupy one line")
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError("Invalid rectangle")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("Empty or out-of-frame rectangle")
            x = round((x1 + x2) * 500 / width)
            y = round((y1 + y2) * 500 / height)
            lines.append(f"{text} || {x},{y}")
        output.append(f"--- PAGE {page_index + 1} ---\n" + "\n".join(lines))
    return "\n\n".join(output)
```

Within this repository, prefer the canonical implementation rather than a
second copy:

```python
from document_ocr.spatial_inputs.paddle import pages_from_record, serialize

# record contains the validated pages list; do not pass the whole record to the model.
positioned_text = serialize(pages_from_record(record), geometry="centroid")
```

## 6. Tokenization and model invocation

The model receives **only the resulting `positionedText` string**. A serving API
can wrap that string in JSON for transport, but extract the string before
tokenization. Do not tokenize the transport JSON, `pages`, `bboxText`, confidence
scores, region IDs, or a full Paddle JSON response.

The input-only prompt is an identity transformation. Do not prepend task
instructions, schema text, chat roles, image placeholders or a chat template.
Do not append a newline because the prompt file happens to end with one.

The released R30 inference contract specifies:

- Load the saved fast tokenizer shipped with the selected adapter.
- `add_special_tokens=True`, right padding, padding to a multiple of 8.
- **No truncation.** Check each example's unpadded token count, including special
  tokens, against **19,200**. Reject overflow explicitly; don't silently cut off
  later pages or create independently predicted chunks with invented relations.
- Reject NUL-containing or empty document input. Do not reject a valid empty
  page merely because it contributes only a marker.
- Use the pinned T5Gemma2 encoder-decoder base and matching PEFT adapter; no
  vision input to the extraction model is involved.

The completed release pins base `google/t5gemma-2-270m-270m` revision
`7c38f16641f455ef0685b18431faf1b17722d5a1`, BF16/SDPA, and explicit greedy
generation (`do_sample=False`, `num_beams=1`). Load its
`generation_config.json`, not the base model's generation defaults.

For that release, `max_length=3329` bounds the **decoder sequence**, including
its start token; it is neither the 19,200-token encoder limit nor
`max_new_tokens`. KV caching is enabled for generation. R28's released decoder
cap was 3,072; don't carry that older value into R30's deployment. The training
target-admission limit of 5,500 is not the serving generation cap.

Historical R30 evaluation used batch 2 in stable descending input-length order.
Different batching, hardware or kernels can alter BF16-generated outputs; exact
serialization parity is required but does not guarantee bit-identical predictions
across arbitrary serving backends.

The output remains direct sparse reduced-V7 JSON, not a `documentPatch` envelope.
Retain JSON/schema/relationship validation, EOS/truncation detection, and error
reporting. Removing GLM and alignment does not make generated JSON automatically
valid. Use the selected release's schema and task constraints; don't copy stale
R28 enums or infer supported output fields from all text Paddle recognizes.

## 7. What to remove or retain in an inference service

### Remove from the R30 request path

- GLM-OCR model/server calls and transcript generation.
- GLM↔Paddle exact/fuzzy/context matching, repeated-text disambiguation and
  split/merged-line matching used solely to attach geometry to GLM lines.
- Union boxes built solely for those matched GLM lines.
- GLM paragraph/table reconstruction and GLM-first text selection.
- Unresolved-coordinate suffixes and transcript preservation checks whose
  authority was the old GLM input.

If the service still hosts R28, keep these only behind the explicitly selected
R28 pipeline. Do not silently fall back to R28 formatting when R30 preprocessing
fails, or silently route an R30 request through two OCR engines.

### Retain or introduce

- Bounded PDF rasterization, complete page accounting and OCR error handling.
- The pinned Paddle detector/recognizer and text-to-recognition-geometry checks.
- Actual page image dimensions and original recognition order.
- The exact centroid serializer and input/token-budget validation.
- Native OCR/text/geometry provenance for debugging and visual overlays.
- Extraction-model loading, release pinning and output validation.

The **R30 synthesis template registry is not a serving dependency**. Templates,
sampling ledgers, anonymization plans and reviewed source-to-target ownership
receipts generate/validate training examples; they must not be required to parse
an unseen customer document. Similarly, retain original boxes for UI overlays,
but do not manufacture output-field boxes from centroids alone: R30 generates
document facts, not evidence-span coordinates.

## 8. Dataset and synthesis changes relevant to interpretation

R30 records retain three text views derived from the same page/region structure:

| Field | Purpose | Used as R30 model input? |
| --- | --- | --- |
| `pages` | Full-precision page dimensions, region text and rectangles | No; serialization authority |
| `joinedRawText` | Page markers and Paddle region text without coordinate suffixes | No |
| `positionedText` | Page markers and region text with two centroid coordinates | **Yes** |
| `bboxText` | Same regions with four normalized rectangle coordinates | No; alternate view for audits/future experiments |

Keeping full rectangles does **not** mean the current adapter was trained on
bounding-box strings. Sending `text || x1,y1,x2,y2` is a different model input.
It should remain a separate, validated model/input version, not an R30 switch
enabled merely because the renderer supports it.

The migration also changed target evidence: Paddle can reveal fields GLM missed,
omit others, or recognize different characters. Real/validation labels were
reviewed against that evidence instead of being copied blindly. Field ownership,
literal text projection and established normalization policy still apply. Do not
repair Paddle values from the PDF, arithmetic expectations, a retained GLM
transcript, or world knowledge during inference preprocessing.

Synthetic examples were re-rendered through Paddle-region templates, with newly
visible dynamic fields bound/generated where required. Full rectangles and
physical wrapping are retained before centroid serialization. Generated regions
use the pinned PaddleX rectangle-order convention to model OCR order; this is
**synthesis-only** and is not an extra sorter to apply to native OCR at inference.
Anonymized real-training inputs are likewise training preparation—not a reason
to anonymize customer input before extracting its actual parties/identifiers.

On the current November 8 dataset, an exact replay of all three text views found:

| Split | Documents | Pages | Recognized regions | Empty/whitespace regions | Empty pages |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic training | 9,400 | 18,257 | 1,662,608 | 1,071 | 3 |
| Anonymized real training | 600 | 1,160 | 104,694 | 377 | 1 |
| Validation | 60 | 111 | 9,584 | 8 | 1 |

All region rectangles validated, and `positionedText`, `bboxText` and
`joinedRawText` matched serialization exactly. Thus coordinate coverage is
**100% of emitted recognized regions**, not the old partial GLM-line match rate.
This is not a claim that OCR detected every word printed in the PDFs.

The current R30 dataset contains post-training November 8 party corrections.
The completed rank-64 model's original splits remain frozen separately; rank-96
configs use the corrected files. This does not change the input format, but
matters when reproducing historical evaluation scores.

## 9. Hosting acceptance checklist

Before removing the old preprocessing deployment:

1. Pin detector, recognizer, tokenizer, adapter and generation configuration.
2. Verify the intermediate example above byte-for-byte, including the empty-page LF.
3. Test fractional centroids with half-even ties, varied page dimensions, native
   non-coordinate-sorted order, repeated text, split identifiers, empty/space
   regions, empty pages and invalid geometry.
4. Confirm every recognized region contributes exactly one suffixed line and no
   detector-only region is accidentally paired with another region's text.
5. Compare the new host's serialization with canonical `positionedText` on a
   reviewed real cohort before judging model predictions. Do not confuse OCR
   version changes with a serializer bug or an adapter regression.
6. Verify the model receives no JSON envelope/instructions and no bbox view;
   verify token limits, generation settings and output validation independently.
7. Preserve observable errors; never route malformed OCR through guessed geometry
   or GLM fallback while reporting it as the Paddle-only path.

## 10. Implementation and evidence map

- [Canonical format/validation](../src/document_ocr/spatial_inputs/paddle.py):
  `PaddlePage`, `PaddleRegion`, `validate`, `serialize`.
- [Formatter tests](../tests/test_paddle_input_contract.py): byte-exact ordering,
  empty pages/regions, geometry and projection controls.
- [Paddle response adapter](../src/document_ocr/paddle_ocr/backend.py):
  `normalize_result`, recognized-array pairing, coordinate-image provenance.
- [OCR tests](../tests/test_paddle_ocr.py): filtered recognition versus detections,
  malformed results and explicit empty pages.
- [Rasterizer](../src/document_ocr/renderer.py) and
  [OCR configuration](../configs/paddle_ocr.all_originals.local.yaml).
- [R28 alignment path](../src/document_ocr/spatial_inputs/alignment.py): historical
  matching/union/enrichment behavior being removed from R30 serving.
- [Identity prompt loader](../src/document_ocr/training/prompting.py).
- [R30 release inference contract](../artifacts/huggingface/t5gemma2-270m-mpci-bl-r30-lora/inference_config.json),
  [generation settings](../artifacts/huggingface/t5gemma2-270m-mpci-bl-r30-lora/generation_config.json)
  and [model-loading example](../artifacts/huggingface/t5gemma2-270m-mpci-bl-r30-lora/README.md).
- [Paddle synthesis reflow](../src/document_ocr/synthesis/paddle_reflow.py):
  physical fragmentation/geometry and synthetic ordering; not a serving stage.
- [Dataset README](../data/curated/mpci-bl-r30-paddle-centroids-v1/README.md):
  active membership, migration evidence and post-training amendments.

Paths under `artifacts/` and `data/` refer to local evidence and may not be present
in a source-only checkout. The format and portable serializer above are
self-contained; the model release supplies its own inference assets.

### Verification of this document

The portable serializer and byte-exact examples above were executed, and the
serializer reproduced `positionedText` for all **10,060 current records**.
Separately, the canonical formatter reproduced all three stored text views.
The formatter, structured-OCR and historical spatial-input test suites passed
**41 tests**; all 16 relative documentation links resolve locally. This is
documentation/serialization validation, not a new GPU OCR or model-quality
evaluation. No production code, dataset, model or running training job was changed.
