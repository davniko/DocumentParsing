"""Freeze source-reviewed reserves; selection is not acceptance or certification."""
import json
from collections import Counter
from run_real_v7_batch import ROOT, SOURCE, prepare
from document_ocr.atomic import atomic_publish_json
from document_ocr.hashing import sha256_file

SCREEN = ROOT / "artifacts/kie-labeling/batch006-screen-20261005"
CHOICES = {
    "39a66fab": "Fire pump set: 15 packages; inspect missing bill header and complete supplied second page.",
    "17440e5e": "Pump sprayers: 3317 cartons distributed across four printed container rows, exact masses/volumes present.",
    "18d0efcf": "Same press-machine parts repeated over three container rows: 8+19+15 pieces, not three distinct products.",
    "218161e4": "Machinery parts: two packages with mass and volume; inspect bill-header omission versus other blocks.",
    "2d393b73": "Material of bags: one cargo, 501 cartons; booking identifier must not become an invented bill number.",
    "2d4248d3": "PVC compounds and filter: one aggregate accounting item; two HS codes do not force two goods.",
    "322e6413": "Bolts: 48 pallets split 24/24, 50673.1 kg and 27.188 CBM; verify date caption using PDF.",
    "5718f96f": "Super absorbent polymer: nine printed allocations of 20 bags; sum is exact, not nine goods.",
    "8591760c": "Cables: 60 drums split 27/23/10; complete declared gross/net totals and attachments.",
    "8a73dafe": "PP rafia: 3960 bags split over four container rows; preserve units without inferring 99 MT as gross.",
    "8cafc8c1": "CCTV cameras/DVR/accessories: one aggregate carton total, two container splits.",
    "8efa78b5": "ALU-ALU: one pallet and explicitly printed gross/net/volume; preserve product specifications.",
    "943c8f9b": "White hibiscus: 1300 bags, two matching allocations; review OCR headers and exporter-number conflict.",
    "9dead588": "Leather chemicals: one 20-package shipment with gross mass; review abbreviated agent block.",
    "9e997126": "Hermetic compressors: seven allocation rows totaling 166 pallets; preserve printed mass/volume precision.",
    "b2ba22f2": "Honda engines: four pallets with mass/volume; PDF can resolve transshipment heading only.",
    "b5b6f81b": "Shoes and bags: one 592-carton shipment, multiple HS codes without separate product totals.",
    "b8982e77": "Dental furniture/equipment: one 168-carton container, full party blocks and mass/volume.",
    "de1dd1ac": "Electric motors/gearboxes: four packages, real versus chargeable CBM distinguished; inspect mass-unit OCR.",
    "dfaddb83": "Sudan grass seed: 1000 bags on 20 pallets; inner package level and product lot/capacity preserved.",
    "e75ff238": "Used machines: 52 packages, 7060 kg, 40 CBM; same goods statement repeated as container summary.",
    "e827ecdd": "Pine timber: 129 packages in 13 printed allocations across two pages; verify every row.",
    "f8a4eec1": "Phenoxyethanol: 12 drums on three pallets, gross 3171/net 3000 kg; marks per-drum mass not shipment total.",
    "21d3ced7": "Lighters DG: same cargo across containers; PSN is a valid product description, old missing description not decisive.",
    "b54eac08": "Loader: 50 packages in two container rows; complete source description exists despite old-label omission.",
    "2c599a8b": "Used machines and parts: repeated whole-lot cargo description in three containers; inspect complete attachment.",
    "6bbd204c": "Tyres/tubes: identical product description in two container allocations with explicit common totals.",
    "70f23755": "Borosilicate ovenware: identical product repeated over two container rows; inspect all originals for completeness.",
    "7b12b905": "Printing/lamination/sewing machinery: explicitly discussed single aggregate consignment, same list in three containers.",
    "8c42f0fd": "Technical glycerin: eight flexibags in eight containers, one aggregate commodity; inspect attached allocations.",
    "aaa232a2": "Wiring-harness materials: one aggregate product description, nine container quantities total 261 packages.",
    "cfdd4f38": "Cable shipment: source attachment determines whether repeated rows are same cargo; no automatic merge.",
    "0cd51c94": "RoRo used tractors: genuine non-container cargo, not missing container labels; PDF verification required.",
    "182bc44d": "Bulk milling wheat: genuine non-container shipment; preserve printed metric units and negotiability.",
    "2d2bd911": "Steel coils: breakbulk, 18 coils; distinguish own shipment quantity from vessel-condition boilerplate.",
    "600a1e89": "Used combine: RoRo single unit, explicit mass and volume; no fabricated container.",
    "dd8cea1c": "Used loader: RoRo single unit, explicit metric and imperial measurements; choose consistent printed metric values.",
    "b94ea1c1": "Monitors: one aggregate three-package LCL shipment; absent container ID is permissible if PDF also has none.",
    "d2a24f8f": "Bulk wheat: actual non-container carriage, not missing equipment; inspect decimal comma and original wording.",
    "61d1d525": "PDF checked: 192 packages birch plywood, 121956 kg, 174.2208 CBM, breakbulk; full cargo/party/route present.",
    "6a1350a5": "PDF checked: same BORSTAR commodity repeated for three container allocations; all 2420 bags and mass totals printed.",
    "40e523d2": "PDF checked: complete party/route/cargo allocation text; missing bill number not invented; exact component sums recover totals.",
}
VALIDATION = {"61d1d525", "6a1350a5", "40e523d2"}

def main():
    inventory = json.loads((SCREEN / "novelty-inventory-v2.json").read_text())
    selected = []
    for prefix, note in CHOICES.items():
        matches = [r for r in inventory["documents"] if r["documentId"].startswith("doc_" + prefix)]
        assert len(matches) == 1
        r = matches[0]
        assert 1 <= r["pages"] <= 5
        assert set(r["exclusions"]) <= {"old_bill_number_missing_select_more_complete_source", "outside_container_topup_selection", "old_labels_not_single_goods", "old_description_missing"}
        selected.append(r | {"historicalSelectionFlags": r["exclusions"], "exclusions": [], "sourceReview": note})
    screened = SCREEN / "reserve008-source-screen.json"
    assignments = SCREEN / "assignments-reserve008.json"
    atomic_publish_json(screened, dict(sourceSha256=sha256_file(SOURCE), documents=selected,
        policy="Old-label screening flags replaced by authored OCR review for selection only. Full flow and manual OCR/PDF adjudication still required. No quality gate relaxed; significant omissions and real multiple-goods remain exclusions."))
    docs = [dict(documentId=r["documentId"], historicalSplit=r["split"], split="validation" if r["documentId"][4:12] in VALIDATION else "train") for r in selected]
    counts = Counter(r["split"] for r in docs)
    atomic_publish_json(assignments, dict(authorization="User approved fresh historical-training sources for the new validation split; disjoint from new training, not claimed unseen by historical models.", documents=docs, splits=dict(counts)))
    prepare(ROOT / "artifacts/kie-labeling/direct-real-batch008-20261005", train=counts["train"], validation=counts["validation"], seed=2026100508, inventory=screened, assignments=assignments)
    print(counts)

if __name__ == "__main__":
    main()
