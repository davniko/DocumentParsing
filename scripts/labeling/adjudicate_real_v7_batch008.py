"""Finite, source/PDF-reviewed batch008 decisions; never a labeling heuristic.

These authored decisions feed the shared exact-before/after receipt writer.
They do not edit OCR, frozen agent outputs, or production labeling instructions.
"""

import json

from record_real_v7_reviews import record
from review_real_v7_batch import ROOT, read

from document_ocr.atomic import atomic_write_json

BATCH = ROOT / "artifacts/kie-labeling/direct-real-batch008-20261005"
NOTES = {
    1: (
        "PDF/OCR compared: populated unconditional Consigned to order of "
        "heading is negotiable. One fire-pump consignment, 15 packages, 7519 "
        "KG/33.47 M3, CAIU4267928/AOL26849/40HC. Date 23-Jun-23 is both issue"
        " and on-board. EGYP remains verbatim in consignee address, not "
        "silently expanded. PDF-only BL omitted."
    ),
    2: (
        "Four pump-sprayer container rows own 852/772/867/826 cartons; exact "
        "sums 3317, 41735.6 KG, 266 CBM. PDF distinguishes the misplaced OCR "
        "NINGBO from an empty vessel field. PDF-only BL/prepaid absent from "
        "OCR remain absent. OCR email contains a space, retained rather than "
        "inventing a deliverable address."
    ),
    3: (
        "One press-machine-parts consignment over three containers, 8/19/15 "
        "pieces and 6460/8340/7340 KG. Full memberships and seals checked. "
        "PDF-only 40HQ and issue details not added. Shipped date 13 OCT 2024 "
        "prints in OCR."
    ),
    4: (
        "PDF/OCR machinery parts: 2 packages, decimal-comma 1210,000 KG and "
        "5,540 CBM, ARKU 839759/6 and seal 22987. Canonical identifier "
        "removes formatting slash only. Spelled-out 40-foot HIGH CUBE "
        "supports same complete category as 40HC. Freight prepaid is an "
        "arrangement, not a payment locality. No PDF-only bill number."
    ),
    5: (
        "Reserve not selected: PDF prints 68 CBM which OCR drops, along with "
        "bill number/on-board stamp; retain source rather than filling "
        "PDF-only cargo values. Other candidates retain more complete cargo "
        "evidence."
    ),
    6: (
        "Front and attachment PDF/OCR: PVC COMPOUNDS plus FILTER share 1209 "
        "packages, 26159.8 KG, 44.4 CBM and one container. Two HS codes "
        "remain valid. Transit destination and delivery agent retained from "
        "OCR. PDF-only on-board date omitted."
    ),
    7: (
        "Two 20GP rows, 24 pallets each, bolts single consignment. Exact "
        "50673.1 KG/27.188 CBM. ALSTAOSOK6077J25 is captioned Booking No in "
        "OCR; PDF also uses it as BL but PDF-only role cannot invent OCR BL "
        "evidence. 21-day detention is explicit shipment instruction. MAY "
        "disambiguates both dates."
    ),
    8: (
        "Front/attachment reviewed: nine 20GP rows each 20 bags/18300 "
        "KG/26.600 CBM, same polymer. The 162 MT quantity is not explicitly "
        "net/gross. 9805 is a form footer, not telephone continuation. "
        "Printed +971 04 816 is visibly incomplete; no digits may be "
        "supplied. Shipper email has explicit ** continuation ownership; "
        "represented-company name retained, only principal postal address "
        "used."
    ),
    9: (
        "All three supplied PDF pages/OCR reviewed (printed 2/4 through 4/4; "
        "missing first page is not an omitted cargo row). Three cable rows "
        "27/23/10 drums sum 60 and 32996 KG; net 26690 KG. Consignee address "
        "restricted to its own block; goods-area continuation not imported. "
        "Notify continuation is explicitly owned. ORIGIN is not a named "
        "payment locality. Made-in wording establishes goods-origin country, "
        "not locality. General free-out charges are not handling labels."
    ),
    10: (
        "Four identical PP RAFIA portions, 990 bags and 25344.000 KG each. "
        "One goods record with 3960 bags/101376 KG. Review hold was comma "
        "insertion in evidence tokens; original numeric values support final "
        "arithmetic. PDF-only Saudi Arabia under loading port and on-board "
        "stamp not supplied to OCR target."
    ),
    11: (
        "Two CCTV/DVR accessory rows 723/1673 cartons, 19462.14 KG and "
        "129.432 CBM. Shipper VIA identity belongs to name, not address. "
        "Delivery-agent three complete printed phone strings are distinct; "
        "the second is a full local number, not a short extension. Full route"
        " headings clarify repeated vessel/POL text."
    ),
    12: (
        "Excluded incomplete OCR: PDF prints 19 REELS ON 01 PALLET; OCR loses"
        " 19 reels. Accepting a one-pallet target would silently substitute "
        "the outer packaging level for missing inner evidence."
    ),
    13: (
        "One WHITE HIBISCUS FLOWER consignment, 1300 bags split 650/650; "
        "26200 KG gross/26000 KG net. Dates are unambiguous ISO. Exact 13-day"
        " combined detention/demurrage statement is shipment-specific. "
        "Exporter-ID disagreement is preserved in full references, which are "
        "projected out."
    ),
    14: (
        "PDF/OCR: 20 packages CHEMICAL FOR LEATHER, 21320 KG, "
        "MCLU3083736/5830072. Standard CONTAINER has no printed OCR size. "
        "RUTH BORCHARD_V.453 has a clear underscore delimiter: literal "
        "scanner rejection is formatting, not invention. Source generic "
        "shipper notification responsibility is not goods handling. TAREK is "
        "uncaptained identity in consignee/notify names."
    ),
    15: (
        "Seven compressor containers: 24/22/24/24/24/24/24 pallets. Gross "
        "sums 99974.544 KG, volume 111.105 M3, explicit net 93002.544 KG. "
        "Each seal/row checked. No printed equipment dimensions: IDs/seals "
        "only. MAR 4TH 2025 disambiguates date; unrelated PO phone not added "
        "to parties."
    ),
    16: (
        "PDF resolves OCR roles: Bangkok loading, Singapore discharge, "
        "Alexandria Old Port final destination. Do not turn Singapore into "
        "inferred transshipment. Four pallets Honda engines/1280 KG/4.185 M3."
        " On-board stamp 21/08/2023 overlaps boilerplate but date is in OCR "
        "and PDF establishes role. MADE IN THAILAND is country origin. Thai "
        "principal address remains separate from represented Japanese party."
    ),
    17: (
        "One shoes/bags consignment, two HS codes and one aggregate 592 "
        "cartons/7630 KG/68 CBM. TFLU4826864/SLG040528/40HQ checked. Selected"
        " HM2 discharge-yard instruction retained; no unsupported consignee "
        "or equipment inference."
    ),
    18: (
        "Excluded incomplete OCR: PDF forwarder DHL and complete "
        "delivery-agent block are absent from OCR, as are selected "
        "prepaid/on-board details. Do not fill missing parties from PDF."
    ),
    19: (
        "Reserve not selected: physical goods mass is printed as OCR KOS, "
        "outside the supported unit vocabulary, whereas PDF is KGS. Keeping "
        "would require omitting gross mass or extending OCR-unit repair "
        "policy. Existing candidate correctly uses actual 3.050 CBM rather "
        "than taxable 5.955 CBM; no arbitrary unit repair applied."
    ),
    20: (
        "PDF/OCR inner 1000 bags on 20 heat-treated pallets. 20 KG is bag "
        "capacity, not independently inferred net weight. Gross 20500 KG, "
        "exact long HS preserved. TEXAS UNITED STATES OF AMERICA is "
        "explicitly labeled goods origin. Delivery-agent /108 is an "
        "extension, not another phone. Dates and container checked."
    ),
    21: (
        "One used-machine consignment, 52 packages/7060 KG/40 CBM, "
        "CAIU4204766/04070/40HQ. Front printed BL absent from OCR, no "
        "invented value. MAY date, Genoa issue/payment locality, Venice "
        "loading/receipt verified."
    ),
    22: (
        "Both PDF pages show one pine-timber consignment, all 13 container "
        "rows and exact 129 packages/328997 KG/595.784 M3 totals. Attachment "
        "MASTERBILLNO explicitly establishes master-number role; front OCR "
        "booking caption does not establish bill-number role. Principal "
        "Kyrgyz postal address not merged with represented UAE company. "
        "Notify CONTACTS *** owns the starred continuation email/phone/name; "
        "consignee does not inherit it."
    ),
    23: (
        "Reserve not selected: front MTD number, shipped-on-board stamp and "
        "original count are absent from OCR; 299467 is an unrelated footer "
        "stamp, not bill number. Marks-column factory contact/address must "
        "not overwrite shipper. Inner 12 drums and printed totals are "
        "recoverable, but more complete reserves meet quota without this "
        "weaker source. Not declared irreparable."
    ),
    24: (
        "All three pages reviewed. Two identical lighter portions own 1820 "
        "cartons each; total 3640, 52416 KG, 119.1 CBM. PSN LIGHTERS and "
        "Chemical name LIGHTER describe same product, not two goods. "
        "UN1057/IMDG2.1 gives GASES. PDF-only HS codes not inferred into OCR "
        "target. Full shipper O/B identity retained."
    ),
    25: (
        "PDF/OCR one LOADER consignment, 50 packages allocated 25/25, 33900 "
        "KG/100 CBM. LOADER is a product description, not discarded equipment"
        " boilerplate. Order wording appears only in conditional caption. ** "
        "consignee telephone occurs outside its block and stays excluded. "
        "Empty payment columns do not imply prepaid despite PDF-only selected"
        " text."
    ),
    26: (
        "Both pages: same used-machine/parts lot repeated for three "
        "containers, one piece each. Sum 3 pieces and 61360 KG, no "
        "independent product accounting that would require multiple goods. "
        "Blank issue date stays absent; 2024-03-16 is on-board only. "
        "Corporate and uncaptained personal consignee identity kept together."
    ),
    27: (
        "Excluded incomplete OCR: PDF has a substantial carrier "
        "destination-agent block with contact and selected terms missing "
        "entirely from OCR. Cargo is one tyres/tubes consignment, but do not "
        "publish as complete-source baseline."
    ),
    28: (
        "Excluded incomplete OCR: PDF prints Notify: Same as Consignee, "
        "omitted on both OCR front-page copies. Four pages are original/copy "
        "plus two corresponding sparse attachments, not four independent "
        "cargo lots. Keep source; do not inject missing party relation."
    ),
    29: (
        "Front and attachment: same machinery assortment repeated over "
        "9/16/10 packages, total 35/12380 KG, three containers. Three "
        "distinct HS codes remain. 02/10 and 02.10 are ambiguous numeric "
        "dates and stay absent. TURKEY is printed only with issued-at "
        "locality, not loading wording. Full local seven-digit delivery "
        "phones are not extensions. Printed package type and selected 21-day "
        "free-time retained."
    ),
    30: (
        "All glycerin pages reviewed. Eight one-flexibag rows, gross 201535 "
        "KG and exact 152 CBM sum; textual comma net 200,455 KG reconciles "
        "with shipment quantity and row scale. Origin PANAMA is exporter "
        "registration, not shipper Brazil address or goods origin. 15-day "
        "detention is a selected shipment instruction."
    ),
    31: (
        "One harness-materials consignment, nine containers/261 packages. All"
        " row amounts sum 80162.080 KG; same description covers all. "
        "Non-negotiable sea waybill, dates 7/8 October ISO, no extra goods "
        "created from plastic reels/wire/material list."
    ),
    32: (
        "Front/attachment two cable containers 10+22 packages, 44708 KG. "
        "Complete 40HC labels from header. Phone +2 continues on next line "
        "within same party block, not a new number. 07/01/2026 ambiguous date"
        " omitted. No payment country licensed by own wording. Retain 21 days"
        " free demurrage and detention."
    ),
    33: (
        "Reserve not selected: genuine maritime eight-tractor consignment "
        "without containers, not a dummy. Printed gross unit itself reads KOS"
        " in PDF/OCR, not an established schema unit; no unit inferred or "
        "model-enriched mass retained. Prefer reserves with fully supported "
        "numeric targets."
    ),
    34: (
        "Held out of this baseline: genuine bulk wheat, TO ORDER, but 500,000"
        " MT has unresolved comma magnitude (500 vs 500000). No independent "
        "numeric evidence fixes interpretation, and destination is generic "
        "rather than named port. Not declared a dummy or irreparable."
    ),
    35: (
        "Excluded incomplete numeric OCR: PDF coil table has QUANTITY/WEIGHT "
        "(MT), but MT and footer issue details disappear in OCR. Do not "
        "assign metric-tonne units visible only on PDF. Eighteen coils are "
        "one product grade/size, not 18 goods."
    ),
    36: (
        "Genuine RoRo combine, one UNIT/15000 KG/174.243 CBM. Imperial "
        "equivalents do not add counts. Care-of identity retained, not "
        "guessed contact ownership. Notify uncaptioned Mohammed Ahmed Dosoky "
        "belongs with company name under agreed rule. EGYI is preserved in "
        "address, not supplied as missing country. Duplicate page boilerplate"
        " not extra cargo."
    ),
    37: (
        "Genuine RoRo single CAT loader with VIN/dimensions in description; "
        "28349.520 KG/123.135 CBM. No container fabricated from VIN. Date "
        "07/05/2025 ambiguous, left absent. EGYFI truncated printed country "
        "not expanded using city knowledge. Forwarder kept; carrier "
        "projection applied separately."
    ),
    38: (
        "Real LCL draft, not test: 3 packages of two monitor sizes under one "
        "aggregate accounting, 613 KG and 6.120 CBM. PDF/OCR EDK215348 is a "
        "marks/reference token, not a container; model fabricated EDKZ1534871"
        " must be removed. No allocation invented. Secondary warehouse phrase"
        " is a delivery/final-destination statement. Canonicalize "
        "null-bearing failed draft after explicit corrections."
    ),
    39: (
        "Held out of baseline: actual bulk-wheat B/L with unclear 500,000 MT "
        "decimal convention and generic ANY EGYPTIAN MEDITERRANEAN PORT. "
        "Source date is clear but does not settle numeric magnitude. Preserve"
        " without guessing; not a dummy."
    ),
    40: (
        "Validation reassignment approved. Genuine breakbulk birch plywood, "
        "192 packages/121956 KG/174,2208 CBM (unambiguous four-place decimal "
        "comma). No container. Shipper represented identity has no address "
        "printed. Country RUSSIA belongs to loading phrase, not bare "
        "issued-at NOVOROSSIYSK. THREE belongs to original-count field; 17-th"
        " June date clear."
    ),
    41: (
        "Validation reassignment approved. Same BORSTAR HE6062 over "
        "990/891/539 bags, sum 2420; gross 61952 KG/net60500 KG. PDF type "
        "statement 3x40 high-cube is absent OCR so labels contain only "
        "IDs/seals. All OCR volumes are 0.0 placeholders; model 0.0297 is "
        "unsupported and removed. Explicit 21-day free time retained."
    ),
    42: (
        "Validation reassignment approved. TO ORDER overrides "
        "nonnegotiable-copy watermark and provides no named consignee. Two "
        "kitchen-parts container allocations 816/989 sum1805; gross17353.73 "
        "KG and volume131.192 CBM exact row totals. No PDF-only BL number. No"
        " inferred country for Sokhna/Ningbo. TEDA ROYAL transit wording "
        "retained."
    ),
}
EXCLUDED = {5, 12, 18, 19, 23, 27, 28, 33, 34, 35, 39}


def action(path, value=None, *, remove=False, reason):
    return (
        dict(path=path, remove=True, reason=reason)
        if remove
        else dict(path=path, value=value, reason=reason)
    )


G = ["goodsItemDetails", 0]
ACTIONS = {
    4: [
        action(
            ["containerInformation", 0, "typeDescription"],
            remove=True,
            reason="Spelled-out complete high-cube equipment maps to canonical pair.",
        ),
        action(
            ["containerInformation", 0, "sizeCategory"],
            "FORTY_FOOT_HIGH_CUBE",
            reason="Printed 40-foot high cube.",
        ),
        action(
            ["containerInformation", 0, "typeCategory"],
            "GENERAL_PURPOSE",
            reason="Complete 40-foot high-cube wording without specialized family.",
        ),
    ],
    7: [
        action(
            ["billOfLadingNumber"],
            remove=True,
            reason="OCR captions this value only Booking No.; no PDF-only role promotion.",
        ),
        action(
            [*G, "handlingInstructions"],
            ["21 DAYS FREE DETENTION AT FINAL DESTINATION"],
            reason="Explicit selected shipment free time.",
        ),
    ],
    8: [
        action(
            ["parties", "shipper", "contactDetails", "phoneNumbers"],
            remove=True,
            reason="Printed telephone is visibly incomplete; footer 9805 is not a continuation.",
        )
    ],
    9: [
        action(
            [*G, "handlingInstructions"],
            remove=True,
            reason="Generic free-out expense allocation, not goods handling.",
        )
    ],
    11: [
        action(
            ["parties", "shipper", "name"],
            "B F T ELECTRONICS LLC VIA DAHUA TECHNOLOGY(HK) LIMITED",
            reason="Retain linked printed identity; no contact/address transfer.",
        ),
        action(
            ["parties", "deliveryAgent", "contactDetails", "phoneNumbers"],
            ["+202 226 873 91", "226 873 92", "0100 517 80 52"],
            reason="Second printed string is a complete seven-digit local number, not a suffix.",
        ),
    ],
    13: [
        action(
            [*G, "handlingInstructions"],
            [
                (
                    "Applicable free time 13 days Combined (Detention & Demurrage) at "
                    "(port of discharge / place of delivery)"
                )
            ],
            reason="Explicit selected combined free time.",
        )
    ],
    14: [
        action(
            [*G, "handlingInstructions"],
            remove=True,
            reason="Generic notification responsibility is not cargo handling.",
        )
    ],
    16: [
        action(
            ["shippedOnBoardDate"],
            "2023-08-21",
            reason="PDF establishes on-board stamp role; exact 21/08/2023 is present in OCR.",
        )
    ],
    20: [
        action(
            [*G, "origin"],
            {"name": "TEXAS UNITED STATES OF AMERICA"},
            reason=(
                "Explicit POINT (STATE) OF ORIGIN: TEXAS UNITED STATES OF AMERICA; "
                "GoodsOrigin.name retains full printed origin wording."
            ),
        )
    ],
    22: [
        action(
            ["parties", "notifyParties", 0, "contactDetails"],
            {
                "contactName": "SALAH TAWFIK",
                "phoneNumbers": ["+20 100 0096670"],
                "emailAddresses": ["TAWFFIK-2011@HOTMAIL.COM"],
            },
            reason=(
                "Notify CONTACTS *** explicitly owns starred continuation; not "
                "imported into consignee."
            ),
        )
    ],
    24: [
        action(
            [*G, "description"],
            "LIGHTERS",
            reason=(
                "PSN and singular chemical synonym describe one product; avoid "
                "duplicate LIGHTERS, LIGHTER."
            ),
        )
    ],
    25: [
        action(
            [*G, "description"],
            "LOADER",
            reason="Printed product under description; omitted by model.",
        ),
        action(
            ["parties", "consignee", "addressLine"],
            "NO.13 ELHELIG ST, FIRST OF ROAD, ELHOSUSUIY ELOUMOMIY-M-ELHANK-ELMELIK",
            reason="Rejoin hyphenated physical line continuation; do not change address tokens.",
        ),
    ],
    29: [
        action(
            ["route", "portOfLoading", "country"],
            remove=True,
            reason="TURKEY is attached only to issued-at, not loading OCR wording.",
        ),
        action(
            ["parties", "deliveryAgent", "contactDetails", "phoneNumbers"],
            ["(+203) 4865088", "4865082", "4865080", "4863088"],
            reason=(
                "Full seven-digit printed local alternatives, not short extensions; "
                "do not synthesize prefixes."
            ),
        ),
        action(
            [*G, "numberAndTypeOfPackages", 0, "typeCategory"],
            "PACKAGE_PACKAGE",
            reason="Every local row and total explicitly say PACKAGES.",
        ),
        action(
            [*G, "handlingInstructions"],
            ["Shipped on deck at shipper's sole risk", "21 DAYS FREETIME OF DEMURRAGE"],
            reason="Keep cargo on-deck note and selected shipment free time.",
        ),
    ],
    30: [
        action(
            [*G, "handlingInstructions"],
            ["Applicable free time 15 days Detention at (port of discharge / place of delivery)"],
            reason="Explicit selected shipment free time.",
        )
    ],
    32: [
        action(
            ["freight", "paymentPlace", "country"],
            remove=True,
            reason="Payment-place text prints only ISTANBUL, no country.",
        ),
        action(
            [*G, "handlingInstructions"],
            ["40/HC 21 DAYS OF FREETIME OF DEMURRAGE & DETENTION"],
            reason="Explicit shipment-specific free-time statement.",
        ),
    ],
    36: [
        action(
            ["parties", "notifyParties", 0, "name"],
            "DIRECT INTERNATIONAL LOGISTICS Mohammed Ahmed Dosoky",
            reason="Uncaptioned personal identity belongs in name under approved rule.",
        ),
        action(
            ["parties", "notifyParties", 0, "contactDetails", "contactName"],
            remove=True,
            reason="No contact caption or role identifies the person separately.",
        ),
    ],
    38: [
        action(
            ["containerInformation"],
            remove=True,
            reason=(
                "EDKZ1534871 is fabricated from marks/reference text EDK215348; no "
                "container printed."
            ),
        ),
        action(
            [*G, "grossWeight"],
            {"value": 613.0, "unit": "kilogram"},
            reason=(
                "Both units print; retain primary metric mass consistent with other "
                "dual-unit records."
            ),
        ),
        action(
            [*G, "numberAndTypeOfPackages", 0, "typeCategory"],
            "PACKAGE_PACKAGE",
            reason="3 PACKAGES is explicit at row heading.",
        ),
    ],
    40: [
        action(
            ["placeOfIssue", "country"],
            remove=True,
            reason="Country belongs to distinct loading field, not bare issue place.",
        ),
        action(
            ["transport", "vesselFlagCountry"],
            "SIERRA LIONE",
            reason="Preserve OCR spelling in full archive; flag is removed in reduced projection.",
        ),
    ],
    41: [
        action(
            [*G, "volume"],
            remove=True,
            reason=(
                "0.0297 is invented; only zero placeholders occur in OCR, not "
                "positive shipment volume."
            ),
        ),
        action(
            [*G, "handlingInstructions"],
            ["21 DAYS FREE OF DEMURRAGE AT PORT OF DISCHARGE"],
            reason="Explicit selected shipment free time.",
        ),
    ],
}


def main():
    review = BATCH / "manual-review"
    review.mkdir(exist_ok=True)
    assert not (review / "decisions.json").exists(), "Do not overwrite reviewed receipts"
    rows = read(BATCH / "selection.json")["documents"]
    assert set(NOTES) == {r["index"] for r in rows}
    atomic_write_json(review / "source-notes.json", {f"{i:03}": n for i, n in NOTES.items()})
    atomic_write_json(
        review / "exclusions.json",
        {r["name"]: NOTES[r["index"]] for r in rows if r["index"] in EXCLUDED},
    )
    instructions = [
        dict(index=i, notes=[NOTES[i]], actions=ACTIONS.get(i, []), canonicalize=i == 38)
        for i in sorted(NOTES.keys() - EXCLUDED)
    ]
    atomic_write_json(review / "instructions.json", instructions)
    record(BATCH, instructions)
    print(json.dumps({"accepted": len(instructions), "excluded": len(EXCLUDED)}))


if __name__ == "__main__":
    main()
