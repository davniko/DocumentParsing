"""Finite manual decisions for batch003; no source OCR or agent artifact edits."""
# Long lines below are literal source text and human adjudication records.
# ruff: noqa: E501

import copy


def reviewed(index, target):
    patch = copy.deepcopy(target["documentPatch"])
    notes = []

    def set_value(path, value, reason):
        node = patch
        for key in path[:-1]:
            node = node[key]
        if value is None:
            del node[path[-1]]
        else:
            node[path[-1]] = value
        notes.append(reason)

    if index == 1:
        for role in ("portOfDischarge", "placeOfDelivery"):
            set_value(
                ["route", role, "name"],
                "ALEXANDRIA OLD PORT",
                "Retain the distinguishing OLD PORT locality; PUBLIC TML is a facility descriptor.",
            )
        notes.append(
            "Owned shipper and destination-agent continuations verified across page break; no bare footer identifier added."
        )
    if index == 2:
        set_value(
            ["negotiability"],
            "non_negotiable",
            "Actual named consignee has no order instruction; generic negotiable-copy caption does not override it.",
        )
        set_value(
            ["goodsItemDetails", 0, "grossWeight"],
            None,
            "GW amount has no OCR-grounded mass unit; KGS explicitly qualifies NET, not a shared gross column.",
        )
        refs = patch["forwardingAndExportReferences"]
        refs.remove("Ref.: P.O. n°00550 dated 26/03/2025")
        refs.insert(1, "B/L Ref. N°: I 025/01")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "Restore explicit B/L reference; package-label PO remains solely in marks.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 DAYS FREE TIME AT DESTINATION"],
            "Selected shipment free-time instruction is printed.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            patch["goodsItemDetails"][0]["marksAndNumbers"]
            + ["COUNTRY OF ORIGIN: EUROPEAN UNION - ITALY"],
            "Explicit SHIPPING MARKS block contains the origin declaration as well as consignee/PO marks.",
        )
        notes.append(
            "Net token 17,600,00 has a decimal punctuation error; parallel 17,600.00 establishes the same numeric magnitude. Seven bundles are packages; 35 PCS is product count."
        )
    if index == 4:
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["Freetime Extension 8 days"],
            "Selected eight-day free-time extension is shipment-specific, not a generic tariff.",
        )
        notes.append(
            "Party continuations and postcode 144911 retained; no country inferred from exporter registration or UK locality."
        )
    if index == 3:
        description = patch["goodsItemDetails"][0]["description"]
        lines = list(dict.fromkeys(description.splitlines()))
        set_value(
            ["goodsItemDetails", 0, "description"],
            " ".join(lines) + " Label",
            "Join physical lines, retain each distinct printed product phrase once, and include the terminal Label product; the assortment is jointly accounted, not separately quantified.",
        )
        notes.append(
            "Ten distinct container rows sum exactly to 93 packages, 179238 kg and 200 CBM; repeated bill copies are not extra cargo. Both separately printed carrier and shipper seals are retained."
        )
    if index == 5:
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            [
                "Applicable free time 21 days Combined (Detention & Demurrage) at (port of discharge / place of delivery)"
            ],
            "Selected 21-day free-time instruction is printed on the continuation.",
        )
        notes.append(
            "One explicitly accounted lot of assorted machines, 28100 kg, one open-top container; booking and service-contract references occur in the second bill copy."
        )
    if index == 6:
        refs = patch["forwardingAndExportReferences"]
        refs.remove("SHIPPER'S REF.: XXXXX")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "XXXXX is an empty-form placeholder, not a shipment reference.",
        )
        set_value(
            ["goodsItemDetails", 0, "description"],
            "UNMANUFACTURED RAW TOBACCO CROP 2023, SCRAP, GRADE : SCRAP - SB PACKING TYPE : 150 KGS CARTON PACKING",
            "Use complete primary product wording and capacity once; remove repeated crop/grade wording and invented semicolon separators.",
        )
        notes.append(
            "111 cartons; 18,370.5 kg gross and 16,650 kg net are explicit totals, not per-carton masses; source shipment tariff is not selected DG cargo."
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            [
                "EXPORTER : AERIZ AGRI EXPORTS PVT. LTD.",
                "IMPORTER : AL GRAFATAH FOR TOBACCO INDUSTRY",
                "ORIGIN : INDIA",
                *patch["goodsItemDetails"][0]["marksAndNumbers"],
            ],
            "PDF clarifies ownership of the printed exporter, importer and origin package marks; product grade/crop remain description facts.",
        )
    if index == 7:
        seals = ["IN1245906", "IN1244860", "IN1244878", "IN1071631"]
        for j, seal in enumerate(seals):
            set_value(
                ["containerInformation", j, "sealNumbers"],
                patch["containerInformation"][j]["sealNumbers"] + [seal],
                "PDF container/seal rows establish the IN token as the second seal, not a cargo mark; token itself is in OCR.",
            )
        first, second = patch["goodsItemDetails"]
        goods = {
            "description": first["description"] + " " + second["description"],
            "grossWeight": {"value": 97420.8, "unit": "kilogram"},
            "netWeight": {"value": 92436.48, "unit": "kilogram"},
            "marksAndNumbers": [s for s in first["marksAndNumbers"] if s not in seals],
            "hsCodes": first["hsCodes"] + second["hsCodes"],
            "origin": first["origin"],
            "numberAndTypeOfPackages": [
                {"packageQuantity": 1888, "typeCategory": "PACKAGE_CARTON"}
            ],
            "splitGoodsPlacement": [
                {"equipmentIdentifier": c["equipmentIdentifier"]}
                for c in patch["containerInformation"]
            ],
        }
        set_value(
            ["goodsItemDetails"],
            [goods],
            "Yarn grades form one jointly accounted assortment with 1888 cartons and explicit total weights. OCR lacks per-grade/per-container quantities; retain membership only. PDF-only 472 carton allocations and CBM are not imported.",
        )
    if index == 8:
        set_value(
            ["containerInformation", 0, "typeDescription"],
            None,
            "Complete 40-foot high-cube wording supports canonical equipment categories.",
        )
        set_value(
            ["containerInformation", 0, "sizeCategory"],
            "FORTY_FOOT_HIGH_CUBE",
            "Printed complete size/height.",
        )
        set_value(
            ["containerInformation", 0, "typeCategory"],
            "GENERAL_PURPOSE",
            "Printed dry high-cube equipment, no specialized type.",
        )
        refs = patch["forwardingAndExportReferences"]
        refs.remove("X20240614375071")
        refs.append("BAILLIE REGISTRATION #: 16-1439224")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "Remove uncaptioned X token; retain the explicit Baillie registration declaration.",
        )
        notes.append(
            "Invalid printed IMO 9299625 stays absent; no check-digit correction. Twelve bundles, not 919 pieces, define packages."
        )
    if index == 9:
        set_value(
            ["goodsItemDetails", 0, "netWeight"],
            None,
            "17.60 TONS qualifies cargo quantity, not an explicit net measure. NW 17600.000 has no printed unit; do not infer a conversion. Gross row explicitly provides KGS.",
        )
        notes.append(
            "80 steel drums are inner packages on 20 pallets. Contact continuation is explicitly shared by consignee and notify; DG class and UN print in OCR."
        )
    if index == 10:
        set_value(
            ["transport", "voyageNumber"],
            "BS2426",
            "Remove the V. voyage caption prefix; keep the identifier unchanged.",
        )
        notes.append(
            "PDF confirms LIANYUNGANG and date belong to issuance, not the adjacent empty freight-payable box. No on-board date is stated."
        )
    if index == 11:
        set_value(
            ["goodsItemDetails", 0, "description"],
            "(POLYETHYLENE) XLPE FOR M.V CABLES -TREE RETARDANT TYPE FOR BONDED INSULATION SCREEN(VDE REQUIREMENTS) (CLNA TR-8142EC) (XL12) PRODUCTION DATE, 2023.10.13 2023.10.14 2023.10.15 2023.10.17 LOT NO. YW428423BH YW428523BH YW428623BH YW428823BH NAME OF MANUFACTURER/EXPORTER, HANWHA SOLUTIONS CORPORATION",
            "Retain printed product, lot, production and manufacturer wording without invented semicolon delimiters; manufacturer identity is a product fact.",
        )
        marks = patch["goodsItemDetails"][0]["marksAndNumbers"]
        marks.remove("NAME OF MANUFACTURER/EXPORTER, HANWHA SOLUTIONS CORPORATION")
        marks.append("COUNTRY OF ORIGIN: SOUTH KOREA")
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            marks,
            "Restore the explicit origin package marking and avoid duplicating manufacturer wording across fields.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            patch["forwardingAndExportReferences"] + ["HANWHA NATIONAL REGISTRATION NO: HRB30046"],
            "Explicit exporter registration continuation is OCR-grounded.",
        )
        notes.append(
            "All ten container memberships and totals retained; PDF-only package count/volume excluded. Address boundary punctuation does not alter any printed street/postcode digits."
        )
    if index == 12:
        notes.append(
            "Single 2300-carton kitchenware shipment; full NZ postcode address, empty notify and exporter reference slots, and issuer branch ownership verified."
        )
    if index == 13:
        refs = patch["forwardingAndExportReferences"]
        refs.remove("SI2024303486")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "Uncaptioned SI token is already represented in its printed marks column; no duplicate administrative-reference interpretation.",
        )
        set_value(
            ["containerInformation", 0, "typeDescription"],
            None,
            "Complete high-cube equipment phrase is canonicalizable.",
        )
        set_value(
            ["containerInformation", 0, "sizeCategory"],
            "FORTY_FOOT_HIGH_CUBE",
            "Complete printed size/height.",
        )
        set_value(
            ["containerInformation", 0, "typeCategory"],
            "GENERAL_PURPOSE",
            "Complete dry high-cube phrase.",
        )
        notes.append(
            "Unprinted bill number and carrier principal remain absent; SINERGY signs as agent. Two pallets directly assigned to the sole container; generic reverse-side DG clauses excluded."
        )
    if index == 14:
        notes.append(
            "PDF establishes PUSAN as receipt despite the missing OCR caption. Inner 433 cartons, not 10 pallets or 19000 product sets. PDF-only notify block, marks and equipment type remain absent."
        )
    if index == 15:
        set_value(
            ["billOfLadingNumber"],
            None,
            "0630977 is the preprinted form serial at the foot of the PDF, not the B/L identifier; actual B/L identifier is absent from OCR.",
        )
        notes.append(
            "Two disc products jointly accounted as 292 cartons; partial NOR equipment kept raw. No party identities, gross/volume or origin imported from PDF-only text."
        )
    if index == 16:
        set_value(
            ["parties", "carrier"],
            None,
            "BORCHARD identity and contacts occur only in PDF, not OCR. Fratelli signs for the carrier, so cannot replace it as principal.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            [
                "EXPORTER ID: IT-02-IT02822620981",
                "IMPORTER ID: 308437950",
                "ACID: 3084379502025050044",
                "IMPORTER TAXATION NUMBER: 308437950",
                "EXPORTER NUMBER: IT02822620981",
            ],
            "Keep actual distinct captions instead of constructing a composite ID caption.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 DAYS FREETIME OF DEMURRAGE"],
            "Selected free time is printed separately from general tariff conditions.",
        )
        notes.append(
            "Issue date 03/06/2025 remains absent because source has no unambiguous day/month convention. Joint 13-package alloy assortment supports both HS codes."
        )
    if index == 17:
        notes.append(
            "PDF row alignment confirms three VIN-identified vehicles with individual 1588/1587/1587 kg weights; one vehicle each in the sole container. PDF-only document number/carrier/issue date excluded."
        )
    if index == 18:
        set_value(
            ["billOfLadingNumber"],
            None,
            "This is a B/L instruction sheet: S00832596 is explicitly the internal instruction reference, not a carrier-issued B/L number.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            patch["forwardingAndExportReferences"] + ["INTERNAL REFERENCE: S00832596"],
            "Keep the supported instruction reference in its declared role.",
        )
        notes.append(
            "Delivery country UNITED KINGDOM is explicitly printed despite unusual geography; preserve source rather than substitute Egypt. No prepaid inference from CPT."
        )
    if index == 19:
        for role in ["placeOfReceipt", "portOfLoading"]:
            set_value(
                ["route", role, "name"],
                "SAVANNAH, GA",
                "Retain the printed state qualifier as part of the locality name.",
            )
        notes.append(
            "ALSO NOTIFY contact-only block remains a separate notify entry, not silently attributed from email similarity. 40HC primary declaration resolves the row OCR 40HO spelling."
        )
    if index == 20:
        for j, seal in enumerate(["ML-TW0300994", "ML-TW0300907"]):
            set_value(
                ["containerInformation", j, "sealNumbers"],
                [seal, *patch["containerInformation"][j]["sealNumbers"]],
                "Carrier seal and separately captioned shipper seal retain their respective printed punctuation.",
            )
        refs = [
            r for r in patch["forwardingAndExportReferences"] if not r.startswith(("SEEG:", "SET:"))
        ]
        refs.append("VAT NO#413467910")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "SEEG/SET identifiers are package markings, not independent administrative references; restore the consignee continuation VAT caption.",
        )
        set_value(
            ["goodsItemDetails", 0, "description"],
            "ASSY OPEN CELL P/N:BN96-62667A P/N:BN96-60281A",
            "Product part numbers remain description; P/NO fractions are package numbers, not product identifiers. Remove invented semicolon delimiters and repeated part number.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            patch["goodsItemDetails"][0]["marksAndNumbers"]
            + ["P/NO.:B82J1/4-4/4", "P/NO.:B82J1/32-32/32", "P/NO.:B82J1/52-52/52"],
            "Retain the three printed package-range marks.",
        )
        notes.append(
            "1556 inner cartons are printed globally; 36/52 pallet placements cannot be substituted as carton counts. Gross and volume are exact complete two-row sums; membership only is supported."
        )
    if index == 21:
        refs = [r for r in patch["forwardingAndExportReferences"] if not r.startswith("P/O:")]
        refs.insert(1, "SI NO:Y25T0643")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "SI/INVOICE in the main description block are administrative declarations; attachment P/O entries belong to package marks.",
        )
        marks = patch["goodsItemDetails"][0]["marksAndNumbers"]
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            [m for m in marks if not m.startswith(("SI NO:", "INVOICE NO:"))],
            "Remove administrative declarations misassigned as package marks; retain genuine attachment marks and ranges.",
        )
        notes.append(
            "553 inner cartons, 15 outer pallets; one known container means the carton placement is exact. Represented shipper phrase, two notify parties and continuation postcode retained."
        )
    if index == 22:
        set_value(
            ["goodsItemDetails", 1, "numberAndTypeOfPackages"],
            [{"typeCategory": "PACKAGE_CRATE_WOODEN"}],
            "WOODEN CRATE explicitly supplies package type for the separately weighed parts; no per-item quantity inferred from total two.",
        )
        notes.append(
            "Separately weighed mining machine and related crate are two goods. Main vessel and pre-carriage remain distinct; typed SH/CN/NP email continuations verified."
        )
    if index == 23:
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            [m for m in patch["goodsItemDetails"][0]["marksAndNumbers"] if m != "C/T NO: -"],
            "C/T NO: - is an empty package-number caption, not a nonempty mark; other package-labelled contact wording is retained.",
        )
    if index == 24:
        set_value(
            ["containerInformation", 0, "sealNumbers"],
            ["465476", "ENOS01721926"],
            "Container-detail line supplies both seals, including full ENOS prefix; these are not package marks.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            None,
            "Move the two seal identifiers to their dedicated equipment field.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 days detention free at destination"],
            "Selected free time is explicitly printed.",
        )
        notes.append(
            "PDF-only bill number, delivery field and cargo masses excluded; two named delivery contacts are retained separately from full postal address."
        )
    if index == 25:
        set_value(
            ["goodsItemDetails", 0, "description"],
            "ACSSESORIS FOR COMPROSSER WASHING MACHINE CLUTCH KETTLER SWITCH",
            "Join source product lines without invented semicolons; no independent product quantities justify splitting the assortment.",
        )
        notes.append(
            "Both printed tax-ID punctuation forms are explicit declarations, not a hallucination. Three row counts/masses/volumes reconcile to printed totals; starred email belongs to consignee."
        )
    if index == 26:
        set_value(
            ["parties", "shipper"],
            {
                "contactDetails": {
                    "contactName": "TUO KANGCUN",
                    "phoneNumbers": ["0086-755-26770000"],
                    "emailAddresses": ["TUO.KANGCUN@ZTE.COM.CN"],
                    "websiteUrls": ["WWW.ZTE.COM.CN"],
                }
            },
            "PDF establishes the single-star continuation as shipper contact; all contact values are in OCR. PDF-only shipper identity/address remain absent.",
        )
        contact = patch["parties"]["notifyParties"][0]["contactDetails"]
        contact["contactName"] = "MR.FARAG ELWARDANY; Mahmoud Samy"
        contact["phoneNumbers"].append("+20 01092220165")
        contact["emailAddresses"].append("mahmoud.elfaky@zte.com.cn")
        set_value(
            ["parties", "notifyParties", 0, "contactDetails"],
            contact,
            "Triple-star notify continuation includes the second named telephone/email contact; do not drop it or infer ownership from email domain.",
        )
        set_value(
            ["containerInformation", 0, "typeDescription"],
            None,
            "20ST explicitly identifies standard dry equipment.",
        )
        set_value(
            ["containerInformation", 0, "sizeCategory"],
            "TWENTY_FOOT_STANDARD_HEIGHT",
            "Printed 20ST size.",
        )
        set_value(
            ["containerInformation", 0, "typeCategory"],
            "GENERAL_PURPOSE",
            "Printed standard dry type.",
        )
        notes.append(
            "Gross 2209.330 kg is OCR-supported; 2230 is tare, not cargo; CBM unit is PDF-only, so 19.440 volume remains absent. Voyage primary and repeated copies establish 0RDFMW1MA."
        )
    if index == 27:
        set_value(
            ["route", "portOfDischarge", "country"],
            None,
            "EGYPT occurs in generic discharge clauses, not this route field; do not enrich ALEXANDRIA from general geography.",
        )
        notes.append(
            "28650 kg cargo and 3300 tare are separate; 40EC remains an unresolved raw equipment type. ORIENT SHIPPING SAL is not a payment locality."
        )
    if index == 28:
        set_value(
            ["parties", "consignee", "name"],
            "SAMSUNG ELECTRONICS EGYPT S.A.ESEEG",
            "SEEG is part of the printed consignee identity, not a street/address token.",
        )
        set_value(
            ["parties", "consignee", "addressLine"],
            patch["parties"]["consignee"]["addressLine"].removeprefix("SEEG "),
            "Address starts at PIECE NO.98, preserving full postal content and postcode.",
        )
        set_value(
            ["goodsItemDetails", 0, "description"],
            "ASSY OPEN CELL P/N:BN96-52421A",
            "Product part number belongs in description; B82J1/7~7/7 is a package range, retained in marks.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            [m for m in patch["goodsItemDetails"][0]["marksAndNumbers"] if m != "P/N:BN96-52421A"],
            "Avoid duplicating product identity as a shipping mark.",
        )
    if index == 29:
        for role in ["placeOfReceipt", "portOfLoading"]:
            set_value(
                ["route", role, "country"],
                "SOUTH KOREA",
                "Normalize the printed KOREA in the BUSAN phrase consistently with SEOUL issuance; country is not inferred from an absent value.",
            )
        for j in range(3):
            set_value(
                ["containerInformation", j, "typeDescription"],
                None,
                "DC 20 and 20 DC jointly establish standard dry equipment.",
            )
            set_value(
                ["containerInformation", j, "sizeCategory"],
                "TWENTY_FOOT_STANDARD_HEIGHT",
                "Printed 20-foot dry container.",
            )
            set_value(
                ["containerInformation", j, "typeCategory"], "GENERAL_PURPOSE", "Printed DC type."
            )
        set_value(
            ["parties", "deliveryAgent"],
            {
                "name": "HMM (NETHERLANDS) SHIPPING B.V.",
                "addressLine": "WESTBLAAK 180, 3012 KN ROTTERDAM, NETHERLANDS",
                "country": "NETHERLANDS",
                "contactDetails": {"phoneNumbers": ["+31-10-280-2555"]},
            },
            "Starred SHIPPING AGENT DETAILS continues onto page two with the destination agent.",
        )
        notes.append(
            "60 bags on 60 pallets; no per-container bag counts inferred. Platform EBL receipt/hash/log dates are not B/L metadata."
        )
    if index == 30:
        set_value(
            ["freight", "paymentArrangement"],
            None,
            "A payment place by itself does not establish a payable-elsewhere arrangement.",
        )
        set_value(
            ["goodsItemDetails", 0, "description"],
            "USED UNPACKED VEHICLE (S) CHEREAU TRAILER CHASSIS NUMBER(S): VM4CSD3000A101406",
            "VIN/chassis identifies the product; generic multi-vehicle liability/condition clause is not an individually selected product description.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            None,
            "Chassis identity is retained in description rather than duplicated as a package mark.",
        )
        set_value(
            ["shippedOnBoardDate"],
            "2023-05-19",
            "PDF locates the second OCR date 19-05-2023 in the on-board box; values, unlike its missing caption, are present in OCR.",
        )
        notes.append(
            "PDF confirms EIS0523 is voyage alongside a PDF-only vessel name; vessel identity and bill number remain absent."
        )
    if index == 31:
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            None,
            "Ten-hour devanning is a general numbered carrier clause, not a selected instruction for this cargo.",
        )
        notes.append(
            "One used-machine lot with six serials, not six inferred packages; cargo gross is explicitly kg in total row, while the volume unit is absent."
        )
    if index == 32:
        set_value(
            ["freight"],
            {"paymentPlace": {"name": "BANGKOK"}},
            "PDF aligns the OCR BANGKOK with freight payment; PREPAID appears only in PDF and is not imported.",
        )
        set_value(
            ["parties", "notifyParties", 0, "contactDetails", "phoneNumbers"],
            ["002023838430"],
            "Both party blocks carry the same single-star contact continuation.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            [
                patch["forwardingAndExportReferences"][0],
                "C.R.105130",
                *patch["forwardingAndExportReferences"][1:],
            ],
            "Explicit consignee commercial registration belongs in references, not address.",
        )
        descriptions = [
            "PSN: NITROCELLULOSE WITH ALCOHOL NITROCELLULOSE DSS 1/8 ( IPA DAMPED ) GROSS WT/DRUM: 146.45 KGS. NET WT / DRUM : 136.00 KGS. BATCH NO. : 24-1-107-0A00, 24-1-108-0A01 MANUFACTUING DATE: MAR-2024 EXPIRY DATE: MAR-2026",
            "PSN: NITROCELLULOSE WITH ALCOHOL NITROCELLULOSE RSX 1-23 ( IPA DAMPED ) NET WT / DRUM : 85.00 KGS. BATCH NO.24-9-020-0A00, 24-9-023-0A00, 24-9-027-0A00, 24-9-028-0A00, 24-9-029-0A00, 24-9-030-0A00, 24-9-031-0A00, 24-9-032-0A00, 24-9-033-0A00 MANUFACTUING DATE : FEB-2024,MAR-2024 EXPIRY DATE : FEB-2025,MAR-2025",
        ]
        for j, description in enumerate(descriptions):
            set_value(
                ["goodsItemDetails", j, "description"],
                description,
                "Preserve source product/batch/capacity/date wording and spelling rather than introduce semicolons or silently repair MANUFACTUING.",
            )
            set_value(
                ["goodsItemDetails", j, "marksAndNumbers"],
                ["COUNTRY OF ORIGIN : THAILAND", "EXPORTER: ABC CHEMICAL EXPORTS PVT LTD."],
                "Product batches remain description facts; explicit package origin/exporter markings belong in marks.",
            )
            set_value(
                ["goodsItemDetails", j, "handlingInstructions"],
                None,
                "Numbered ten-hour devanning boilerplate is not a selected cargo instruction.",
            )
        notes.append(
            "Distinct 80/240-drum products retain their explicit net masses and shared printed DG classification. No allocation guessed from two 160-drum equipment rows; PDF-only HS, temperature and second per-drum gross excluded."
        )
    if index == 33:
        set_value(
            ["parties", "forwardingAgent"],
            None,
            "VIA MEDICINOS is a second identity/address inside shipper box, not the explicitly empty forwarding-agent field. Keep primary shipper; preserve represented/intermediary block in the audit only.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            [
                "COMPANY ID NUMBER/ COMMERCIAL REGISTER NUMBER: 173995",
                *patch["forwardingAndExportReferences"],
            ],
            "Restore explicitly captioned consignee registration, excluded from postal address.",
        )
        notes.append(
            "PDF-only booking and notify relation excluded. NON DG remains description, not a positive DG classification. ETD/ETA are not issue/on-board dates."
        )
    if index == 34:
        set_value(
            ["route", "placeOfDelivery"],
            None,
            "Delivery field exists in PDF but its separate value/occurrence is omitted by OCR; loading and discharge occurrences do not license copying into delivery.",
        )
        set_value(
            ["parties", "shipper", "addressLine"],
            "NORTH OF GUANGRUI ROAD, GUANGRAO ECONOMIC AND DEVELOPMENT ZONE",
            "Exporter registration country is customs metadata, not a postal continuation.",
        )
        set_value(
            ["parties", "shipper", "country"],
            None,
            "No shipper postal country is printed; do not infer from registration-country metadata.",
        )
        set_value(
            ["parties", "notifyParties", 0, "addressLine"],
            patch["parties"]["notifyParties"][0]["addressLine"] + ", EGYPT",
            "Single-star EGYPT on attachment belongs to notify postal continuation.",
        )
        set_value(
            ["parties", "notifyParties", 0, "country"],
            "EGYPT",
            "Explicit owned postal continuation.",
        )
        set_value(
            ["parties", "carrier", "contactDetails"],
            {"websiteUrls": ["www.evergreen-line.com"]},
            "Carrier homepage is explicitly stated, unlike the attachment-free shipper country.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            patch["forwardingAndExportReferences"] + ["EGYPTIAN IMPORTER TAX ID: 440378737"],
            "Retain explicit importer-ID caption alongside party TAX ID.",
        )
        for j in range(3):
            set_value(
                ["containerInformation", j, "typeDescription"],
                None,
                "40H explicitly qualified HI-CUBE identifies complete equipment.",
            )
            set_value(
                ["containerInformation", j, "sizeCategory"],
                "FORTY_FOOT_HIGH_CUBE",
                "Explicit HI-CUBE.",
            )
            set_value(
                ["containerInformation", j, "typeCategory"],
                "GENERAL_PURPOSE",
                "Dry high-cube cargo equipment.",
            )
        notes.append(
            "43370 kg is exactly 16880+13170+13320, not an unsupported invented total; 192+984+980=2156. PDF-only CBM excluded. PDF confirms receipt-role of OCR Qingdao."
        )
    if index == 35:
        set_value(
            ["transport", "voyageNumber"],
            "2025-642",
            "V. is a voyage caption, not identifier prefix.",
        )
        set_value(
            ["parties", "carrier"],
            None,
            "Issuer explicitly signs AS AGENTS ONLY and its terms disclaim carrier status; do not promote an agent into carrier principal.",
        )
        set_value(
            ["goodsItemDetails", 0, "description"],
            "Lubricating oils NON HAZARDOUS",
            "Retain the explicit product nonhazardous qualifier, without creating DG labels.",
        )
        notes.append(
            "PDF resolves duplicated OCR port captions: Piraeus loading, Alexandria discharge. European decimal formatting resolves 3017.25 kg and 9.56 CBM."
        )
    if index == 36:
        set_value(
            ["placeOfIssue", "country"],
            None,
            "GERMANY qualifies Hamburg, not issue-place Langenhagen; no country borrowing from agent address.",
        )
        set_value(
            ["freight", "paymentPlace"],
            None,
            "ORIGIN is a payment-direction word, not a named locality.",
        )
        set_value(
            ["parties", "notifyParties"],
            patch["parties"]["notifyParties"][:2],
            "X-RAY FILM FACTORY is the account party under ACMA, not a third named notify role. Keep the unambiguous primary ACMA identity/address; related alternate block remains in audit.",
        )
        goods = patch["goodsItemDetails"]
        joint = {
            "description": goods[3]["description"]
            + " ENVIRONMENTALLY HAZARDOUS SUBSTANCE, LIQUID, N.O.S., (HYDROQUINONE) ACETIC ACID SOLUTION CORROSIVE LIQUID, N.O.S., (GLUTARAL)",
            "grossWeight": {"value": 24184, "unit": "kilogram"},
            "netWeight": {"value": 23044, "unit": "kilogram"},
            "handlingInstructions": [
                "FREETIME OF MAXIMUM 21 DAYS DEMURRAGE AND DETENTION COMBINED UPON DISCHARGE OF THE VESSEL AT THE SEAPORT OF ALEXANDRIA."
            ],
            "dangerousGoods": [g["dangerousGoods"][0] for g in goods[:3]],
            "numberAndTypeOfPackages": [{"packageQuantity": 80, "typeCategory": "PACKAGE_DRUM"}],
            "splitGoodsPlacement": [{"equipmentIdentifier": "APZU3990457", "packageQuantity": 80}],
        }
        set_value(
            ["goodsItemDetails"],
            [joint],
            "Commercial 80-drum film-chemical assortment is the shipment total, not additional cargo on top of its DG declarations. No source mapping links named PART A/B/C to DG subquantities. Preserve all printed product identities and DG categories in one jointly accounted goods group; avoid 99-drum double counting and residual arithmetic.",
        )
        notes.append(
            "PDF-only shipper, consignee/order declaration and fifth commercial product omitted. Numeric spaces are thousands separators. Subdeclaration counts/masses are preserved in source/audit, not invented as additional packages."
        )
    if index == 37:
        for path in [
            ["parties", "consignee", "addressLine"],
            ["parties", "notifyParties", 0, "addressLine"],
        ]:
            set_value(
                path,
                "PORT SAID, EGYPT",
                "Delineate the printed city and country postal components without stripping either.",
            )
        notes.append(
            "Uncaptioned personal names belong in identity; no issue date/payment arrangement inferred from a notice or empty charges table; free-zone text is not assigned to a new route role."
        )
    if index == 38:
        for j, grade in enumerate(["13LBS/1PLY", "20LBS/1PLY"]):
            set_value(
                ["goodsItemDetails", j, "description"],
                grade
                + " CRT QUALITY, BANGLADESH JUTE YARN, PRECISION WOUND ON PAPER CONICAL SPOOL, EACH SPOOL ABOUT 18-19 KGS. PACKED ON PALLET. LOT NO. 01",
                "Keep printed grade notation and common product specifications rather than paraphrasing COUNT/PLY wording.",
            )
            set_value(
                ["goodsItemDetails", j, "handlingInstructions"],
                ["21 DAYS FREE COMBINED DETENTION INBOUND CONTAINER"],
                "Selected free-time term applies to both cargo grades.",
            )
        notes.append(
            "40/80 pallets are independently quantified grades. 52/104 M.TONS are not explicitly gross/net; unitless container weights/volumes not promoted. No grade placement deduced from equal 20-pallet equipment rows."
        )
    if index == 39:
        set_value(
            ["forwardingAndExportReferences"],
            patch["forwardingAndExportReferences"] + ["TAX: 726246646"],
            "Retain explicit double-star consignee tax declaration, separate from address.",
        )
        notes.append(
            "Single glue shipment: printed class 3/UN1133 retained once; no inferred packing group or HS; dates independently captioned in ISO format."
        )
    if index == 40:
        notes.append(
            "Avocados: 4560 boxes, 27100 kg, 46 CBM and 5 C all explicitly assigned to the sole reefer. Party personal identity and postal components retained; selected temperature declaration is not an extra goods row."
        )
    if index == 41:
        refs = [
            r
            for r in patch["forwardingAndExportReferences"]
            if not r.startswith(("INV.NO", "ORD", "PO"))
        ]
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "Package-label invoice/order/PO identifiers stay in marks, not duplicated as administrative references.",
        )
        marks = patch["goodsItemDetails"][0]["marksAndNumbers"]
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            ["SSAMSUNG ELECTRONICS EGYPT S.A.E.", *marks],
            "First package label has the distinct printed SSAMSUNG spelling; retain it as well as later SAMSUNG labels, without silent spelling repair.",
        )
        notes.append(
            "Three marked pallet portions total 19; product and customer part numbers remain description facts. Full separate notify continuation retained."
        )
    if index == 42:
        refs = patch["forwardingAndExportReferences"]
        refs.remove("SHIPPER'S REF. XXXXX")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "Empty shipper-reference placeholder is not a value.",
        )
        notes.append(
            "18825.30 is explicit net and 18885.300 explicit gross; decimal display differences are not contradictions. ENOS seal prefix retained, tare excluded, no generic reefer clause imported."
        )
    if index == 43:
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            None,
            "LOPALLETS is the OCR rendering of the ten-pallet quantity, confirmed by SAY TEN PALLETS ONLY, not a cargo mark. Empty marking captions have no values.",
        )
        notes.append(
            "Ten-pallet exact count resolves numeric hold. Missing PDF-only voyage remains absent. ZHL repeatedly signs as agent for carrier; no unambiguous carrier principal assigned."
        )
    if index == 44:
        set_value(
            ["parties", "carrier"],
            {"name": "MAERSK"},
            "Carrier brand is explicitly printed on the continuation; do not import an unprinted legal suffix or promote the Lanka agent.",
        )
        refs = patch["forwardingAndExportReferences"] + [
            "Egyptian Importer VAT Number: 484250515",
            "Exporter Registration Number: LK021141064457000",
        ]
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "Retain separately captioned VAT and exporter registration values, including the distinct printed registration formatting.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            [
                "Applicable free time 21 days Combined (Detention & Demurrage) at (port of discharge / place of delivery)"
            ],
            "Selected shipment free time is explicitly printed.",
        )
        notes.append(
            "1300 sacks, gross/net and volume all grounded; absent shipper/consignee and unresolved SAME AS ABOVE notify are not invented."
        )
    if index == 45:
        set_value(
            ["parties", "notifyParties", 0, "country"],
            "TAIWAN",
            "Select the printed postal country name; R.O.C and TW are aliases/codes, retained in the full address.",
        )
        set_value(
            ["parties", "shipper", "country"],
            "China",
            "Party country retains printed form, unlike route-country normalization.",
        )
        set_value(["parties", "consignee", "country"], "EGYPT", "Retain party country source form.")
        set_value(
            ["parties", "notifyParties", 1, "country"], "EGYPT", "Retain party country source form."
        )
        refs = patch["forwardingAndExportReferences"]
        refs = [r for r in refs if not r.startswith(("P/O No.", "Customer P/O NO.", "P/L No."))]
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "Ship To package-label PO and pallet-range values belong in marks; separately printed invoices remain references.",
        )
        goods = {
            "description": "ASSY OPEN CELL BN96-59789A GPC CODE:10001404 DESC: TELEVISIONS-VD PARTS",
            "grossWeight": {"value": 20035.974, "unit": "kilogram"},
            "volume": {"value": 110.25, "unit": "cubic_metre"},
            "marksAndNumbers": [
                "Ship To: SAMSUNG ELECTRONICS EGYPT SAE (SEEG)",
                "P/O No.: 3099898377",
                "Customer P/O NO. : 3099881035",
                "P/L No.: 1-12",
                "P/O No.: 3100009057",
                "Customer P/O NO. : 3100008964",
                "P/L No.: 1-4",
                "P/L No.: 1-18",
            ],
            "hsCodes": ["8524911000"],
            "numberAndTypeOfPackages": [{"packageQuantity": 42, "typeCategory": "PACKAGE_PALLET"}],
            "splitGoodsPlacement": [
                {"equipmentIdentifier": c["equipmentIdentifier"], "packageQuantity": n}
                for c, n in zip(patch["containerInformation"], [18, 16, 8], strict=True)
            ],
        }
        set_value(
            ["goodsItemDetails"],
            [goods],
            "Identical product/part across invoices is one shared goods identity. Restore all three printed pallet placements and exact complete sums; do not duplicate identical goods or mistake PCS product units for the package level.",
        )
    if index == 46:
        set_value(
            ["parties", "shipper", "country"],
            "TAIWAN",
            "Select the printed postal country name; preserve aliases in addressLine.",
        )
        set_value(
            ["parties", "notifyParties", 0, "country"],
            "TAIWAN",
            "Select the printed postal country name, not a list of aliases/codes.",
        )
        notes.append(
            "LED bar shipment: full printed 300 postcode and Taiwan aliases retained; uncaptioned continuation phone numbers included, explicitly marked FAX excluded elsewhere. Complete 17-pallet row and separate also-notify block verified."
        )
    if index == 47:
        set_value(
            ["freight", "paymentPlace"],
            None,
            "ORIGIN is a direction, not a named payment locality.",
        )
        notes.append(
            "Main sea freight explicitly prepaid despite destination ancillary charges collect. High-cube open-top fully specified; broad reefer/foodstuff tariff does not describe this spare-parts shipment."
        )
    if index == 48:
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 days detention free at destination"],
            "Shipment-specific detention allowance is printed.",
        )
        notes.append(
            "Complete HC40 equipment and both distinct seals retained. Empty mass/volume columns not populated; carrier company registration is not a shipment reference. Two explicitly identified agent contacts retained."
        )
    if index == 49:
        notes.append(
            "Primary vessel field, on-board stamp and first continuation all say CMA CGM CENTAURUS; isolated OMA rendering on third copy does not establish a different vessel. Full shipper starred address retained; cargo gross and tare kept distinct."
        )
    if index == 50:
        set_value(
            ["containerInformation"],
            None,
            "Only OCR equipment token is malformed ONEEU5008417 (five-letter prefix). Do not silently delete E or import the PDF-only corrected identifier. Preserve raw equipment/seal/type evidence in audit; no representable identified container or placement.",
        )
        notes.append(
            "All goods/package/mass facts remain extractable despite unrepresentable container identifier. Full destination-agent continuation, both dates and main-freight payment verified; no PDF-only values imported."
        )
    if index == 51:
        notes.append(
            "Separate source from 046 with 18 pallets, 3013 kg and 23.34 CBM; all document-specific identifiers verified, same party layout does not authorize copying cargo values. Repeated copies add no facts."
        )
    if index == 52:
        set_value(
            ["forwardingAndExportReferences"],
            [
                *patch["forwardingAndExportReferences"][:-1],
                "REG: 786925",
                patch["forwardingAndExportReferences"][-1],
            ],
            "Second notify commercial registration is a party-owned reference in the OCR continuation.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 DAYS FREE LINE DETENTION AT POD"],
            "Selected line-detention instruction is explicitly printed.",
        )
        notes.append(
            "Four identified 1000-bag rows reconcile to 4000, 100200 kg gross and 100000 kg net. Triple-star delivery-agent contact continuation is not assigned to notify merely because both use three stars; company/email/layout ownership distinguishes it. Printed malformed extra phone digits are not guessed/repaired."
        )
    if index == 53:
        for path in [
            ["placeOfIssue", "country"],
            ["route", "placeOfReceipt", "country"],
            ["route", "portOfLoading", "country"],
        ]:
            set_value(
                path,
                "SOUTH KOREA",
                "Normalize printed KOREA consistently for Seoul/Busan location phrases, not party source-form country.",
            )
        for j in range(3):
            set_value(
                ["containerInformation", j, "typeDescription"],
                None,
                "DC 20 establishes complete standard dry equipment.",
            )
            set_value(
                ["containerInformation", j, "sizeCategory"],
                "TWENTY_FOOT_STANDARD_HEIGHT",
                "Printed complete dry equipment.",
            )
            set_value(
                ["containerInformation", j, "typeCategory"], "GENERAL_PURPOSE", "Printed DC type."
            )
        set_value(
            ["parties", "deliveryAgent"],
            {
                "name": "HMM (NETHERLANDS) SHIPPING B.V.",
                "addressLine": "WESTBLAAK 180, 3012 KN ROTTERDAM, NETHERLANDS",
                "country": "NETHERLANDS",
                "contactDetails": {"phoneNumbers": ["+31-10-280-2555"]},
            },
            "SHIPPING AGENT DETAILS continuation explicitly supplies the destination agent.",
        )
        notes.append(
            "60 inner bags on pallets; no equal allocations inferred. Main ocean vessel is GARAM, not pre-carriage RAON. EBL transfer history is not bill metadata."
        )
    if index == 54:
        notes.append(
            "Net19465.20 and gross19755.200 are explicit amounts; decimal token-screen hold is false positive. Both full seals including ENOS prefix retained; source seven-digit HS preserved without padding, and dry-container tariff does not create temperature/DG facts."
        )
    if index == 55:
        set_value(
            ["parties", "shipper", "addressLine"],
            "ARNISSA PELLAS",
            "TEL caption is joined to PELLAS in OCR; strip the caption, not postal content.",
        )
        set_value(
            ["parties", "notifyParties"],
            [{"sameAs": "consignee"}],
            "SAME AS CNN is the explicit notify reference to consignee, confirmed by form layout.",
        )
        set_value(
            ["negotiability"],
            "non_negotiable",
            "PDF confirms Consigned to order of is the stock caption, whereas the actual filled party has no order instruction and the bill is explicitly non-negotiable.",
        )
        set_value(
            ["freight"],
            {"paymentArrangement": "prepaid"},
            "PDF shows PREPAID as the filled payment value; As agreed payable at destination is stock small-print form text, not a second selected arrangement.",
        )
        notes.append(
            "1328 inner boxes on 21 pallets; OCR lacks PDF seal, gross weight and carrier logo, so these remain absent. No temperature inferred from apples or reefer type."
        )
    if index == 56:
        for path in [
            ["parties", "consignee", "addressLine"],
            ["parties", "notifyParties", 0, "addressLine"],
        ]:
            set_value(
                path,
                "Port Tawfik Free Zone Area, Suez, Egypt",
                "PDF confirms primary postal address directly beneath company. Alexandria55698 follows the named email contact as an alternative contact-locality line; preserve in audit, not concatenate two cities into one postal address.",
            )
        set_value(
            ["freight", "paymentPlace"],
            {"name": "PIRAEUS"},
            "PDF aligns OCR PIRAEUS with freight-to-be-paid-at field; not a copied loading place.",
        )
        notes.append(
            "Eight explicit 24-IBC rows yield192 plus separately printed104 packages, not a 192-shipment total. Nine seals verified across page break. Equipment sizes and CBM unit occur only in PDF and are excluded; kg unit exists in OCR total."
        )
    if index == 57:
        for path in [
            ["parties", "consignee", "addressLine"],
            ["parties", "notifyParties", 0, "addressLine"],
        ]:
            set_value(
                path,
                "MENYAT SAMNOD, AGA CITY, DAQHLIA, EGYPT",
                "Comma-separate postal components while preserving wording/order.",
            )
        set_value(
            ["parties", "notifyParties", 1, "addressLine"],
            "P.O BOX 120861, SHARJAH, U.A.E",
            "Separate locality and postal country components.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 DAYS FREE TIME AT THE DESTINATION"],
            "Selected destination allowance is OCR-grounded.",
        )
        notes.append(
            "PDF establishes second OCR DAMIETTA as delivery, not adjacent transshipment caption. Only one date occurrence is OCR-present at issue block; PDF-only on-board stamp, masses, volume and forwarding-agent occurrence excluded."
        )
    if index == 58:
        notes.append(
            "Three tractors, no containers. Malformed KOS unit does not license kg; no repair of printed unit or inferred cargo mass. Primary postal addresses retained, banking/registration metadata excluded, deck carriage is selected handling."
        )
    if index == 59:
        notes.append(
            "5420+1923=7343 packages and20945.95+25180.85=46126.80kg net; distinct container portions preserved under shared product, without duplicate aggregate package row. Full ENOS seals, source typo INTERANTIONAL and seven-digit address number retained."
        )
    if index == 60:
        set_value(
            ["forwardingAndExportReferences"],
            patch["forwardingAndExportReferences"]
            + [
                "Egyptian VAT number: 511481217",
                "Exporter Registration Number:811124779",
                "Customer Code 11311666",
            ],
            "Restore explicit VAT/exporter/customer reference roles, distinct from commodity/mark identifiers.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            [
                "Applicable free time 10 days Combined (Detention & Demurrage) at (port of discharge / place of delivery)"
            ],
            "Join explicit continuation across page break, excluding the intervening freight-table boilerplate.",
        )
        notes.append(
            "2383 cartons plus633 pieces are parallel packages, not nesting.3016 single-container quantity equals exact sum. European net decimal4543.501 retained, and main freight prepaid does not conflict with ancillary collect fees."
        )
    if index == 61:
        set_value(
            ["goodsItemDetails", 0, "hsCodes"],
            ["7220209000", "7210499090", "7212309090", "7209159000"],
            "Four HS codes are explicitly printed with presentation dots/hyphens; preserve all ten digits in each. The reviewer incorrectly removed this entire supported set.",
        )
        notes.append(
            "Jointly accounted steel assortment has24 pallets and exact14/10 placements; all net/gross and volume totals reconcile. Uncaptioned C C agency contact block has no established destination-agent role even in PDF, so retained in audit only rather than guessed. Missing issue place remains absent."
        )
    if index == 62:
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            [
                "Goods shipped in refrigerated container(s) set at shipper's requested carriage temperature as per above"
            ],
            "Preserve printed handling wording; do not synthesize a new sentence by substituting the structured temperature.",
        )
        notes.append(
            "Full shipper rural/postcode address and both seal declarations retained. Proper shipment net/gross and -18C consistent; bare51486450-6 and signatory CPF not promoted to commercial references."
        )
    if index == 63:
        set_value(
            ["route", "portOfDischarge", "name"],
            "ALEXANDRIA OLD PORT",
            "Preserve distinguishing OLD PORT locality; ACCHCO is terminal/operator context.",
        )
        set_value(
            ["parties", "notifyParties", 1, "addressLine"],
            "Abd El Khalek Tharwat St., Cairo, Egypt",
            "EG and Egypt are adjacent duplicate country components; retain printed full country once.",
        )
        set_value(
            ["parties", "notifyParties", 1, "country"],
            "Egypt",
            "Postal country is Egypt, not combined country code/name string.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            [
                *patch["forwardingAndExportReferences"][:-1],
                "Egyptian Importer Tax ID: 100296432",
                "Foreign Exporter ID: 311802128",
                patch["forwardingAndExportReferences"][-1],
            ],
            "Restore explicitly owned customs/exporter reference declarations, not the uncaptioned X token.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["Demurrage Free Tm:21 Calendar Days"],
            "Selected21-day allowance differs from the generic equipment tariff.",
        )
        notes.append(
            "PDF resolves shifted OCR captions: receipt empty, loading NEWYORK, transshipment ALIAGA, discharge ALEXANDRIA OLD PORT. Seven unique container portions total131 packages and25863.808kg. Real printed00000 postcode remains source text, not synthetic placeholder cleanup."
        )
    if index == 64:
        set_value(
            ["parties", "shipper", "addressLine"],
            patch["parties"]["shipper"]["addressLine"].replace("PK 34843", "34843"),
            "PK is postal caption; retain all postcode digits.",
        )
        issuer = copy.deepcopy(patch["parties"]["carrier"])
        set_value(
            ["parties", "forwardingAgent"],
            issuer,
            "PDF signature establishes letterhead issuer Merden as forwarder only; all name/address/contact values already appear in OCR.",
        )
        set_value(
            ["parties", "carrier"],
            {"name": "MLH"},
            "Explicit CARRIER: MLH identifies the carrier, distinct from the forwarding issuer.",
        )
        set_value(
            ["route", "portOfDischarge", "country"],
            "EGYPT",
            "PDF shows (EGYPT) overflowing from the discharge value, not a separately selected delivery country.",
        )
        set_value(
            ["route", "placeOfDelivery"],
            None,
            "Delivery box is empty; OCR reading order misplaced the end of discharge phrase.",
        )
        notes.append(
            "Four260-package rows plus71 sum1111 and9660kg; slash-form container IDs preserve every character. On-board date not copied to undated issue field."
        )
    if index == 65:
        set_value(
            ["transport", "vesselName"],
            "JIAN GUO HAI",
            "MV is vessel-class caption, not part of the vessel identity.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["STOWED INTO HOLDS NUMBER: 1,2,3,4,5"],
            "Selected hold placement is a shipment-specific carriage instruction, not a container allocation.",
        )
        notes.append(
            "Bulk cargo without package count; unqualified said-to-weigh quantity does not establish gross/net. TO ORDER implies negotiable, no consignee identity invented. National adjective supports destination country but not an invented named port."
        )
    if index == 66:
        set_value(
            ["parties", "notifyParties", 0, "country"],
            "TAIWAN",
            "Country name is printed; adjacent aliases/codes remain in full address.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            [
                r
                for r in patch["forwardingAndExportReferences"]
                if not r.startswith(("P/O No.", "Customer P/O NO.", "P/L No."))
            ],
            "Package-label references are marks, whereas invoice/customs declarations remain shipment references.",
        )
        marks = [
            [
                "Ship To: SAMSUNG ELECTRONICS EGYPT SAE (SEEG)",
                "P/O No.: 3095353250",
                "Customer P/O NO. : 3095383527",
                "P/L No.: 1-2",
            ],
            [
                "Ship To: SAMSUNG ELECTRONICS EGYPT SAE (SEEG)",
                "P/O No.: 3093242104",
                "Customer P/O NO. : 3093240839",
                "P/L No.: 1-2",
                "P/O No.: 3096129719",
                "Customer P/O NO. : 3096129540",
                "P/L No.: 1-6",
            ],
            [
                "Ship To: SAMSUNG ELECTRONICS EGYPT SAE (SEEG)",
                "P/O No.: 3095967610",
                "Customer P/O NO. : 3095965784",
                "P/L No.: 1-2",
            ],
        ]
        for j, g in enumerate(patch["goodsItemDetails"]):
            set_value(
                ["goodsItemDetails", j, "description"],
                g["description"].replace("; TELEVISIONS", " GPC CODE:10001404 DESC: TELEVISIONS"),
                "Keep shared product classification and source wording without invented semicolon.",
            )
            set_value(
                ["goodsItemDetails", j, "marksAndNumbers"],
                marks[j],
                "Product-linked package-label block gives exact ownership of shipping marks.",
            )
        notes.append(
            "Three distinct part numbers remain separate; two portions of59390A share one identity. All belong to sole container. Product PCS counts and package-number ranges are not silently converted to pallet quantities;12-pallet total and5094.766kg/23.232CBM have no product-specific allocation and remain audit metadata."
        )
    if index == 67:
        set_value(
            ["route", "finalDestination"],
            None,
            "Uncaptioned PORT SAID FREE ZONE in cargo continuation does not establish a separate final-destination role.",
        )
        set_value(
            ["parties", "carrier"],
            {"name": "EVERGREEN LINE"},
            "Explicit carrier service brand printed; do not invent an unprinted underlying legal company.",
        )
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 FREE CALENDAR DAYS OF CONTAINER USAGE"],
            "Selected equipment free-time statement is shipment-specific.",
        )
        notes.append(
            "Three15-package rows and27830+28220+27100kg reconcile. Both seal strings per container retained including3RDSPC prefix; two numbered notify contacts continue across pages. Bare40H remains raw without guessing complete equipment type."
        )
    if index == 68:
        set_value(
            ["forwardingAndExportReferences"],
            [
                patch["forwardingAndExportReferences"][0],
                "COMMERICAL REGISTER 200227475",
                *patch["forwardingAndExportReferences"][1:],
            ],
            "Explicit consignee/notify commercial registration is a reference, not postal text; source spelling preserved.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            None,
            "HROB2297 and WKESD000000844299 identify the carrying trailer/vehicle;021260 is paired with an explicitly named E-SEAL. They are not package marks for the harness parts, and cannot be forced into ISO-container schema.",
        )
        notes.append(
            "69CLI retains unknown source package type; unitless12416.10 not assigned kg. Shipper agency identity retained as printed; tax number removed from address only because exact same value has explicit VAT role."
        )
    if index == 69:
        set_value(
            ["freight", "paymentArrangement"],
            None,
            "Named payment place alone is not an explicit payable-elsewhere arrangement.",
        )
        for j, g in enumerate(patch["goodsItemDetails"]):
            set_value(
                ["goodsItemDetails", j, "description"],
                g["description"].split(" VEHICLES ARE LOADED")[0],
                "Generic multi-vehicle damage/liability clause is not an individually selected product description; keep each trailer and exact chassis number.",
            )
        notes.append(
            "Two individually quantified vehicles remain separate. Ambiguous05/03 date and PDF-only vessel name not inferred from UK issuer or voyage code."
        )
    if index == 70:
        set_value(
            ["freight", "paymentArrangement"],
            None,
            "Named payment locality alone does not specify arrangement.",
        )
        set_value(
            ["forwardingAndExportReferences"],
            None,
            "O/NO occurs exclusively in package marks and is already retained there.",
        )
        set_value(
            ["containerInformation", 2, "typeDescription"],
            None,
            "20ST explicitly defines standard dry container.",
        )
        set_value(
            ["containerInformation", 2, "sizeCategory"],
            "TWENTY_FOOT_STANDARD_HEIGHT",
            "Printed complete20ST.",
        )
        set_value(
            ["containerInformation", 2, "typeCategory"],
            "GENERAL_PURPOSE",
            "Printed standard dry equipment.",
        )
        notes.append(
            "15+8+6=29 packages; missing goods description and measurement unit remain absent. Cargo15990kg is distinct from9810tare; final seal correctly continues onto page2. Bank named after TO THE ORDER OF retained as consignee."
        )
    if index == 71:
        set_value(
            ["freight"],
            None,
            "PDF confirms Prepaid and Collect are empty form captions; OCR uppercase PREPAID is not a selected arrangement. PDF-only FREIGHT AS ARRANGED does not add a value.",
        )
        set_value(
            ["parties", "shipper", "addressLine"],
            patch["parties"]["shipper"]["addressLine"].removesuffix(", CHINA"),
            "SHIPPER COUNTRY is customs metadata, not an owned postal continuation.",
        )
        set_value(
            ["parties", "shipper", "country"],
            None,
            "Do not enrich postal country from registration metadata or loading locality.",
        )
        notes.append(
            "Two screw types jointly accounted in2200 bags with exact1100/1100 placements; multiple HS do not force goods splitting. DistinctSOKHNA occurrences support discharge and delivery."
        )
    if index == 72:
        notes.append(
            "Four explicit container rows total3427 cartons,39352kg and272CBM. Full postal and malformed printed email retained without speculative character repair. No missing B/L/carrier inferred from PDF or registry."
        )
    if index == 73:
        notes.append(
            "260 bales with100kg capacity;26000kg explicitly printed independently of multiplication. Distinct TOLIARA II and TOLIARA retained, repeated MADAGASCAR once withpostcode601. Blank dates and notify remain absent; uncaptioned invoice-party numbers not guessed as references."
        )
    if index == 74:
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 days detention free at destination"],
            "Restore selected destination detention allowance.",
        )
        notes.append(
            "1260 packages, bothfullseal values and completeHC40. Missing vessel and masses not copied from adjacent similar ARL documents; company registry is not a shipment reference."
        )
    if index == 75:
        notes.append(
            "Sparse OCR genuinely lacks parties, route/cargo and equipment. Only bill number, issue date/place and carrier are supported; do not force missing arrays or infer cargo from PDF."
        )
    if index == 76:
        notes.append(
            "One five-piece used-forklift shipment; named VIA intermediary remains part of printed shipper identity with one address. FullpostcodeEC1V98D and malformed doubleplus phone retained. On-board date not copied into blank issue date."
        )
    if index == 77:
        set_value(
            ["freight", "paymentPlace"],
            None,
            "ORIGIN is a direction, not a named payment locality.",
        )
        notes.append(
            "Main sea freight prepaid despite ancillary collect. No goods description/package count/size in OCR; generic all-equipment tariff cannot fill missing cargo or equipment facts. Names, personal identity, postcode and both party contacts verified."
        )
    if index == 78:
        set_value(
            ["containerInformation"],
            patch["containerInformation"][:1],
            "Second draft identifier TRKJ4463298 is neither valid container syntax nor OCR text; raw TrkuU4463298 is itself malformed. Retain first valid container; preserve second raw identifier/seal in audit without guessing a letter deletion.",
        )
        set_value(
            ["goodsItemDetails", 0, "splitGoodsPlacement"],
            [{"equipmentIdentifier": "BSIU9478390", "packageQuantity": 6}],
            "First container row explicitly owns the six-piece CAT980G item.",
        )
        set_value(
            ["goodsItemDetails", 1, "description"],
            "Dismantled CAT 980H SN CAT0980HHJMS01605",
            "Source serial continuation on page2 completes the second product description.",
        )
        notes.append(
            "Three separately accounted cargo rows6/5/36 and20727/20727/20113 sum47/61567. No third container invented from claimed container total; invalid second container cannot carry a target foreign key. Source-grounded issue date retained, missing on-board date excluded."
        )
    if index == 79:
        refs = patch["forwardingAndExportReferences"]
        refs.insert(3, "Company registration. 54098")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "Explicit consignee company registration belongs in references, outside postal address.",
        )
        set_value(
            ["goodsItemDetails", 0, "description"],
            patch["goodsItemDetails"][0]["description"].replace(";", ""),
            "Join product lines without adding non-source semicolon; preserve exact truck/serial identity.",
        )
        notes.append(
            "Full party postal content retained with duplicated BADRASHIN/EGYPT only once and0001 preserved. Single34700kg/125.775CBMtruck; generic damage boilerplate excluded, absentcarrier/dates not invented."
        )
    if index == 80:
        set_value(
            ["transport", "vesselName"],
            "MSC PRELUDE",
            "V is the voyage caption before OM404R, not part of the vessel name.",
        )
        notes.append(
            "46cu.ft is an unsupported schema volume unit; do not relabel asCBM or convert without policy. Five-degree setpoint and15CBMventilation are distinct. Multi-port Saudi overland transit remains exact handling text, not a guessed single transshipment port. Generic reefer detention tariff excluded."
        )
    if index == 81:
        set_value(
            ["parties", "shipper", "contactDetails", "phoneNumbers"],
            ["(07480) 233524, 25, 26, 27"],
            "Retain printed abbreviated phone range together; isolated25/26/27 are suffixes, not standalone phone numbers.",
        )
        refs = patch["forwardingAndExportReferences"]
        refs.remove("SERVICE CONTRACT NO.: PNQN00218A")
        set_value(
            ["forwardingAndExportReferences"],
            refs,
            "PNQN00218A has no OCR reference caption; PDF-only caption does not create an explicit shipment reference.",
        )
        notes.append(
            "32woodencrates,16percontainer;38.153/37.273MTexplicit, not converted tokg. HS854511.00 preservesall8digits. BRAND HEG INDIA gives product-origin wording, not customs-country inference. FullFW continuation and fax-only exclusions verified."
        )
    if index == 82:
        set_value(
            ["route", "placeOfDelivery"],
            None,
            "Second PORT SAID delivery occurrence is PDF-only; one OCR occurrence supports discharge, not both roles.",
        )
        set_value(
            ["parties", "notifyParties", 0, "country"],
            "TAIWAN",
            "Retain printed country name, not combined country aliases/codes.",
        )
        set_value(
            ["goodsItemDetails", 0, "description"],
            "SIGNAL CABLE",
            "PDF separates SEEG in marks column from product wording; both tokens already in OCR.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            ["SEEG", "MADE IN CHINA"],
            "Restore actual package markings; blank PO/CNO captions supply no mark.",
        )
        notes.append(
            "One pallet in OCR; thirteen inner cartons,2420PCS and mass/volume units occur only in PDF and are excluded. Do not replace supported one-pallet target with PDF-only inner count. Country origin declaration is printed."
        )
    if index == 83:
        set_value(
            ["goodsItemDetails", 0, "origin"],
            None,
            "FOREIGN COUNTRY/COUNTRY CODE accompany exporter registration, not a goods manufacture/origin declaration.",
        )
        notes.append(
            "PDF confirms two different payment localities in the same field: neither is selected as a singular paymentPlace. FREIGHT COLLECT is separately printed and retained. Missing consignee and unresolved SAME AS CONSIGNEE remain absent rather than importing PDF identity.734 packages/28350kg/65CBM exact."
        )
    if index == 84:
        notes.append(
            "Six separately quantified steel grades retained, with exact19-coil/391.005gross/390.910net MT totals. PDF confirms ST.PETERSBURG/date are issuance, not payment. Some coils rusted and some bands missing cannot be assigned to particular grades; no target field for unallocated subset condition. Open-area/rain history is not an instruction. These remarks remain unchanged in OCR and documented here; no grade is declared universally rusty. Missing shipper and on-board date not imported from PDF."
        )
    if index == 85:
        set_value(
            ["transport", "vesselName"],
            "MSC MADELEINE",
            "V. is a voyage caption, not a vessel-name suffix.",
        )
        for path in (["parties", "consignee"], ["parties", "notifyParties", 0]):
            set_value(
                [*path, "addressLine"],
                "Public Free Zone, Ameriya - Alexandria-Egypt",
                "Operating Within The Freezone Authority describes company operating status, not a postal component.",
            )
        set_value(
            ["goodsItemDetails", 0, "description"],
            "VERSATROL M 50 LB NEW MULTIWALL PAPER BAGS BATCH NOS. 23077",
            "Product capacity and lot identity belong in description even when also printed on package labels; no invented semicolon.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            None,
            "All apparent marks are product name, package capacity and batch qualifier; move product wording to description rather than duplicate it.",
        )
        notes.append(
            "2100 paper bags are inner packages on70pallets; one of two container IDs absent from OCR.35-pallet portions do not establish bag allocations. Explicit50155/47670kg and80CBM totals retained, ambiguous individual CBM reading not used. Represented OIES buyer is not substituted for primary shipper. PDF-only delivery agent/IMO excluded."
        )
    if index == 86:
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            [
                "Applicable free time 21 days Combined (Detention & Demurrage) at (port of discharge / place of delivery)"
            ],
            "Restore selected shipment-specific free-time allowance.",
        )
        notes.append(
            "Yarn description,458cartons,24342.7gross/23083.2net, full source addresses and destination-agent continuation verified. Third-party SONVIGO exporter not substituted for primary VARDHMAN shipper. Manufactured INDIA distinct from exporter registration MAURITIUS."
        )
    if index == 87:
        notes.append(
            "PDF confirms printed KGS rows occupy gross-weight column and HAMINA/date are issue field. All eight portions sum524packages,2196.199CBM,1207909kg.502-package on-deck subset is handling, not extra cargo. TRANSBAY,loading location,carrier and B/L number are absent OCR despite PDF presence; do not import. Charter-party reference does not select freight arrangement."
        )
    if index == 88:
        set_value(
            ["goodsItemDetails", 0, "handlingInstructions"],
            ["21 days detention free at destination"],
            "Restore explicit selected destination allowance.",
        )
        notes.append(
            "2377 packages jointly describe tyres/tubes; multiple HS do not force split. No container identifier printed in OCR, so no equipment object invented from40HC phrase alone. Both delivery contacts retained; alphanumeric ACID typo preserved, not replaced by digit guess."
        )
    if index == 89:
        for path in (["placeOfIssue"], ["route", "placeOfReceipt"], ["route", "portOfLoading"]):
            set_value(
                [*path, "country"],
                "SOUTH KOREA",
                "Normalize printed KOREA in BUSAN/SEOUL context to conventional English country name.",
            )
        notes.append(
            "143 jointly accounted pallets; all four product codes retained across shifted OCR heading. Three container memberships without invented equal quantities. EL TERIAK mark is absent OCR, not added from party name/PDF. Explicit agent-for-carrier signature retains FIRST UNION, not DAESUNG as principal."
        )
    if index == 90:
        set_value(
            ["freight"],
            None,
            "Freight As Arranged does not select prepaid/collect/third-party/other payment place; empty table captions are not selections.",
        )
        set_value(
            ["goodsItemDetails", 0, "numberAndTypeOfPackages"],
            [{"packageQuantity": 70, "typeOfPackages": "SETS"}],
            "Restore explicit70SETS cargo packages; preserve source type without inventing a taxonomy code.",
        )
        set_value(
            ["goodsItemDetails", 0, "splitGoodsPlacement", 0, "packageQuantity"],
            70,
            "Sole container explicitly has70SETS; restore numeric relation.",
        )
        notes.append(
            "Joint gearbox/motor shipment;19000kg/22.5CBM, vessel continuation and complete notify/delivery addresses verified. Shipper ID text is not a postal address. Generic demurrage tariff excluded."
        )
    if index == 91:
        refs = patch["forwardingAndExportReferences"]
        set_value(
            ["forwardingAndExportReferences"],
            [
                "REGISTRATION NO:91340104MA2NKH314U",
                *refs[:2],
                "IMPORTER TAX NO.:716943689",
                *refs[2:],
            ],
            "Restore explicit registration and importer declarations in source order; distinct reference captions retained.",
        )
        notes.append(
            "OCR22863 500KGS has decimal punctuation replaced by whitespace; PDF confirms22863.500 with exactly the same digits and physical role. This is numeric normalization, not importing a missing mass.1175cartons/61.910CBM/HS841490; PDF-only B/L number excluded. Consignee MOHAMED SADEK KARAT is street wording, not an uncaptioned personal identity above postal block."
        )
    if index == 92:
        notes.append(
            "Full named-company plus uncaptioned personal identity preserved; printed EGYI remains in postal address but not guessed as a recognized country code. Ambiguous05/08/2024 date omitted. One loader, source serial/dimension versions retained; no manufactured mass or containers. Fax-only number excluded."
        )
    if index == 93:
        set_value(
            ["forwardingAndExportReferences"],
            [
                patch["forwardingAndExportReferences"][0],
                "TRADE REGISTERED NO : ISTANBUL 143420/90928",
                *patch["forwardingAndExportReferences"][1:],
            ],
            "Restore explicit trade registry reference, outside postal address.",
        )
        notes.append(
            "Thirteen separately quantified products; inner drums with exact1/3/24/17/16/6/1/11/17/17/3/27/34 allocations. Eight outer pallets are not double-counted. Unlabelled product KG amounts cannot be assumed net; PDF-only total mass/volume excluded. Unstackable applies to printed palletized rows, not an invented package total. PDF confirms20-Mar on-board, not issue; owned Egypt continuation after contact retained."
        )
    if index == 94:
        set_value(
            ["goodsItemDetails", 0, "description"],
            "CHASSIS NOS : WDB96340310295283 USED UNPACKED VEHICLE (S) MERCEDES BENZ TRUCK",
            "Vehicle chassis is product identity; generic multi-vehicle damage disclaimer is not selected condition. Keep product words in source order.",
        )
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            None,
            "Only apparent mark is the product chassis identifier, now in description.",
        )
        notes.append(
            "One unpacked vehicle,7500kg/61.908CBM. Shipper duplicate6280 retained once; printed country IT is not geocorrected from city. ECG0924 is voyage-shaped, not invented vessel name; no unseen carrier/B/L/date imported."
        )
    if index == 95:
        notes.append(
            "Two-pallet sole-container liquid-filter shipment. First printed mass568Lbs retained; printed2.989CBM used rather than converting105.556Cbf. ExplicitgoodsU.S.A.origin not copied to shipper postal country.10779168 occurs independently as a cargo mark and B/L/booking. Uncaptioned2305063006 not guessed as reference."
        )
    if index == 96:
        notes.append(
            "Duplicate waybill copies counted once:40bundles,20eachcontainer and42470.4kg sum.1680x25KG has no named inner package type and remains source product packing wording, not a guessedbag label or computednet mass. Full postal including POBOX37 preserved; no missing HS/volume/destination invented."
        )
    if index == 97:
        notes.append(
            "Four valid OCR container/seal pairs; all cargo counts/weights and issue date are PDF-only and excluded. S2300245857 is bill number; uncategorizedSDR23TREG008659 not promoted to master bill/reference. Named bank after TO THE ORDER retained. EXW locality is not carrier postal address or receipt. Product destination suffix EGYPT excluded from identity; no guessed goods country or quantities."
        )
    if index == 98:
        set_value(
            ["containerInformation", 1, "typeDescription"],
            None,
            "DC20 explicitly states20-foot dry container.",
        )
        set_value(
            ["containerInformation", 1, "sizeCategory"],
            "TWENTY_FOOT_STANDARD_HEIGHT",
            "Complete DC20 equipment phrase.",
        )
        set_value(
            ["containerInformation", 1, "typeCategory"],
            "GENERAL_PURPOSE",
            "Complete dry equipment phrase.",
        )
        for j in range(2):
            set_value(
                ["goodsItemDetails", j, "handlingInstructions"],
                ["21 days detention free days at destination"],
                "Selected destination allowance applies to both goods portions.",
            )
        notes.append(
            "Two separately accounted goods888/1212cartons with68/28CBM and exactcontainer placements. Total35870kg cannot be divided by container or cargo without evidence, so remains unallocated audit fact. Booking/service ID not recast as unprinted B/L number; shipper country not inferred from name/registration."
        )
    if index == 99:
        set_value(
            ["goodsItemDetails", 0, "description"],
            "Steel Casing.",
            "Five racks per container is package information, not a product variant or a five-rack description repeated as an aggregate.",
        )
        set_value(
            ["goodsItemDetails", 0, "numberAndTypeOfPackages"],
            [{"packageQuantity": 20, "typeOfPackages": "Racks"}],
            "Twenty explicitly named racks across four equal five-rack portions; retain precise printed package type.",
        )
        for j in range(4):
            set_value(
                ["containerInformation", j, "typeDescription"],
                None,
                "Complete40-foot high-cube dry equipment phrase supports canonical categories.",
            )
            set_value(
                ["containerInformation", j, "sizeCategory"],
                "FORTY_FOOT_HIGH_CUBE",
                "Explicit40foot/highcube.",
            )
            set_value(
                ["containerInformation", j, "typeCategory"],
                "GENERAL_PURPOSE",
                "Complete high-cube equipment phrase, no specialized type.",
            )
        notes.append(
            "Four gross masses reconcile86471kg; same steel casing is one goods identity. Ship-from/principal ON BEHALF OF identity preserved intact. Owned Egypt after TaxID belongs to postal continuation, not tax number. PDF-only contact numbers/email excluded; both delivery-agent postal and leading-zero seals retained."
        )
    if index == 100:
        set_value(
            ["parties", "deliveryAgent", "name"],
            "MSC PORT SAID Mediterranean Shipping Co. (Misr Maritime Agency)",
            "Branch identity belongs with company name, not postal address.",
        )
        set_value(
            ["parties", "deliveryAgent", "addressLine"],
            "Abou Elkhyer Building, First floor, 3 23rd July & Kaetbay st",
            "Remove company/branch identity from postal target while preserving every printed address component.",
        )
        notes.append(
            "3270+2377=5647 packages,41957.85gross/40928.95net explicittotal; unitlesspartialnet14650.6 does not override them. Both carrier and ENOS seals preserved. No website printed for carrier; email domain is not a website. Discharge country and notify locality cannot be copied from consignee. Generic equipment tariff excluded."
        )
    if index == 20:
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            [
                "SEEG((SEEG)VD PLANT",
                "SEEG:3107355054",
                "SET:3107359679",
                "P/NO.:B82J1/4-4/4",
                "Made in Taiwan",
                "802070123",
                "SEEG((SEEG))VD PLANT)",
                "SEEG:3107055660/3107564174",
                "SET:3107057001/3107563390",
                "P/NO.:B82J1/32-32/32",
                "802070437",
                "SEEG((SEEG)VD PLANT)",
                "SEEG:3107055660",
                "SET:3107057001",
                "P/NO.:B82J1/52-52/52",
                "802069091",
            ],
            "Final source-order check: preserve every distinct printed mark variant and package-number range in its first cargo occurrence order.",
        )
    if index == 41:
        set_value(
            ["goodsItemDetails", 0, "marksAndNumbers"],
            [
                "SSAMSUNG ELECTRONICS EGYPT S.A.E.",
                "INV.NO:MSZ2515446",
                "ORD.NO:202512313491",
                "PO.NO:3106412368",
                "PALLET NO:1-3",
                "SAMSUNG ELECTRONICS EGYPT S.A.E.",
                "INV.NO:MSZ2515447",
                "PALLET NO:1-13",
                "INV.NO:MSZ2515448",
            ],
            "Final source-order check uses the marks block, not earlier occurrence of the same company as consignee; retain both actual source spelling variants.",
        )
    assert notes, f"Document {index} has no manual adjudication"
    return {**target, "documentPatch": patch}, notes
