"""Compact manual inspection view; does not classify, accept or modify labels."""
import argparse
import json
import re
from pathlib import Path
from review_real_v7_batch import ROOT, candidate, read

LEGAL = re.compile(r"^(?:\d+[.][ )]*|RECEIVED\b|SHIPPED,|The (?:Merchant|Carrier)|All claims|In WITNESS|In witness|IN WITNESS|Where the|If the|Demurrage and detention|Furthermore in case|Notwithstanding|DELIVERY will|This Waybill|This shipment is subject|This contract|According to|In accepting|IN ACCEPTING)", re.I)


def show(batch, numbers, target_only=False, source_only=False):
    decisions_path = batch / "manual-review/decisions.json"
    decisions = read(decisions_path) if decisions_path.exists() else {}
    for row in read(batch / "selection.json")["documents"]:
        if row["index"] not in numbers:
            continue
        print("\nDOCUMENT", row["name"], row["split"])
        if not target_only:
            for n,line in enumerate((ROOT / row["input"] / "ocr.txt").read_text().splitlines(),1):
                if not line.strip(): continue
                if len(line)>300 and LEGAL.match(line):
                    web = re.findall(r"(?:https?://|www\.)[^\s]+",line)
                    print(n,"[LONG LEGAL PARAGRAPH]",line[:95],"...",line[-60:],web)
                else: print(n,line)
        if source_only:
            continue
        if (batch / "runs" / row["name"] / "refine/status.json").exists():
            path,target,state=candidate(batch,row)
            if row["name"] in decisions:
                from publish_real_v7_batch2 import apply_decisions
                target=apply_decisions(target,decisions[row["name"]]["changes"])
            print("TARGET",json.dumps(target["documentPatch"],ensure_ascii=False))
            print("FINDINGS", json.dumps({k:v for k,v in state["reviews"].items() if v["findings"]},ensure_ascii=False))
        else: print("FINAL TARGET PENDING")

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("numbers",type=int,nargs="+");p.add_argument("--target-only",action="store_true")
    p.add_argument("--source-only",action="store_true")
    p.add_argument("--batch",type=Path,required=True)
    a=p.parse_args();show(ROOT / a.batch,a.numbers,a.target_only,a.source_only)
