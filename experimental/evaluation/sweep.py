"""Coverage probes are generated intents, not observed requests or relevance labels."""
from collections import defaultdict
import copy
import random
import re


STRATA = ("head", "middle", "tail", "no_history", "image_only", "mixed_history",
          "tag_only", "embedding_only", "partial")
DEFAULT_POLICY = {"target":128,"maximum":512,"batch":32,"patience":2,"minimum_gain":0.001}


def contexts(snapshot, *, split, policy=None):
    from experimental.evaluation.evaluate import digest, inputs_at, item_key, require, seeded, timestamp
    policy = dict(DEFAULT_POLICY if policy is None else policy)
    require(set(policy) == set(DEFAULT_POLICY), "invalid sweep policy fields")
    require(all(type(policy[k]) is int and policy[k] > 0 for k in ("target","maximum","batch","patience")), "invalid sweep limits")
    require(policy["maximum"] >= policy["target"] >= 128 and 0 <= policy["minimum_gain"] <= 1, "invalid conditional sweep target")
    candidates = defaultdict(dict)
    diagnostic = snapshot["manifest"]["mode"] == "current_state"
    popularity_unknown = False
    bases = {}
    for c in sorted(snapshot["contexts"],key=lambda c:(timestamp(c["cutoff"]),c["id"])):
        if c["split"] != "purged" and (split == "all" or c["split"] == split):
            bases.setdefault(c["split"],c)
    for base in bases.values():
        probe = copy.deepcopy(base)
        probe["history_kinds"] = ["image","video"]
        probe["kinds"] = ["video","image"]
        if diagnostic:
            probe["kinds"] = list(snapshot["manifest"]["diagnostic_kinds"])
        probe["intent"] = {"seed":None,"tag_ids":[]}
        probe["exclude"] = []
        inputs, safe, counts, _ = inputs_at(snapshot,probe)
        allowed = {item_key(i) for i in safe["eligible_ids"]}
        catalog = {item_key(i):i for i in inputs["catalog"] if item_key(i) in allowed}
        features = {item_key(f):f for f in inputs["features"]}
        counts_complete = not diagnostic or all(i in counts for i in catalog)
        popularity_unknown |= not counts_complete
        levels = sorted({counts.get(i,0) for i in catalog}) if counts_complete else []
        history_kinds = {e["kind"] for e in inputs["evidence"] if e["type"] != "initial_state"
                         or e["value"].get("rating") is not None or e["value"].get("engagement_count", 0) > 0
                         or e["value"].get("watch", {}).get("watched_s", 0) > 0}
        for key,item in sorted(catalog.items()):
            tags = item["tag_ids"]
            vectors = features.get(key,{}).get("vectors",{})
            strata = ["no_history"]
            if len(levels) >= 3:
                position = levels.index(counts.get(key,0)) / (len(levels)-1)
                strata.append("head" if position >= 2/3 else "tail" if position <= 1/3 else "middle")
            if key[0] == "image" and "image" in history_kinds:
                strata.append("image_only")
            if history_kinds == {"video","image"}:
                strata.append("mixed_history")
            if tags and not vectors:
                strata.append("tag_only")
            if vectors and not tags:
                strata.append("embedding_only")
            if key not in features or not tags or not vectors:
                strata.append("partial")
            for stratum in strata:
                c = copy.deepcopy(probe)
                c.update(judgment=False,stratum=stratum)
                c["intent"] = {"seed":{"kind":key[0],"id":key[1]},"tag_ids":sorted(tags)}
                if stratum == "no_history":
                    c["history_kinds"] = []
                elif stratum == "image_only":
                    c["history_kinds"] = ["image"]
                    c["kinds"] = ["image"]
                identity = digest({k:v for k,v in c.items() if k != "id"})
                candidates[stratum][identity] = c
    pools = {}
    for stratum in STRATA:
        pool = sorted(candidates[stratum].items())
        random.Random(seeded(snapshot["manifest"]["seed"],stratum,"coverage_stratum")).shuffle(pool)
        pools[stratum] = pool
    ordered = []
    while len(ordered) < policy["maximum"] and any(pools.values()):
        for stratum in STRATA:
            if pools[stratum] and len(ordered) < policy["maximum"]:
                identity,c = pools[stratum].pop()
                c["id"] = "sweep:" + str(len(ordered)).zfill(6) + ":" + identity[:12]
                ordered.append(c)
    support = {s:len(candidates[s]) for s in STRATA}
    missing = [s for s,n in support.items() if not n]
    supported = not missing and len(ordered) >= policy["target"]
    target = policy["target"] if supported else min(policy["target"],len(ordered))
    return ordered, {"policy":policy,"support":support,"missing_strata":missing,
        "popularity_status":"unavailable_unknown_counts" if popularity_unknown else "measured",
        "conditional_target_supported":supported,"minimum_contexts":target,"available_contexts":len(ordered),
        "pool_hash":digest(ordered),"accuracy":"unmeasured_generated_intents",
        "popularity_bands":("distinct captured-play count terciles; ties never split by ID" if diagnostic else
                            "distinct training-play count terciles; ties never split by ID"),
        "context_origin":"planned_current_state" if diagnostic else "generated_from_frozen_observed_context"}


def stop_reason(rows, plan, *, paired=False):
    """Stop after minimum coverage, then patience consecutive low-gain batches."""
    policy = plan["policy"]
    n = len(rows)
    if n >= plan["available_contexts"]:
        return "budget_exhausted" if n >= policy["maximum"] else "catalog_context_pool_exhausted"
    if paired or n < plan["minimum_contexts"] or n % policy["batch"]:
        return None
    patience,batch = policy["patience"],policy["batch"]
    if n < patience * batch:
        return None
    for end in range(n-(patience-1)*batch,n+1,batch):
        before = {i for r in rows[:end-batch] for i in r["order"]}
        after = {i for r in rows[:end] for i in r["order"]}
        eligible = {i for r in rows[:end] for i in r["eligible"]}
        if not eligible or len(after-before)/len(eligible) > policy["minimum_gain"]:
            return None
    return "marginal_coverage_stable"


def reachability(inventory, eligible):
    """Only complete pre-budget route inventories establish structural exclusion.

    Scope is the declared frozen context, never a universal library claim. A
    finite list of recommendation outputs cannot satisfy this contract.
    """
    from experimental.evaluation.evaluate import item_key, require
    unavailable = {"status":"unmeasured","structurally_unreachable":None,
                   "reason":"complete_production_route_inventory_not_supplied"}
    if inventory is None:
        return unavailable
    require(type(inventory) is dict and inventory.get("scope") == "frozen_context", "invalid route inventory scope")
    names = inventory.get("supported_routes")
    require(type(names) is list and all(type(n) is str and n for n in names) and len(names) == len(set(names)), "invalid supported routes")
    routes = inventory.get("routes")
    require(type(routes) is dict and set(routes) == set(names), "route inventory omits supported routes")
    if inventory.get("complete") is not True or not all(r.get("complete") is True and r.get("stage") == "pre_budget" for r in routes.values()):
        return unavailable | {"reason":"incomplete_or_post_budget_route_inventory"}
    hard = {item_key(i) for i in inventory["eligible_ids"]}
    require(hard <= set(eligible), "route inventory violates shared eligibility")
    reached = set()
    for route in routes.values():
        require(type(route["enabled"]) is bool, "route enabled must be explicit")
        items = {item_key(i) for i in route["eligible_ids"]}
        require(items <= hard and (route["enabled"] or not items), "invalid route eligibility")
        reached.update(items)
    return {"status":"measured","scope":"frozen_context","eligible":len(hard),"route_reachable":len(reached),
            "structurally_unreachable":len(hard-reached)/len(hard) if hard else None,
            "unreachable_ids":[{"kind":k,"id":i} for k,i in sorted(hard-reached)],"supported_routes":names}


def admission_parity(old_rows, corrected_rows, *, require_support=True):
    """Check owner-produced traces, not inferred policies or substitute scores.

    Digests or explicitly encoded canonical values bind profile, eligibility,
    configuration and non-admission paths. Equal-admission controls must return
    equal content. run() also pins shared sources. This does not prove serving parity.

    audited_tag_gated_look_then_cut requires tag candidates, appends new look
    candidates, then applies a global rough-score cut. It is not tag-only or a
    full historical-ranker replay; the paired arms share current non-admission paths.
    """
    from experimental.evaluation.evaluate import digest, require
    require(len(old_rows) == len(corrected_rows) and old_rows, "admission parity requires paired contexts")
    controls = changes = 0
    invariants = ("profile", "eligibility", "configuration", "scoring", "selection", "images", "fallback", "explore", "control")
    for old,new in zip(old_rows,corrected_rows):
        require((old["context_id"],old["input_hash"],old["ranking_seed"]) ==
                (new["context_id"],new["input_hash"],new["ranking_seed"]), "admission parity input mismatch")
        a,b = old.get("admission_trace"),new.get("admission_trace")
        require(type(a) is dict and type(b) is dict, "admission contract mismatch: production admission_trace required")
        encoding = a.get("invariant_encoding", "sha256")
        require(encoding in ("sha256", "canonical_values_not_digests") and
                encoding == b.get("invariant_encoding", "sha256"), "unsupported or mismatched invariant encoding")
        require(a.get("policy") == "audited_tag_gated_look_then_cut" and b.get("policy") == "union",
                "first experiment must be audited_tag_gated_look_then_cut versus corrected union")
        for name in invariants:
            left,right = a.get("invariants",{}).get(name),b.get("invariants",{}).get(name)
            equal = (left is not None and right is not None and digest(left) == digest(right)
                     if encoding == "canonical_values_not_digests" else
                     type(left) is str and re.fullmatch(r"[0-9a-f]{64}",left) and left == right)
            require(equal,
                    "non-admission parity mismatch: " + name)
        require(type(a.get("admitted_ids")) is list and type(b.get("admitted_ids")) is list, "admitted ID trace required")
        if a["admitted_ids"] == b["admitted_ids"]:
            def content(rows):
                return [{**item,"explanation":{k:v for k,v in item["explanation"].items() if k not in ("sources","admission_policy")}}
                        for item in rows]
            require(digest(content(old["items"])) == digest(content(new["items"])), "equal-admission control changed ranking")
            controls += 1
        else:
            changes += 1
    supported = controls > 0 and changes > 0
    require(not require_support or supported, "admission experiment requires unchanged-admission controls and changed-admission contexts")
    return {"status":"owner_trace_parity_verified" if supported else "insufficient_context_support",
            "isolation_verified":supported,"unchanged_controls":controls,
            "changed_admission_contexts":changes,"invariant_paths":list(invariants),"serving_parity":"external_gate"}
