"""
Step 2 taxonomy + classification prompts.

Step 2 makes ONE model call per batch: every PT is assigned to exactly one of the
seven fixed intrinsic categories below, judged drug-agnostically (by what the term
intrinsically IS, not whether it's plausible for a given drug). The two categories
in KEEP (CLINICAL_EVENT, LAB_OR_INVESTIGATION) are retained as candidate adverse
reactions; the other five form the blocklist.
"""



###########
# 1. Set-up
###########
from __future__ import annotations


#############
# 2. Taxonomy
#############
# Fixed taxonomy: category name -> description shown to the model in the prompt.
CATEGORIES = {
    "CLINICAL_EVENT": (
        "A genuine clinical sign, symptom, diagnosis, or injury actually experienced "
        "by the patient (e.g. Pancreatitis, Rash, Seizure, Hepatic failure)."
    ),
    "LAB_OR_INVESTIGATION": (
        "The name of a laboratory test, imaging, or vital-sign measurement, or a "
        "qualitative/quantitative result of one (e.g. Blood glucose increased, "
        "Liver function test abnormal, Ejection fraction decreased)."
    ),
    "LACK_OF_EFFICACY": (
        "The drug not working: treatment failure, disease progression, or condition "
        "worsening (e.g. Drug ineffective, Condition aggravated, Disease progression)."
    ),
    "MEDICATION_ERROR_OR_MISUSE": (
        "Errors, misuse, abuse, overdose, off-label or wrong administration -- not a "
        "reaction to a correctly-used drug (e.g. Accidental overdose, Off label use, "
        "Wrong technique in product usage process)."
    ),
    "PRODUCT_OR_DEVICE_ISSUE": (
        "A complaint about the product or device itself, not a patient reaction "
        "(e.g. Device malfunction, Product quality issue, Needle issue)."
    ),
    "ADMINISTRATION_SITE_OR_ROUTE": (
        "A local reaction at the injection/infusion/application/implant site, or an "
        "administration-route/context term -- tied to how the drug was given rather "
        "than a systemic reaction (e.g. Injection site reaction, Infusion site pain, "
        "Application site erythema, Extravasation)."
    ),
    "NONSPECIFIC_OR_CONTEXT": (
        "Everything intrinsically NOT a patient clinical event: therapeutic drug-level "
        "monitoring, procedure/therapy/hospitalisation context, and administrative/"
        "no-event terms -- plus anything you cannot confidently classify (e.g. Drug level "
        "increased, Dialysis, Hospitalisation, Drug exposure, No adverse event)."
    ),
}

# Categories retained as candidate adverse reactions; everything else is blocked.
KEEP = {"CLINICAL_EVENT", "LAB_OR_INVESTIGATION"}

# Where to file a PT the model returns with an unknown/blank category.
CATCH_ALL = "NONSPECIFIC_OR_CONTEXT"


################
# 3. Prompt text
################
# Formal grounding so the model reasons from standards, not vibes. (ICH E2A.)
## Definitions from https://database.ich.org/sites/default/files/E2A_Guideline.pdf
AE_ADR_DEFINITIONS = """\
Definitions (ICH E2A):
- Adverse Event (AE): any untoward medical occurrence in a patient administered a
  medicinal product that does NOT necessarily have a causal relationship with the
  treatment.
- Adverse (Drug) Reaction (ADR): a response to a medicinal product that is noxious
  and unintended, where a causal relationship with the product is at least a
  reasonable possibility.
A MedDRA Preferred Term (PT) is one coded medical concept used to record what was
reported. FAERS (the FDA Adverse Event Reporting System) is a spontaneous
adverse-event reporting database, but each PT on a report is either a genuine
clinical event/reaction the patient experienced, or one of many NON-reaction
terms -- lab/test names, product or device complaints, medication errors,
procedures, and vague administrative terms."""

GOAL = """\
Goal: pharmacovigilance signal detection. We want to retain only PTs that could
plausibly be a genuine adverse reaction to some drug. KEEP PTs denoting a clinical
event/sign/symptom/diagnosis/injury actually experienced by a patient, and
lab/investigation findings (which can be the earliest signal of a reaction).
DISCARD PTs that are intrinsically NOT a patient clinical event.
This judgement is DRUG-AGNOSTIC: classify each term by what it intrinsically IS,
never by whether it is plausible for a particular drug."""

_ANALYST = "You are a senior pharmacovigilance analyst and MedDRA expert."


####################
# 4. Prompt builders
####################
def classification_system_prompt() -> str:
    lines = [
        _ANALYST,
        "",
        AE_ADR_DEFINITIONS,
        "",
        GOAL,
        "",
        "Classify each MedDRA Preferred Term (PT) into EXACTLY ONE category below,",
        "judging only what the term intrinsically IS (drug-agnostic).",
        "",
        "CATEGORIES:",
    ]
    for name, desc in CATEGORIES.items():
        lines.append(f"- {name}: {desc}")
    lines += [
        "",
        f"If a term fits none, use {CATCH_ALL}.",
        'Respond ONLY as JSON: {"results": [{"pt": "<verbatim>", "pt_type": "<CATEGORY>"}]}.',
        "Echo each PT verbatim and include every PT exactly once.",
    ]
    return "\n".join(lines)


def classification_user_prompt(pts: list[str]) -> str:
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(pts, 1))
    return f"Classify these {len(pts)} PTs:\n{numbered}"
