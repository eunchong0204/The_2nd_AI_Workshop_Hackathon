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
        "A genuine sign, symptom, diagnosis, or injury experienced by the patient "
        "(e.g. Pancreatitis, Rash, Seizure, Hepatic failure)."
    ),
    "LAB_OR_INVESTIGATION": (
        "A laboratory test, medical imaging, or vital-sign measurement, "
        "including a change or abnormality in its result "
        "(e.g. Blood glucose increased, Liver function test abnormal)."
    ),
    "LACK_OF_EFFICACY": (
        "Treatment failure, recurrence due to lack of effect, or worsening/progression "
        "of the condition being treated (e.g. Drug ineffective, Condition aggravated)."
    ),
    "MEDICATION_ERROR_OR_MISUSE": (
        "A medication error, accidental or intentional misuse, abuse, overdose, "
        "underdose, inappropriate use, or incorrect administration (e.g. Accidental "
        "overdose, Off label use)."
    ),
    "PRODUCT_OR_DEVICE_ISSUE": (
        "A quality complaint, malfunction, defect, packaging issue, or other problem "
        "involving the medicinal product or device itself, without specifying a patient "
        "injury or clinical reaction (e.g. Device malfunction, Product quality issue)."
    ),
    "ADMINISTRATION_SITE_OR_ROUTE": (
        "A clinical reaction specifically localized to an administration, infusion, "
        "injection, application, implantation, or instillation site (e.g. Injection "
        "site pain, Extravasation)."
    ),
    "NONSPECIFIC_OR_CONTEXT": (
        "A concept that is not intrinsically a patient clinical event and does not fit "
        "another category, including procedures, treatment or route context, drug-level "
        "monitoring, hospitalization context, administrative terms, or explicit "
        "no-event terms, and vague non-specific descriptors (e.g. Drug level "
        "increased, Dialysis, Drug intolerance, No adverse event)."
    ),
}

# Categories retained as candidate adverse reactions; everything else is blocked.
KEEP = {"CLINICAL_EVENT", "LAB_OR_INVESTIGATION"}

# Where to file a PT the model returns with an unknown/blank category.
CATCH_ALL = "NONSPECIFIC_OR_CONTEXT"


################
# 3. Prompt text
################
_ANALYST = "You are a senior pharmacovigilance analyst and MedDRA expert."

# Formal grounding so the model reasons from standards, not vibes. (ICH E2A.)
## Definitions from https://database.ich.org/sites/default/files/E2A_Guideline.pdf
AE_ADR_DEFINITIONS = """\
Definitions:
- Adverse Event (AE): any untoward medical occurrence in a patient administered a
  medicinal product, whether or not causally related to treatment.
- Adverse Drug Reaction (ADR): a noxious and unintended response for which a causal
  relationship with the medicinal product is at least a reasonable possibility.
- MedDRA Preferred Term (PT): one coded medical concept recorded in a FAERS report."""

# Why we are classifying: FAERS mixes reactions with many non-reaction terms.
FRAMING = """\
I have a list of PTs obtained from FAERS reports. FAERS records reported adverse
events, which are not necessarily adverse drug reactions or causally related to a
drug. I want to identify PTs that could plausibly represent an adverse drug reaction
to some drug."""

GOAL = """\
Classify each PT into exactly one of the categories above according to what the PT
intrinsically represents."""

# Rendered as a numbered "Classification rules:" list in the system prompt.
CLASSIFICATION_RULES = [
    "Make the classification drug-agnostic. Do not consider whether the PT is "
    "associated with or plausible for a particular drug.",
    "Classify the PT itself, not a possible cause, consequence, or associated "
    "condition.",
    "If a PT could fit multiple categories, select the category that most directly "
    "describes the PT's primary MedDRA concept.",
    f"If none of the categories clearly applies, use {CATCH_ALL}.",
    "Include every input PT exactly once and preserve the original order and spelling.",
]

OUTPUT_CONTRACT = """\
Respond only with valid JSON in this exact structure:
{"results":[{"pt":"<verbatim>","pt_type":"<CATEGORY>"}]}
Do not include explanations, Markdown, comments, additional keys, or text outside the
JSON."""


####################
# 4. Prompt builders
####################
def classification_system_prompt() -> str:
    lines = [
        _ANALYST,
        "",
        AE_ADR_DEFINITIONS,
        "",
        "Background:",
        FRAMING,
        "",
        "Based on an initial review of the PTs, I created the following categories to help determine which PTs could plausibly represent ADRs.",
        "Use only these categories:",
        "",
    ]
    for name, desc in CATEGORIES.items():
        lines.append(f"- {name}: {desc}")
    lines += ["", "Goal:", GOAL, "", "Classification rules:"]
    for i, rule in enumerate(CLASSIFICATION_RULES, 1):
        lines.append(f"{i}. {rule}")
    lines += ["", OUTPUT_CONTRACT]
    return "\n".join(lines)


def classification_user_prompt(pts: list[str]) -> str:
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(pts, 1))
    return f"Classify the following {len(pts)} PTs:\n{numbered}"
