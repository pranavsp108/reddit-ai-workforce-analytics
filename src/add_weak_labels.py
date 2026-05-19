import re
from pathlib import Path

import pandas as pd


INPUT_PATH = "data/processed/reddit_comments_sentiment.csv"
FALLBACK_INPUT_PATH = "data/processed/reddit_comments_clean_features.csv"
OUTPUT_PATH = "data/processed/reddit_comments_nlp_features.csv"


# -----------------------------
# Regex helpers
# -----------------------------

def compile_patterns(patterns):
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def contains_any(text, patterns):
    text = "" if pd.isna(text) else str(text)
    return int(any(pattern.search(text) for pattern in patterns))


def first_matching_bucket(text, bucket_patterns):
    text = "" if pd.isna(text) else str(text)

    for bucket, patterns in bucket_patterns.items():
        if any(pattern.search(text) for pattern in patterns):
            return bucket

    return "unknown"


# -----------------------------
# Workforce disruption patterns
# -----------------------------

DISRUPTION_PATTERNS = {
    "job_displacement_flag": compile_patterns([
        r"\blost\s+(my|their|his|her|our)?\s*job\b",
        r"\blose\s+(my|their|his|her|our)?\s*job\b",
        r"\blosing\s+(my|their|his|her|our)?\s*job\b",
        r"\blaid\s+off\b",
        r"\blayoffs?\b",
        r"\bfired\b",
        r"\bterminated\b",
        r"\bunemployed\b",
        r"\bredundant\b",
        r"\blast\s+working\s+day\b",
        r"\bjob\s+loss\b",
        r"\blost\s+(my|their|his|her|our)?\s*(business|clients?|income|work)\b",
        r"\bwent\s+out\s+of\s+business\b",
    ]),

    "replacement_risk_flag": compile_patterns([
        r"\breplac(e|ed|ing|ement)\b",
        r"\btake\s+(my|our|their|his|her)?\s*job\b",
        r"\btaking\s+(my|our|their|his|her)?\s*job\b",
        r"\btake\s+over\b",
        r"\bautomate(d|s|ing)?\b",
        r"\bautomation\b",
        r"\bobsolete\b",
        r"\bno\s+longer\s+need\b",
        r"\bfewer\s+workers\b",
        r"\bone\s+person\s+.*\bdozens\b",
        r"\bai\s+worker\b",
        r"\brobots?\s+(taking|take|replace|replacing)\b",
    ]),

    "hiring_disruption_flag": compile_patterns([
        r"\bhiring\b",
        r"\brecruit(er|ers|ing)?\b",
        r"\bresume[s]?\b",
        r"\bcv\b",
        r"\bats\b",
        r"\bapplicant[s]?\b",
        r"\bapplication[s]?\b",
        r"\binterview[s]?\b",
        r"\bjob\s+posting[s]?\b",
        r"\bjob\s+market\b",
        r"\bghosted\b",
        r"\bkeyword\s+matching\b",
        r"\bchatgpt\s+.*resume\b",
        r"\bai-generated\s+resume\b",
        r"\bfake\s+applicant[s]?\b",
        r"\bprompt\s+injection\b",
    ]),

    "entry_level_pressure_flag": compile_patterns([
        r"\bentry[-\s]?level\b",
        r"\bnew\s+grad[s]?\b",
        r"\brecent\s+grad[s]?\b",
        r"\bjunior[s]?\b",
        r"\bintern[s]?\b",
        r"\bcollege\s+student[s]?\b",
        r"\bfirst\s+job\b",
        r"\bgraduate[s]?\b",
        r"\bno\s+experience\b",
    ]),

    "career_anxiety_flag": compile_patterns([
        r"\bscared\b",
        r"\bafraid\b",
        r"\bworried\b",
        r"\banxious\b",
        r"\bterrified\b",
        r"\bconcerned\b",
        r"\bdoomed\b",
        r"\bscrewed\b",
        r"\bpanic\b",
        r"\bwhat\s+should\s+i\s+do\b",
        r"\bnow\s+what\b",
        r"\bwhat\s+career\b",
        r"\bcareer\s+advice\b",
        r"\bcareer\s+path\b",
        r"\bfuture\s+proof\b",
        r"\bburned?\s+out\b",
    ]),

    "adaptation_reskilling_flag": compile_patterns([
        r"\breskill(ing)?\b",
        r"\bupskill(ing)?\b",
        r"\blearn(ing)?\b",
        r"\bpivot(ing)?\b",
        r"\bswitch(ing)?\s+career[s]?\b",
        r"\bcareer\s+change\b",
        r"\badapt(ing)?\b",
        r"\buse\s+ai\b",
        r"\busing\s+ai\b",
        r"\bai\s+tools?\b",
        r"\bwork\s+with\s+ai\b",
        r"\bembrace\s+ai\b",
    ]),

    "safe_jobs_flag": compile_patterns([
        r"\bsafe\s+job[s]?\b",
        r"\bjobs?\s+.*\bsafe\b",
        r"\bwon'?t\s+be\s+replaced\b",
        r"\bwill\s+not\s+be\s+replaced\b",
        r"\bleast\s+likely\s+to\s+be\s+replaced\b",
        r"\birreplaceable\b",
        r"\bfuture[-\s]?proof\b",
        r"\bblue\s+collar\b",
        r"\btrades?\b",
        r"\bmanual\s+labor\b",
        r"\bhands[-\s]?on\b",
    ]),

    "macro_economic_concern_flag": compile_patterns([
        r"\bubi\b",
        r"\buniversal\s+basic\s+income\b",
        r"\bunemployment\b",
        r"\beconom(y|ic|ics)\b",
        r"\bcapitalism\b",
        r"\bwages?\b",
        r"\bsalar(y|ies)\b",
        r"\bincome\b",
        r"\bcost\s+of\s+living\b",
        r"\bprofits?\b",
        r"\brevolution\b",
        r"\bstandard\s+of\s+living\b",
        r"\bwealth\b",
        r"\binequality\b",
        r"\bdepression\b",
        r"\bjobless\b",
    ]),

    "quality_trust_safety_flag": compile_patterns([
        r"\bquality\b",
        r"\bshitty\s+ai\b",
        r"\bbad\s+ai\b",
        r"\bhallucinat(e|es|ed|ing|ion|ions)\b",
        r"\bwrong\b",
        r"\bmistake[s]?\b",
        r"\berror[s]?\b",
        r"\bunsafe\b",
        r"\bsafety\b",
        r"\btrust\b",
        r"\breliable\b",
        r"\bregulatory\b",
        r"\blegal\s+risk\b",
        r"\bliability\b",
        r"\bdoesn'?t\s+know\b",
        r"\bnot\s+good\s+enough\b",
    ]),

    "corporate_cost_cutting_flag": compile_patterns([
        r"\bceo[s]?\b",
        r"\bexecutive[s]?\b",
        r"\bmanagement\b",
        r"\bshareholder[s]?\b",
        r"\bprofits?\b",
        r"\bcost[-\s]?cutting\b",
        r"\bcut\s+costs?\b",
        r"\breduce\s+headcount\b",
        r"\bheadcount\b",
        r"\bfree\s+up\s+capital\b",
        r"\bcapital\b",
        r"\bcheaper\b",
        r"\blabor\s+costs?\b",
        r"\bproductivity\b",
        r"\befficiency\b",
    ]),

    "ai_hype_skepticism_flag": compile_patterns([
        r"\bhype\b",
        r"\boverhyped\b",
        r"\bbubble\b",
        r"\bspeculation\b",
        r"\bnot\s+new\b",
        r"\bcan'?t\s+think\b",
        r"\bnot\s+intelligence\b",
        r"\bnot\s+actually\b",
        r"\bai\s+isn'?t\b",
        r"\bnot\s+ai\b",
        r"\bblaming\s+ai\b",
        r"\boutsourcing\b",
        r"\boffshor(e|ing)\b",
        r"\bindia\b",
        r"\bexecutives?\s+are\b",
    ]),

    "productivity_augmentation_flag": compile_patterns([
        r"\bproductivity\b",
        r"\bproductive\b",
        r"\bmake\s+.*\beasier\b",
        r"\bhelps?\s+.*\bwork\b",
        r"\btool\b",
        r"\bassist(ant|ed|s)?\b",
        r"\bcopilot\b",
        r"\bworkflow\b",
        r"\bautomate\s+tasks?\b",
        r"\baugment(s|ed|ing)?\b",
        r"\bdo\s+more\s+with\s+less\b",
    ]),
}


# -----------------------------
# Occupation patterns
# -----------------------------

OCCUPATION_PATTERNS = {
    "software_engineering_it": compile_patterns([
        r"\bsoftware\s+engineer[s]?\b",
        r"\bsoftware\s+developer[s]?\b",
        r"\bdeveloper[s]?\b",
        r"\bprogrammer[s]?\b",
        r"\bcoding\b",
        r"\bcode\b",
        r"\bweb\s+developer[s]?\b",
        r"\bfrontend\b",
        r"\bbackend\b",
        r"\bfull[-\s]?stack\b",
        r"\bdevops\b",
        r"\bsysadmin\b",
        r"\bsystem\s+admin\b",
        r"\bcybersecurity\b",
        r"\binformation\s+technology\b",
        r"\bit\s+field\b",
        r"\bit\s+job[s]?\b",
        r"\bit\s+support\b",
        r"\bhelp\s+desk\b",
    ]),

    "data_ai_analytics": compile_patterns([
        r"\bdata\s+scientist[s]?\b",
        r"\bdata\s+analyst[s]?\b",
        r"\banalytics\b",
        r"\bbusiness\s+intelligence\b",
        r"\bbi\s+analyst[s]?\b",
        r"\bsql\b",
        r"\bmachine\s+learning\b",
        r"\bml\s+engineer[s]?\b",
        r"\bai\s+engineer[s]?\b",
        r"\bdata\s+engineer[s]?\b",
        r"\bmodeler[s]?\b",
    ]),

    "creative_content_media": compile_patterns([
        r"\bwriter[s]?\b",
        r"\bcopywriter[s]?\b",
        r"\bcontent\s+writer[s]?\b",
        r"\beditor[s]?\b",
        r"\bjournalist[s]?\b",
        r"\bblogger[s]?\b",
        r"\bseo\b",
        r"\btechnical\s+writer[s]?\b",
        r"\bdesigner[s]?\b",
        r"\bgraphic\s+design(er)?[s]?\b",
        r"\bartist[s]?\b",
        r"\billustrator[s]?\b",
        r"\bcreative[s]?\b",
        r"\bux\b",
        r"\bui\b",
        r"\bvoice\s+actor[s]?\b",
        r"\bvoice[-\s]?over\b",
        r"\bnarrator[s]?\b",
        r"\bvideo\s+editor[s]?\b",
        r"\bmedia\b",
        r"\byoutube\b",
        r"\bdubbing\b",
    ]),

    "business_operations_admin": compile_patterns([
        r"\bbusiness\s+analyst[s]?\b",
        r"\boperations\b",
        r"\badmin\b",
        r"\badministrative\b",
        r"\bproject\s+manager[s]?\b",
        r"\bproduct\s+manager[s]?\b",
        r"\bprogram\s+manager[s]?\b",
        r"\bconsultant[s]?\b",
        r"\bhr\b",
        r"\bhuman\s+resources\b",
        r"\bcoordinator[s]?\b",
    ]),

    "customer_sales_support": compile_patterns([
        r"\bcustomer\s+service\b",
        r"\bcustomer\s+support\b",
        r"\bcall\s+center\b",
        r"\bchat\s+support\b",
        r"\bsales\b",
        r"\bsales\s+rep\b",
        r"\baccount\s+manager[s]?\b",
        r"\bhelp\s+desk\b",
    ]),

    "education_research": compile_patterns([
        r"\bteacher[s]?\b",
        r"\bprofessor[s]?\b",
        r"\btutor[s]?\b",
        r"\beducation\b",
        r"\bschool\b",
        r"\bcollege\b",
        r"\buniversity\b",
        r"\bstudent[s]?\b",
        r"\bresearcher[s]?\b",
        r"\bacademic[s]?\b",
    ]),

    "healthcare_social_services": compile_patterns([
        r"\bdoctor[s]?\b",
        r"\bnurse[s]?\b",
        r"\bhealthcare\b",
        r"\bmedical\b",
        r"\btherapist[s]?\b",
        r"\bpsycholog(y|ist|ists)\b",
        r"\bsocial\s+worker[s]?\b",
        r"\bclinic\b",
        r"\bhospital[s]?\b",
        r"\bdiagnos(is|e|es)\b",
    ]),

    "legal_finance_accounting": compile_patterns([
        r"\blawyer[s]?\b",
        r"\blegal\b",
        r"\bparalegal[s]?\b",
        r"\battorney[s]?\b",
        r"\bfinance\b",
        r"\bfinancial\b",
        r"\baccountant[s]?\b",
        r"\baccounting\b",
        r"\bbanking\b",
        r"\binsurance\b",
        r"\bunderwriting\b",
        r"\binvestment\s+banker[s]?\b",
    ]),

    "trades_manual_labor": compile_patterns([
        r"\bblue\s+collar\b",
        r"\btrade[s]?\b",
        r"\belectrician[s]?\b",
        r"\bplumber[s]?\b",
        r"\bmechanic[s]?\b",
        r"\bconstruction\b",
        r"\bcarpenter[s]?\b",
        r"\bweld(er|ers|ing)\b",
        r"\bmanual\s+labor\b",
        r"\bskilled\s+labor\b",
        r"\bfactory\s+worker[s]?\b",
    ]),

    "manufacturing_agriculture_logistics": compile_patterns([
        r"\bfactory\b",
        r"\bmanufacturing\b",
        r"\bwarehouse\b",
        r"\blogistics\b",
        r"\btruck\s+driver[s]?\b",
        r"\bdriver[s]?\b",
        r"\bagriculture\b",
        r"\bfarm(er|ers|ing)?\b",
        r"\bcrop[s]?\b",
    ]),

    "management_executive": compile_patterns([
        r"\bmanager[s]?\b",
        r"\bmanagement\b",
        r"\bceo[s]?\b",
        r"\bexecutive[s]?\b",
        r"\bdirector[s]?\b",
        r"\bvp\b",
        r"\bchief\b",
    ]),

    "general_white_collar": compile_patterns([
        r"\bwhite\s+collar\b",
        r"\boffice\s+job[s]?\b",
        r"\bdesk\s+job[s]?\b",
        r"\bknowledge\s+work(er|ers)?\b",
        r"\bcorporate\b",
        r"\bemail\s+job[s]?\b",
    ]),
}


# -----------------------------
# Primary category assignment
# -----------------------------

def assign_primary_disruption_category(row):
    """
    Assigns one primary category while preserving all individual flags.
    Priority matters. Direct real-world impact comes first.
    """

    if row["job_displacement_flag"]:
        return "direct_displacement"

    if row["hiring_disruption_flag"]:
        return "hiring_market_disruption"

    if row["entry_level_pressure_flag"]:
        return "entry_level_pressure"

    if row["replacement_risk_flag"]:
        return "replacement_risk"

    if row["safe_jobs_flag"]:
        return "safe_jobs_discussion"

    if row["career_anxiety_flag"]:
        return "career_anxiety"

    if row["adaptation_reskilling_flag"]:
        return "adaptation_reskilling"

    if row["quality_trust_safety_flag"]:
        return "quality_trust_safety_concern"

    if row["corporate_cost_cutting_flag"]:
        return "corporate_cost_cutting"

    if row["macro_economic_concern_flag"]:
        return "macro_economic_concern"

    if row["ai_hype_skepticism_flag"]:
        return "ai_hype_skepticism"

    if row["productivity_augmentation_flag"]:
        return "productivity_augmentation"

    return "general_discussion"


def assign_impact_severity(row):
    """
    Severity is mainly based on comment-level evidence.
    Post-title fallback can raise weak/general comments only to moderate levels, not direct displacement.
    """

    # Strongest evidence: comment itself says job/client/business loss
    if row.get("comment_job_displacement_flag", 0):
        return 5

    # Strong replacement + anxiety or entry-level concern
    if row.get("comment_replacement_risk_flag", 0) and (
        row.get("comment_career_anxiety_flag", 0)
        or row.get("comment_entry_level_pressure_flag", 0)
    ):
        return 4

    # Hiring disruption or entry-level concern
    if row.get("comment_hiring_disruption_flag", 0) or row.get("comment_entry_level_pressure_flag", 0):
        return 4

    # Replacement, corporate cost-cutting, macro concern
    if (
        row.get("comment_replacement_risk_flag", 0)
        or row.get("comment_corporate_cost_cutting_flag", 0)
        or row.get("comment_macro_economic_concern_flag", 0)
    ):
        return 3

    # Softer concern/adaptation/safety discussion
    if (
        row.get("comment_career_anxiety_flag", 0)
        or row.get("comment_quality_trust_safety_flag", 0)
        or row.get("comment_safe_jobs_flag", 0)
        or row.get("comment_adaptation_reskilling_flag", 0)
    ):
        return 2

    # If only the post title gives context, cap severity at 2 or 3
    if row.get("label_source", "") == "post_title_fallback":
        if row.get("replacement_risk_flag", 0) or row.get("job_displacement_flag", 0):
            return 3
        return 2

    return 1


def main():
    input_path = Path(INPUT_PATH)

    if not input_path.exists():
        input_path = Path(FALLBACK_INPUT_PATH)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH} or {FALLBACK_INPUT_PATH}"
        )

    df = pd.read_csv(input_path)

    # Preserve original text fields
    df["comment_body"] = df["comment_body"].fillna("").astype(str)

    if "post_title" in df.columns:
        df["post_title"] = df["post_title"].fillna("").astype(str)
    else:
        df["post_title"] = ""

    # Separate comment-level signal from post-title context.
    # This prevents post titles like "People who lost jobs to AI..." from labeling every comment as direct displacement.
    df["comment_analysis_text"] = df["comment_body"].fillna("").astype(str)
    df["post_context_text"] = df["post_title"].fillna("").astype(str)

    # Occupation detection should primarily use comment text.
    # This avoids assigning every comment in a post the occupation implied by the title.
    df["occupation_text"] = df["comment_body"].fillna("").astype(str)

    # Comment-level disruption flags
    for flag, patterns in DISRUPTION_PATTERNS.items():
        comment_flag = "comment_" + flag
        df[comment_flag] = df["comment_analysis_text"].apply(lambda x: contains_any(x, patterns))

    # Post-title context flags
    for flag, patterns in DISRUPTION_PATTERNS.items():
        post_flag = "post_context_" + flag
        df[post_flag] = df["post_context_text"].apply(lambda x: contains_any(x, patterns))

    # For backward-compatible final flags:
    # Use comment-level signal first. Only use post title when comment has no workforce signal.
    comment_flag_cols = ["comment_" + flag for flag in DISRUPTION_PATTERNS.keys()]
    post_flag_cols = ["post_context_" + flag for flag in DISRUPTION_PATTERNS.keys()]

    df["comment_has_workforce_signal"] = (df[comment_flag_cols].sum(axis=1) > 0).astype(int)

    for flag in DISRUPTION_PATTERNS.keys():
        comment_flag = "comment_" + flag
        post_flag = "post_context_" + flag

        df[flag] = df[comment_flag]

        fallback_mask = (df["comment_has_workforce_signal"] == 0) & (df[post_flag] == 1)
        df.loc[fallback_mask, flag] = 1

    df["label_source"] = "comment"
    df.loc[df["comment_has_workforce_signal"] == 0, "label_source"] = "post_title_fallback"

    # Occupation bucket
    df["occupation_bucket"] = df["occupation_text"].apply(
        lambda x: first_matching_bucket(x, OCCUPATION_PATTERNS)
    )

    # Optional fallback: if comment has no occupation but title has clear context,
    # use title-level context. This keeps unknowns but reduces obvious misses.
    title_occ = df["post_title"].apply(
        lambda x: first_matching_bucket(x, OCCUPATION_PATTERNS)
    )

    df["occupation_bucket_source"] = "comment"

    fallback_mask = (df["occupation_bucket"] == "unknown") & (title_occ != "unknown")
    df.loc[fallback_mask, "occupation_bucket"] = title_occ[fallback_mask]
    df.loc[fallback_mask, "occupation_bucket_source"] = "post_title"

    # Primary category and severity
    df["primary_disruption_category"] = df.apply(
        assign_primary_disruption_category, axis=1
    )

    df["impact_severity"] = df.apply(assign_impact_severity, axis=1)

    # Useful binary consolidation flags
    df["any_workforce_risk_flag"] = (
        df[
            [
                "job_displacement_flag",
                "replacement_risk_flag",
                "hiring_disruption_flag",
                "entry_level_pressure_flag",
                "career_anxiety_flag",
                "macro_economic_concern_flag",
                "corporate_cost_cutting_flag",
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)

    df["any_adaptation_or_safe_jobs_flag"] = (
        df[
            [
                "adaptation_reskilling_flag",
                "safe_jobs_flag",
                "productivity_augmentation_flag",
            ]
        ].sum(axis=1)
        > 0
    ).astype(int)

    # Primary category using final fallback-aware flags
    df["primary_disruption_category"] = df.apply(
        assign_primary_disruption_category, axis=1
    )

    # Comment-only category for audit/debugging
    original_flags = {}

    for flag in DISRUPTION_PATTERNS.keys():
        original_flags[flag] = df[flag].copy()
        df[flag] = df["comment_" + flag]

    df["comment_primary_disruption_category"] = df.apply(
        assign_primary_disruption_category, axis=1
    )

    # Restore final fallback-aware flags
    for flag in DISRUPTION_PATTERNS.keys():
        df[flag] = original_flags[flag]

    # Engagement target
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        engagement_threshold = df["score"].quantile(0.90)
        df["high_engagement"] = (df["score"] >= engagement_threshold).astype(int)
        df["engagement_threshold_90p"] = engagement_threshold

    # Drop helper text columns if you do not want them in the final file
    # Keeping them can help debugging, but they are not needed for final modeling.
    # df = df.drop(columns=["analysis_text", "occupation_text"])

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nPrimary disruption categories:")
    print(df["primary_disruption_category"].value_counts())

    print("\nOccupation buckets:")
    print(df["occupation_bucket"].value_counts())

    print("\nImpact severity:")
    print(df["impact_severity"].value_counts().sort_index())

    if "vader_sentiment_label" in df.columns:
        print("\nVADER sentiment:")
        print(df["vader_sentiment_label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()