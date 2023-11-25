import pandas as pd
import numpy as np

CATEGORIES = {
    1: "participant",
    2: "pilot",
    3: "exclusion",
    4: "condition",
    5: "group",
    6: "task",
    7: "context",
}

METRICS = {
    1: "ID",
    2: "info",
    3: "description",
    4: "birthdate",
    5: "gender",
    6: "race",
    7: "ethnicity",
    8: "gestational age",
    9: "pregnancy term",
    10: "birth weight",
    11: "disability",
    12: "language",
    13: "country",
    14: "state",
    15: "setting",
    16: None,
    17: "name",
    18: "description",
    19: None,
    20: "name",
    21: "reason",
    22: "description",
    23: "name",
    24: "description",
    25: "info",
    26: "name",
    27: "description",
    28: "info",
    29: "name",
    30: "description",
    31: "info",
    32: "name",
    33: "setting",
    34: "language",
    35: "country",
    36: "state",
}

CATEGORIES_TO_METRICS = {
    1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    2: [17, 18],
    3: [20, 21, 22],
    4: [23, 24, 25],
    5: [26, 27, 28],
    6: [29, 30, 31],
    7: [32, 33, 34, 35, 36],
}

CATEGORIES_DOCSTRING = {
    1: "An individual human subject whose data are used or represented",
    2: "Indicates that the methods used were not finalized or were non-standard",
    3: "Indicates that data were not usable",
    4: "An experimenter-determined manipulation (within or between sessions)",
    5: "A grouping determined by an aspect of the data (participant ability, age, grade level, experience, longitudinal visit, measurements used/available)",
    6: "A particular task, activity, or phase of the session or study",
    7: "A particular setting or other aspect of where/when/how data were collected",
}

METRICS_DOCSTRING = {
    1: "A unique, anonymized, primary identifier, such as participant ID",
    2: "Other information or alternate identifier",
    3: "A longer explanation or description",
    4: "Date of birth (used with session date to calculate age; you can also use the group category to designate age groups)",
    5: '"Male", "Female", or any other relevant gender',
    6: "As classified by NIH, or user-defined classification",
    7: "As classified by NIH (Hispanic/Non-Hispanic), or user-defined classification",
    8: "Pregnancy age in weeks between last menstrual period and birth (or pre-natal observation)",
    9: '"Full term", "Preterm", or other gestational term (assumed "Full term" by default)',
    10: "Weight at birth (in grams, e.g., 3250)",
    11: 'Any developmental, physical, or mental disability or disabilities (assumed "typical" by default)',
    12: 'Primary language(s) spoken by and to participant (assumed "English" by default)',
    13: 'Country where participant was born (assumed "US" by default)',
    14: "State/territory where participant was born",
    15: "The physical context of the participant (please do not use for new data: see the context category instead)",
    16: None,
    17: "A label or identifier referring to the pilot method",
    18: "A longer explanation or description of the pilot method",
    19: None,
    20: "A label or identifier referring to the exclusion criterion",
    21: "The reason for excluding these data",
    22: "A longer explanation or description of the reason for excluding data",
    23: "A label or identifier for the condition",
    24: "A longer explanation or description of the condition",
    25: "Other information or alternate identifier",
    26: "A label or identifier for the grouping",
    27: "A longer explanation or description of the grouping",
    28: "Other information or alternate identifier",
    29: "A label or identifier for the task",
    30: "A longer explanation or description of the task",
    31: "Other information or alternate identifier",
    32: "A label or identifier for the context",
    33: "The physical context",
    34: "Language used in this context (assumed 'English' by default)",
    35: "Country of data collection (assumed 'US' by default)",
    36: "State/territory of data collection",
}


def build_record_df(record):
    cat_name = CATEGORIES[record["category"]]
    index = []
    vals = []
    for k in sorted(record["measures"], key=int):
        index.append((cat_name, METRICS[int(k)]))
        vals.append(record["measures"][k])
    if "age" in record:
        # this comes from build_sessions_df_row merging the "age" field
        # into the actual participant records
        index.append((cat_name, "age"))
        vals.append(record["age"])
    index = pd.MultiIndex.from_tuples(index)

    return pd.DataFrame(vals, index=index).T


def build_records_df(records):
    dfs = []
    context_counter = 1
    for record in sorted(records, key=lambda x: (int(x["category"]), int(x["id"]))):
        df = build_record_df(record)
        # there might be multiple contexts; rename them to deduplicate
        if CATEGORIES[record["category"]] == "context":
            new_col_idx = [
                (f"context{context_counter}", metric) for category, metric in df.columns
            ]
            df.columns = pd.MultiIndex.from_tuples(new_col_idx)
            context_counter += 1
        dfs.append(df)
    dfs.append(pd.DataFrame())  # placeholder in case empty
    return pd.concat(dfs, axis=1)


def build_session_df_row(container, volume_records):
    container_records = []
    for elem in container["records"]:
        rec_id = elem["id"]
        rec = list(filter(lambda x: x["id"] == rec_id, volume_records))[0]
        if "age" in elem:
            rec["age"] = elem["age"]
        container_records.append(rec)
    folder_df = pd.DataFrame(
        [
            container.get("name", np.nan),
            container.get("date", np.nan),
            len(container["assets"]),
        ]
    ).T
    folder_df.columns = pd.MultiIndex.from_tuples(
        [("folder", "name"), ("folder", "date"), ("folder", "files")]
    )
    records_df = build_records_df(container_records)
    return pd.concat([folder_df, records_df, pd.DataFrame()], axis=1)
