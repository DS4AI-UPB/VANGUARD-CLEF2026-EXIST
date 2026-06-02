class Task21:
    LABELS = ["NO", "YES"]
    NUM_CLASSES = 2


class Task22:
    LABELS = ["NO", "DIRECT", "JUDGEMENTAL"]
    LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
    NUM_CLASSES = 3
    HIERARCHY = {"YES": ["DIRECT", "JUDGEMENTAL"], "NO": []}


class Task23:
    LABELS = [
        "NO",
        "IDEOLOGICAL-INEQUALITY",
        "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"
    ]
    LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
    NUM_CLASSES = 6
    HIERARCHY = {
        "YES": [
            "IDEOLOGICAL-INEQUALITY",
            "STEREOTYPING-DOMINANCE", "OBJECTIFICATION", "SEXUAL-VIOLENCE", "MISOGYNY-NON-SEXUAL-VIOLENCE"
        ],
        "NO": []
    }


class Config:
    SKIP_LABELS = {"-", "UNKNOWN", ""}
    TEST_CASE = "EXIST2025"
    HARD_THRESHOLD_2_1 = 3
    HARD_THRESHOLD_2_2 = 2
    HARD_THRESHOLD_2_3 = 1
