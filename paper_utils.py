import pandas as pd
from scipy import stats


def p_values(controls_f, patients_f):
    controls = pd.read_csv(controls_f)
    patients = pd.read_csv(patients_f)

    p_value = lambda field: stats.ttest_ind(controls[field], patients[field])

    return dict(age=p_value('age').pvalue, gender=p_value('isFemale').pvalue)